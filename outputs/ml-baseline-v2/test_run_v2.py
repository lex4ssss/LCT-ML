import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import duckdb
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = Path(os.environ.get('RUN_V2_SCRIPT', HERE / 'run_v2.py'))
FEATURES = ['event_count_24h', 'alarm_count_24h', 'event_count_7d', 'alarm_count_7d', 'event_count_30d', 'alarm_count_30d', 'active_days_30d', 'episode_starts_7d', 'episode_starts_30d', 'last_event_age_hours', 'last_alarm_age_hours_30d', 'channel_age_days', 'log_gap_hours_30d']
DAY = np.timedelta64(86400 * 10**6, 'us')


def synthetic(seed, flip_2025=False):
    rng = np.random.default_rng(seed)
    rows = []
    for channel in range(40):
        for day in range(0, 1000, 2):
            as_of = np.datetime64('2023-05-01', 'us') + day * DAY
            alarm24 = int(rng.random() < 0.25)
            common = rng.random() < 0.7
            label = int(rng.random() < (0.45 if alarm24 else 0.06))
            split = 'train' if as_of < np.datetime64('2025-01-01') else 'validation' if as_of < np.datetime64('2026-01-01') else 'test'
            if flip_2025 and split == 'validation':
                label = 1 - label
            values = dict(zip(FEATURES, rng.integers(0, 30, len(FEATURES)).astype(float)))
            values.update(event_count_24h=float(rng.integers(1, 20)) if common else 0.0, alarm_count_24h=float(alarm24 and common),
                          alarm_count_7d=float(alarm24 or rng.random() < 0.2), last_event_age_hours=float(rng.uniform(0, 700)),
                          last_alarm_age_hours_30d=None if rng.random() < 0.5 else float(rng.uniform(0, 700)))
            excluded = rng.random() < 0.05
            rows.append((str(channel), as_of.astype(object), *[values[f] for f in FEATURES], split, None if excluded else label, 'active_alarm' if excluded else None))
    return rows


def write_inputs(directory, seed, flip_2025=False):
    rows = synthetic(seed, flip_2025)
    examples, episodes = directory / 'examples.parquet', directory / 'episodes.parquet'
    with duckdb.connect() as c:
        c.execute('CREATE TABLE t(channel_id VARCHAR, as_of TIMESTAMP, ' + ', '.join(f + ' DOUBLE' for f in FEATURES) + ', split VARCHAR, label INTEGER, exclusion_reason VARCHAR)')
        c.executemany('INSERT INTO t VALUES (' + ','.join(['?'] * (len(FEATURES) + 5)) + ')', rows)
        c.execute(f"COPY t TO '{examples}' (FORMAT PARQUET)")
        c.execute("CREATE TABLE e AS SELECT channel_id, as_of + INTERVAL '5 hours' AS first_alarm_at, 'observed_quiet_gap' AS left_boundary FROM t WHERE label = 1 UNION ALL SELECT channel_id, as_of + INTERVAL '7 hours', 'observed_quiet_gap' FROM t WHERE label = 1 AND hash(channel_id || as_of::VARCHAR) % 3 = 0 UNION ALL SELECT channel_id, as_of + INTERVAL '30 hours', 'observed_quiet_gap' FROM t WHERE exclusion_reason IS NOT NULL")
        c.execute(f"COPY e TO '{episodes}' (FORMAT PARQUET)")
    return rows, examples, episodes


def run(examples, episodes, output):
    return subprocess.run([sys.executable, str(SCRIPT), str(examples), str(episodes), str(output)], capture_output=True, text=True, cwd=HERE, timeout=900)


class RunV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.rows, examples, episodes = write_inputs(root, 5)
        result = run(examples, episodes, root / 'out')
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.report = json.loads((root / 'out' / 'report.json').read_text())
        cls.examples, cls.episodes, cls.root = examples, episodes, root
        flipped_dir = root / 'flipped'
        flipped_dir.mkdir()
        _, fe, fp = write_inputs(flipped_dir, 5, True)
        result = run(fe, fp, flipped_dir / 'out')
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.flipped = json.loads((flipped_dir / 'out' / 'report.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_existing_output_refused(self):
        result = run(self.examples, self.episodes, self.root / 'out')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('FileExistsError', result.stderr)

    def test_structure_flags_and_models(self):
        self.assertEqual(set(self.report['candidates']), {'rule_alarm_24h', 'rule_alarm_7d', 'logistic_v1', 'logistic_v2', 'hgb'})
        self.assertIs(self.report['flags']['test_evaluation'], False)
        for name in ('logistic_v1', 'logistic_v2', 'hgb'):
            self.assertTrue((self.root / 'out' / (name + '.joblib')).exists())
            self.assertIn('calibration', self.report['candidates'][name])

    def test_rule_metrics_match_direct_computation(self):
        labeled = [r for r in self.rows if r[-2] is not None and r[-3] != 'test']
        in_2025 = [r for r in labeled if r[1].year == 2025]
        in_2024 = [r for r in labeled if r[1].year == 2024]
        alarm = FEATURES.index('alarm_count_24h') + 2
        best = None
        for threshold in sorted({float(r[alarm] > 0) for r in in_2024}):
            tp = sum(1 for r in in_2024 if float(r[alarm] > 0) >= threshold and r[-2] == 1)
            fp = sum(1 for r in in_2024 if float(r[alarm] > 0) >= threshold and r[-2] == 0)
            fn = sum(1 for r in in_2024 if r[-2] == 1) - tp
            f1 = 2 * tp / (2 * tp + fp + fn)
            if best is None or f1 >= best[1]:
                best = (threshold, f1)
        rule = self.report['candidates']['rule_alarm_24h']
        self.assertEqual(rule['threshold_A'], best[0])
        frozen = rule['frozen']['all']
        tp = sum(1 for r in in_2025 if float(r[alarm] > 0) >= best[0] and r[-2] == 1)
        fp = sum(1 for r in in_2025 if float(r[alarm] > 0) >= best[0] and r[-2] == 0)
        self.assertEqual((frozen['tp'], frozen['fp'], frozen['rows']), (tp, fp, len(in_2025)))
        common_rows = sum(1 for r in in_2025 if r[2] > 0)
        self.assertEqual(rule['frozen']['common']['rows'], common_rows)

    def test_episode_coverage_counts(self):
        labeled_2025 = {(r[0], r[1]) for r in self.rows if r[-2] is not None and r[1].year == 2025}
        with duckdb.connect() as c:
            keys = c.execute(f"SELECT channel_id, date_trunc('day', first_alarm_at - INTERVAL '1 microsecond') FROM '{self.episodes}' WHERE first_alarm_at > TIMESTAMP '2025-01-01' AND first_alarm_at <= TIMESTAMP '2026-01-01'").fetchall()
        coverage = self.report['candidates']['logistic_v2']['coverage']
        self.assertEqual(coverage['episodes_total'], len(keys))
        self.assertEqual(coverage['episodes_with_eligible_snapshot'], sum(1 for k in keys if k in labeled_2025))
        self.assertLessEqual(coverage['episodes_in_alerted_windows'], coverage['episodes_with_eligible_snapshot'])

    def test_alerted_episodes_use_fold_b_model(self):
        import joblib
        sys.path.insert(0, str(HERE))
        import experiment_v2 as ex
        eligible = [r for r in self.rows if r[-2] is not None and r[1].year == 2025]
        columns = {name: np.array([np.nan if r[FEATURES.index(name) + 2] is None else r[FEATURES.index(name) + 2] for r in eligible], dtype=float) for name in FEATURES}
        columns['as_of'] = np.array([np.datetime64(r[1], 'us') for r in eligible])
        scores = joblib.load(self.root / 'out' / 'logistic_v2.joblib').predict_proba(ex.matrix_v2(columns, False))[:, 1]
        candidate = self.report['candidates']['logistic_v2']
        alerted = {(r[0], r[1]) for r, s in zip(eligible, scores) if s >= candidate['threshold_A']}
        with duckdb.connect() as c:
            keys = c.execute(f"SELECT channel_id, date_trunc('day', first_alarm_at - INTERVAL '1 microsecond') FROM '{self.episodes}' WHERE first_alarm_at > TIMESTAMP '2025-01-01' AND first_alarm_at <= TIMESTAMP '2026-01-01'").fetchall()
        self.assertEqual(candidate['coverage']['episodes_in_alerted_windows'], sum(1 for k in keys if k in alerted))

    def test_2025_labels_do_not_affect_2024_decisions(self):
        for name, candidate in self.report['candidates'].items():
            with self.subTest(name=name):
                self.assertEqual(candidate['threshold_A'], self.flipped['candidates'][name]['threshold_A'])
                self.assertEqual(candidate['selection_2024'], self.flipped['candidates'][name]['selection_2024'])
                self.assertNotEqual(candidate['frozen']['all']['tp'], self.flipped['candidates'][name]['frozen']['all']['tp'])

    def test_calibration_uses_only_2024_labels(self):
        for name in ('logistic_v1', 'logistic_v2', 'hgb'):
            with self.subTest(name=name):
                normal, flipped = self.report['candidates'][name]['calibration'], self.flipped['candidates'][name]['calibration']
                self.assertEqual([(b['lower'], b['rows'], b['mean_score']) for b in normal['reliability_after']],
                                 [(b['lower'], b['rows'], b['mean_score']) for b in flipped['reliability_after']])
                self.assertAlmostEqual(normal['brier_before'], self.report['candidates'][name]['frozen']['all']['brier'], places=12)

    def test_logistic_v1_trained_on_common_train_rows(self):
        import joblib
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, StandardScaler
        i = {name: FEATURES.index(name) + 2 for name in FEATURES}
        train = [r for r in self.rows if r[-2] is not None and r[1] < np.datetime64('2024-12-31T00:00:00', 'us').astype(object) and r[i['event_count_24h']] > 0]
        alarm_age = lambda r: r[i['last_alarm_age_hours_30d']] if r[i['last_alarm_age_hours_30d']] is not None and r[i['last_alarm_age_hours_30d']] < 24 else np.nan
        x = np.array([[r[i['event_count_24h']], r[i['alarm_count_24h']], r[i['last_event_age_hours']], alarm_age(r)] for r in train], dtype=float)
        y = np.array([r[-2] for r in train])
        order = np.lexsort(([r[1].timestamp() for r in train], [r[0] for r in train]))
        reference = make_pipeline(SimpleImputer(strategy='constant', fill_value=24), FunctionTransformer(np.log1p), StandardScaler(),
                                  LogisticRegression(C=1, max_iter=500, tol=1e-6, solver='lbfgs')).fit(x[order], y[order])
        saved = joblib.load(self.root / 'out' / 'logistic_v1.joblib')
        np.testing.assert_allclose(saved[-1].coef_, reference[-1].coef_, rtol=1e-6, atol=1e-8)

    def test_rows_exclude_test_and_nulls(self):
        with duckdb.connect() as c:
            expected_b_eval = c.execute(f"SELECT count(*) FROM '{self.examples}' WHERE label IS NOT NULL AND as_of >= TIMESTAMP '2025-01-01' AND as_of < TIMESTAMP '2026-01-01'").fetchone()[0]
        self.assertEqual(self.report['candidates']['hgb']['retrained']['all']['rows'], expected_b_eval)


if __name__ == '__main__':
    unittest.main()
