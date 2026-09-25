import importlib.util
import os
import sys
import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

path = os.environ.get('EXPERIMENT_MODULE', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_v2.py'))
spec = importlib.util.spec_from_file_location('experiment_v2', path)
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

FEATURES = ['event_count_24h', 'alarm_count_24h', 'event_count_7d', 'alarm_count_7d', 'event_count_30d', 'alarm_count_30d', 'active_days_30d', 'episode_starts_7d', 'episode_starts_30d', 'last_event_age_hours', 'last_alarm_age_hours_30d', 'channel_age_days', 'log_gap_hours_30d']


def brute_confusion(labels, scores, threshold):
    p = scores >= threshold
    tp, fp = int((p & (labels == 1)).sum()), int((p & (labels == 0)).sum())
    fn, tn = int((~p & (labels == 1)).sum()), int((~p & (labels == 0)).sum())
    return tp, fp, fn, tn


def brute_best(labels, scores):
    best = None
    for t in sorted(set(scores.tolist())):
        tp, fp, fn, _ = brute_confusion(labels, scores, t)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        if best is None or f1 >= best[1]:
            best = (t, f1)
    return best


def brute_recall_at(labels, scores, target):
    values = [0.0]
    for t in set(scores.tolist()):
        tp, fp, fn, _ = brute_confusion(labels, scores, t)
        if tp + fp and tp / (tp + fp) >= target:
            values.append(tp / (tp + fn) if tp + fn else 0.0)
    return max(values)


def make_columns(rng, n, start='2019-02-01', days=2500):
    columns = {name: rng.integers(0, 50, n).astype(float) for name in FEATURES}
    columns['last_alarm_age_hours_30d'] = rng.uniform(0, 719, n)
    columns['last_alarm_age_hours_30d'][rng.random(n) < 0.4] = np.nan
    columns['channel_age_days'] = rng.uniform(1, 2000, n)
    columns['as_of'] = np.datetime64(start, 'us') + rng.integers(0, days, n) * np.timedelta64(86400 * 10**6, 'us')
    columns['label'] = (rng.random(n) < 0.2).astype(int)
    return columns


class ExperimentTests(unittest.TestCase):
    def test_fold_masks(self):
        day = np.timedelta64(86400 * 10**6, 'us')
        base = np.datetime64('2023-12-29', 'us')
        as_of = base + np.arange(0, 734) * day
        for fold, train_end, eval_start, eval_end in [('A', '2023-12-31', '2024-01-01', '2025-01-01'), ('B', '2024-12-31', '2025-01-01', '2026-01-01')]:
            train, evaluation = ex.fold_masks(as_of, fold)
            np.testing.assert_array_equal(train, as_of < np.datetime64(train_end, 'us'))
            np.testing.assert_array_equal(evaluation, (as_of >= np.datetime64(eval_start, 'us')) & (as_of < np.datetime64(eval_end, 'us')))
            self.assertFalse((train & evaluation).any())
        with self.assertRaises(ValueError):
            ex.fold_masks(np.array([np.datetime64('2026-01-01', 'us')]), 'A')
        with self.assertRaises(ValueError):
            ex.fold_masks(as_of[:3], 'C')

    def test_matrices(self):
        columns = make_columns(np.random.default_rng(1), 500)
        columns['last_alarm_age_hours_30d'][:3] = [23.999, 24.0, 30.0]
        m1 = ex.matrix_v1(columns)
        self.assertEqual(m1.shape, (500, 4))
        alarm = columns['last_alarm_age_hours_30d']
        np.testing.assert_array_equal(m1[:, 3], np.where(alarm < 24, alarm, np.nan))
        np.testing.assert_array_equal(m1[:, :3], np.column_stack([columns[k] for k in ['event_count_24h', 'alarm_count_24h', 'last_event_age_hours']]))
        m2 = ex.matrix_v2(columns, False)
        expected = np.column_stack([np.minimum(columns[k], 365) if k == 'channel_age_days' else columns[k] for k in FEATURES])
        np.testing.assert_array_equal(m2, expected)
        m3 = ex.matrix_v2(columns, True)
        self.assertEqual(m3.shape, (500, 15))
        dates = columns['as_of'].astype('datetime64[D]')
        months = dates.astype('datetime64[M]').astype(int) % 12 + 1
        weekdays = (dates.astype(int) + 3) % 7
        np.testing.assert_array_equal(m3[:, 13], months)
        np.testing.assert_array_equal(m3[:, 14], weekdays)
        self.assertEqual(m1.dtype, np.float64)

    def test_models(self):
        self.assertEqual(ex.make_model('logistic_v1').steps[0][1].fill_value, 24)
        self.assertEqual(ex.make_model('logistic_v2').steps[0][1].fill_value, 720)
        self.assertIsInstance(ex.make_model('logistic_v2').steps[-1][1], LogisticRegression)
        hgb = ex.make_model('hgb')
        self.assertIsInstance(hgb, HistGradientBoostingClassifier)
        self.assertEqual((hgb.max_iter, hgb.learning_rate, hgb.min_samples_leaf, hgb.early_stopping), (300, 0.05, 200, False))
        with self.assertRaises(ValueError):
            ex.make_model('forest')
        columns = make_columns(np.random.default_rng(2), 400)
        model = ex.make_model('logistic_v1').fit(ex.matrix_v1(columns), columns['label'])
        self.assertTrue(np.isfinite(model.predict_proba(ex.matrix_v1(columns))).all())

    def test_threshold_confusion_recall_against_brute_force(self):
        rng = np.random.default_rng(3)
        for trial in range(20):
            n = int(rng.integers(5, 300))
            labels = (rng.random(n) < rng.uniform(0.05, 0.6)).astype(int)
            scores = np.round(rng.random(n), int(rng.integers(1, 4)))
            with self.subTest(trial=trial):
                threshold, f1 = ex.max_f1_threshold(labels, scores)
                want_t, want_f1 = brute_best(labels, scores)
                self.assertEqual(threshold, want_t)
                self.assertAlmostEqual(f1, want_f1, places=12)
                result = ex.confusion(labels, scores, threshold)
                self.assertEqual((result['tp'], result['fp'], result['fn'], result['tn']), brute_confusion(labels, scores, threshold))
                self.assertAlmostEqual(result['alerts_per_1000'], 1000 * (result['tp'] + result['fp']) / n, places=12)
                for target in (0.3, 0.5):
                    self.assertAlmostEqual(ex.recall_at_precision(labels, scores, target), brute_recall_at(labels, scores, target), places=12)

    def test_f1_tie_takes_highest_threshold(self):
        threshold, f1 = ex.max_f1_threshold(np.array([1, 0, 0, 1]), np.array([0.9, 0.8, 0.7, 0.6]))
        self.assertEqual(threshold, 0.9)
        self.assertAlmostEqual(f1, 2 / 3, places=12)

    def test_zero_division_and_types(self):
        result = ex.confusion(np.array([0, 0, 1]), np.array([0.1, 0.2, 0.3]), 0.9)
        self.assertEqual((result['precision'], result['recall'], result['f1']), (0.0, 0.0, 0.0))
        self.assertEqual(ex.recall_at_precision(np.array([0, 0, 1]), np.array([0.9, 0.8, 0.1]), 0.5), 0.0)
        report = ex.evaluate(np.array([0, 1, 0, 1, 1]), np.array([0.1, 0.9, 0.4, 0.35, 0.8]), 0.5)
        for key in ['rows', 'positives', 'prevalence', 'average_precision', 'brier', 'roc_auc', 'recall_at_precision_0_3', 'recall_at_precision_0_5', 'threshold', 'tp', 'fp', 'fn', 'tn', 'precision', 'recall', 'f1', 'alerts_per_1000']:
            self.assertIn(key, report)
            self.assertIn(type(report[key]), (int, float))
        self.assertEqual((report['tp'], report['fp'], report['fn'], report['tn']), (2, 0, 1, 2))

    def test_reliability(self):
        scores = np.array([0.0, 0.05, 0.1, 0.19, 0.95, 1.0, 0.5])
        labels = np.array([0, 1, 0, 0, 1, 1, 0])
        bins = ex.reliability(labels, scores)
        self.assertEqual([(b['lower'], b['upper'], b['rows']) for b in bins], [(0.0, 0.1, 2), (0.1, 0.2, 2), (0.5, 0.6, 1), (0.9, 1.0, 2)])
        self.assertAlmostEqual(bins[3]['mean_score'], 0.975)
        self.assertAlmostEqual(bins[0]['observed_fraction'], 0.5)


if __name__ == '__main__':
    unittest.main()
