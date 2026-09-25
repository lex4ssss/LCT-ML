import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import duckdb
import joblib
from test_run_v2 import write_inputs

HERE = Path(__file__).resolve().parent


class RunV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        rows, examples, episodes = write_inputs(root, 5)
        cls.examples = examples
        cls.labels, cls.directory = root / 'labels.parquet', root / 'channels.csv'
        cls.failure, cls.incident = root / 'failure.parquet', root / 'incident.parquet'
        cls.directory.write_text('"ид_канала_данных","тип_датчика","ид_объект"\n' + ''.join(f'{c},Датчик дыма,{c % 5}\n' for c in range(40)))
        with duckdb.connect() as c:
            c.execute(f"""COPY (SELECT channel_id, as_of, split, label,
                                 CASE WHEN label IS NULL THEN NULL WHEN hash(channel_id || as_of::VARCHAR) % 17 = 0 THEN NULL WHEN hash(as_of::VARCHAR || channel_id) % 2 = 0 THEN label ELSE 0 END AS label_failure,
                                 CASE WHEN label IS NULL THEN NULL ELSE label END AS label_incident
                          FROM read_parquet('{examples}') ORDER BY random()) TO '{cls.labels}' (FORMAT PARQUET)""")
            for path in (cls.failure, cls.incident):
                c.execute(f"COPY (SELECT * FROM read_parquet('{episodes}')) TO '{path}' (FORMAT PARQUET)")
            cls.null_failure = c.execute(f"SELECT count(*) FROM read_parquet('{cls.labels}') WHERE label IS NOT NULL AND label_failure IS NULL").fetchone()[0]
        result = cls.launch(root / 'out')
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.report = json.loads((root / 'out' / 'report.json').read_text())

    @classmethod
    def launch(cls, output):
        args = [cls.examples, cls.labels, cls.failure, cls.incident, cls.directory, output]
        return subprocess.run([sys.executable, str(HERE / 'run_v6.py'), *map(str, args)], capture_output=True, text=True, cwd=HERE, timeout=900)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_targets_reported_with_all_units(self):
        self.assertEqual(set(self.report['targets']), {'failure', 'incident'})
        for target in self.report['targets'].values():
            for period in ('year_2025', 'test_2026'):
                self.assertEqual(set(target[period]), {'channel_day', 'object_day', 'incident', 'incident_rule_alarm_24h'})
        for name in ('failure', 'incident'):
            self.assertTrue((self.root / 'out' / f'hgb_{name}.joblib').exists())

    def test_unknown_class_labels_dropped_only_for_that_target(self):
        self.assertEqual(self.report['targets']['failure']['rows']['unknown_boundary_dropped'], self.null_failure)
        self.assertEqual(self.report['targets']['incident']['rows']['unknown_boundary_dropped'], 0)
        self.assertGreater(self.null_failure, 0)

    def test_test_threshold_comes_from_2025(self):
        failure = self.report['targets']['failure']
        self.assertEqual(failure['test_2026']['channel_day']['model']['threshold'], failure['thresholds']['channel_2025'])
        self.assertEqual(failure['year_2025']['channel_day']['model']['threshold'], failure['thresholds']['channel_2024'])

    def test_models_differ_between_targets(self):
        failure, incident = (joblib.load(self.root / 'out' / f'hgb_{name}.joblib') for name in ('failure', 'incident'))
        self.assertNotEqual(failure.predict_proba([[1.0] * failure.n_features_in_])[0, 1], incident.predict_proba([[1.0] * incident.n_features_in_])[0, 1])

    def test_existing_output_refused(self):
        self.assertIn('FileExistsError', self.launch(self.root / 'out').stderr)


if __name__ == '__main__':
    unittest.main()
