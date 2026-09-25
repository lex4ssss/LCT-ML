import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import duckdb
import joblib
import numpy as np
import run_v5
from test_run_v2 import write_inputs

HERE = Path(__file__).resolve().parent
TYPES = [('Датчик дыма', 'Пожарная охрана'), ('КД Дверь', 'Охранная подсистема'), ('Состояние насоса', 'Диспетчерский контроль')]


def write_directory(path, channels):
    lines = ['"ид_канала_данных","тип_инж_системы","тип_датчика","ид_объект"']
    lines += [f'{channel},{TYPES[int(channel) % 3][1]},{TYPES[int(channel) % 3][0]},{int(channel) % 4}' for channel in channels if int(channel) % 10 != 7]
    path.write_text('\n'.join(lines) + '\n')


class CodeTests(unittest.TestCase):
    def test_codes_follow_sorted_vocabulary_and_unknown_is_nan(self):
        result = run_v5.codes(['b', None, 'a', 'zzz', 'b'], ['a', 'b'])
        np.testing.assert_array_equal(result[[0, 2, 4]], [1.0, 0.0, 1.0])
        self.assertTrue(np.isnan(result[1]) and np.isnan(result[3]))


class RunV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = cls.root = Path(cls.tmp.name)
        _, cls.examples, cls.episodes = write_inputs(root, 5)
        cls.directory, cls.features = root / 'channels.csv', root / 'features.parquet'
        write_directory(cls.directory, [str(channel) for channel in range(40)])
        sys.path.insert(0, str(HERE.parent / 'ml-dataset-v2'))
        import object_features
        with duckdb.connect() as c:
            c.execute(f"CREATE TABLE s AS SELECT * EXCLUDE (episode_starts_7d, episode_starts_30d), 1.0 AS episode_starts_7d, 2.0 AS episode_starts_30d FROM read_parquet('{cls.examples}')")
            source = root / 'source.parquet'
            c.execute(f"COPY (SELECT * FROM s ORDER BY random()) TO '{source}' (FORMAT PARQUET)")
            object_features.build(c, source, cls.directory, cls.features)
        cls.baseline = root / 'baseline.json'
        with cls.examples.open('rb') as handle:
            cls.baseline.write_text(json.dumps(dict(input_sha256=hashlib.file_digest(handle, 'sha256').hexdigest(), candidates=dict(hgb=dict(marker=1)))))
        result = cls.launch(root / 'out')
        if result.returncode:
            raise AssertionError(result.stdout + result.stderr)
        cls.report = json.loads((root / 'out' / 'report.json').read_text())

    @classmethod
    def launch(cls, output, features=None):
        args = [cls.examples, features or cls.features, cls.episodes, cls.directory, cls.baseline, output]
        return subprocess.run([sys.executable, str(HERE / 'run_v5.py'), *map(str, args)], capture_output=True, text=True, cwd=HERE, timeout=900)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_report_and_flags(self):
        self.assertEqual(set(self.report['candidates']), {'hgb_objects'})
        self.assertEqual(self.report['flags']['object_features'], True)
        self.assertEqual(self.report['baseline']['hgb'], dict(marker=1))
        self.assertGreater(self.report['rows_missing_object'], 0)
        self.assertIn('object_day_2025', self.report['candidates']['hgb_objects'])

    def test_model_uses_categorical_columns(self):
        model = joblib.load(self.root / 'out' / 'hgb_objects.joblib')
        width = len(run_v5.ex.V2_FEATURES) + 2 + len(run_v5.OBJECT_FEATURES) + 2
        self.assertEqual(model.n_features_in_, width)
        self.assertEqual(list(np.flatnonzero(model.is_categorical_)), [width - 2, width - 1])

    def test_misaligned_features_refused(self):
        broken = self.root / 'broken.parquet'
        with duckdb.connect() as c:
            c.execute(f"COPY (SELECT * FROM read_parquet('{self.features}') WHERE channel_id <> '3') TO '{broken}' (FORMAT PARQUET)")
        result = self.launch(self.root / 'out-broken', broken)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('do not match', result.stderr)


if __name__ == '__main__':
    unittest.main()
