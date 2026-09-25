import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import joblib
import numpy as np
from test_run_v2 import FEATURES, HERE, run, write_inputs

sys.path.insert(0, str(HERE))
import experiment_v2 as ex


def final(*args):
    return subprocess.run([sys.executable, str(HERE / 'final_v2.py'), *map(str, args)], capture_output=True, text=True, cwd=HERE, timeout=900)


class FinalV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.rows, cls.examples, cls.episodes = write_inputs(cls.root, 9)
        assert run(cls.examples, cls.episodes, cls.root / 'run').returncode == 0
        cls.model = cls.root / 'run' / 'hgb.joblib'
        result = final('decide', cls.examples, cls.model, cls.root / 'decision.json')
        assert result.returncode == 0, result.stderr
        cls.decision = json.loads((cls.root / 'decision.json').read_text())
        result = final('test', cls.examples, cls.episodes, cls.root / 'decision.json', cls.root / 'test.json')
        assert result.returncode == 0, result.stderr
        cls.report = json.loads((cls.root / 'test.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def part(self, year):
        rows = [r for r in self.rows if r[-2] is not None and r[1].year == year]
        columns = {name: np.array([np.nan if r[FEATURES.index(name) + 2] is None else r[FEATURES.index(name) + 2] for r in rows], dtype=float) for name in FEATURES}
        columns['as_of'] = np.array([np.datetime64(r[1], 'us') for r in rows])
        scores = joblib.load(self.model).predict_proba(ex.matrix_v2(columns, True))[:, 1]
        return np.array([r[-2] for r in rows]), scores

    def test_decision_uses_only_2025(self):
        labels, scores = self.part(2025)
        self.assertEqual(self.decision['validation_rows'], len(labels))
        f1_by_threshold = {t: 2 * ((scores >= t) & (labels == 1)).sum() / ((scores >= t).sum() + labels.sum()) for t in set(scores.tolist())}
        best = max(f1_by_threshold.values())
        self.assertEqual(self.decision['threshold'], max(t for t, f in f1_by_threshold.items() if f == best))
        self.assertFalse(self.decision['test_evaluated'])

    def test_test_metrics_match_direct_count(self):
        labels, scores = self.part(2026)
        predicted = scores >= self.decision['threshold']
        hgb = self.report['hgb']['all']
        self.assertEqual((hgb['rows'], hgb['tp'], hgb['fp']), (len(labels), int((predicted & (labels == 1)).sum()), int((predicted & (labels == 0)).sum())))
        self.assertEqual(self.report['rows'], len(labels))
        self.assertTrue(self.report['flags']['test_evaluation'])

    def test_refuses_changed_model(self):
        copy = self.root / 'decision-copy.json'
        decision = dict(self.decision, model_path=str(self.root / 'other.joblib'))
        shutil.copy(self.model, self.root / 'other.joblib')
        with open(self.root / 'other.joblib', 'ab') as handle:
            handle.write(b'x')
        copy.write_text(json.dumps(decision))
        result = final('test', self.examples, self.episodes, copy, self.root / 'test2.json')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('decision does not match files', result.stderr)


if __name__ == '__main__':
    unittest.main()
