import unittest

import numpy as np

import validate_stability as vs


class StabilityTest(unittest.TestCase):
    def test_precision_recall_rows(self):
        precision, recall = vs.precision_recall(np.array([[1, 1, 0, 0], [1, 0, 0, 0]]), np.array([[True, False, True, False], [False, False, False, False]]))
        self.assertEqual(precision.tolist(), [0.5, 0.0])
        self.assertEqual(recall.tolist(), [0.5, 0.0])

    def test_by_month(self):
        day = np.array(['2026-01-05', '2026-01-20', '2026-02-01', '2026-02-02'], dtype='datetime64[us]')
        table = vs.by_month(np.array([1, 0, 1, 1]), np.array([True, True, True, False]), day)
        self.assertEqual(list(table), ['2026-01', '2026-02'])
        self.assertEqual((table['2026-01']['precision'], table['2026-02']['recall']), (0.5, 0.5))

    def test_alerts_per_day(self):
        day = np.array(['2026-01-01', '2026-01-01', '2026-01-01', '2026-01-02', '2026-01-02', '2026-01-03'], dtype='datetime64[us]')
        result = vs.alerts_per_day(np.array([True, True, False, True, False, False]), day)
        self.assertEqual(result, dict(days=3, mean=1.0, median=1.0, max=2))

    def test_bootstrap_brackets_point_estimate(self):
        rng = np.random.default_rng(1)
        labels = (rng.random(2000) < 0.3).astype(np.int64)
        alarms = np.where(labels == 1, rng.random(2000) < 0.7, rng.random(2000) < 0.1)
        groups = np.arange(2000) % 40
        result = vs.bootstrap(labels, alarms, groups)
        precision, recall = vs.precision_recall(labels, alarms)
        self.assertTrue(result['rows']['precision'][0] <= precision <= result['rows']['precision'][1])
        self.assertTrue(result['objects']['recall'][0] <= recall <= result['objects']['recall'][1])


if __name__ == '__main__':
    unittest.main()
