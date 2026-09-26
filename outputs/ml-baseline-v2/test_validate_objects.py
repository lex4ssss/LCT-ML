import unittest

import numpy as np

import validate_objects as vo


class ScoredTest(unittest.TestCase):
    def test_counts(self):
        result = vo.scored(np.array([1, 1, 0, 0, 1]), np.array([0.9, 0.2, 0.8, 0.1, 0.7]), 0.5)
        self.assertEqual((result['precision'], result['recall'], result['alerts']), (0.6667, 0.6667, 3))


class PerObjectTest(unittest.TestCase):
    def test_sorted_by_alerts(self):
        table = vo.per_object(np.array([1, 0, 1, 1]), np.array([True, True, False, True]), np.array([3, 3, 3, 7]))
        self.assertEqual([(row['object'], row['alerts'], row['tp'], row['positives']) for row in table], [(3, 2, 1, 2), (7, 1, 1, 1)])


class FixedThresholdTest(unittest.TestCase):
    def test_margin_when_target_reachable_else_max_f1(self):
        labels = np.array([1, 1, 1, 0, 1, 0, 0, 0])
        self.assertEqual(vo.fixed_threshold(labels, np.linspace(1, 0.3, 8))[1], 'margin')
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        threshold, chosen_by = vo.fixed_threshold(labels, np.linspace(1, 0.3, 8))
        self.assertEqual(chosen_by, 'max_f1')
        self.assertAlmostEqual(threshold, 0.3)


class NaiveScoresTest(unittest.TestCase):
    def test_columns(self):
        history, rhythm = np.arange(8).reshape(2, 4) * 1.0, np.arange(18).reshape(2, 9) * 1.0
        naive = vo.naive_scores(history, rhythm)
        self.assertEqual(naive['target starts 7 d'].tolist(), [1.0, 5.0])
        self.assertEqual(naive['active days of 30'].tolist(), [5.0, 14.0])


if __name__ == '__main__':
    unittest.main()
