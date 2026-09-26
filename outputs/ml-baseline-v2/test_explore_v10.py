import unittest

import numpy as np

import explore_v7 as e7
import explore_v10 as e10


class GridTest(unittest.TestCase):
    def test_sixteen_distinct_points(self):
        points = e10.grid_points()
        self.assertEqual(len(points), 16)
        self.assertEqual(len({tuple(sorted(p.items())) for p in points}), 16)


class InnerMaskTest(unittest.TestCase):
    def test_inner_split_never_touches_2025(self):
        day = np.array(['2023-12-27', '2023-12-29', '2024-01-01', '2024-12-28', '2024-12-29', '2025-01-02'], dtype='datetime64[us]')
        train, evaluation = e10.inner_masks(day, np.ones(6, bool), 72)
        self.assertEqual(train.tolist(), [True, False, False, False, False, False])
        self.assertEqual(evaluation.tolist(), [False, False, True, True, True, False])
        self.assertFalse((day[train | evaluation] >= e7.VALIDATION[0]).any())


class StrictThresholdTest(unittest.TestCase):
    def test_highest_precision_at_recall_0_6(self):
        labels = np.array([1, 1, 0, 1, 1, 0, 1, 0, 0, 1])
        scores = np.linspace(1, 0.1, 10)
        self.assertAlmostEqual(e10.strict_threshold(labels, scores), scores[4])

    def test_recall_floor_is_0_6_not_0_5(self):
        labels = np.array([1, 1, 1, 0, 0, 1, 0, 1, 0, 1])
        scores = np.linspace(1, 0.1, 10)
        self.assertAlmostEqual(e10.strict_threshold(labels, scores), scores[5])


class AcceptanceTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.labels = (rng.random(3000) < 0.2).astype(np.int64)
        self.noise = {s: rng.normal(0, 1, 3000) for s in (0.3, 0.6, 1.0)}
        self.rule = np.zeros(3000)

    def scores(self, r, t, m):
        return dict(R=self.labels + r * self.noise[1.0], T=self.labels + t * self.noise[0.3], M=self.labels + m * self.noise[0.6],
                    E=self.labels + 0.5 * (t * self.noise[0.3] + m * self.noise[0.6]))

    def test_better_variant_is_chosen(self):
        results, chosen = e10.evaluate_2025(self.labels, self.scores(1.0, 0.3, 1.0), self.rule)
        self.assertTrue(results['T']['accepted'])
        self.assertIn(chosen, ('T', 'E'))
        self.assertGreaterEqual(results[chosen]['ap'], results['T']['ap'])

    def test_reference_stays_without_gain(self):
        _, chosen = e10.evaluate_2025(self.labels, dict(R=self.labels + self.noise[0.3], T=self.labels + self.noise[1.0],
                                                        M=self.labels + self.noise[1.0], E=self.labels + self.noise[1.0]), self.rule)
        self.assertEqual(chosen, 'R')

    def test_equal_variants_do_not_replace_reference(self):
        same = self.labels + 0.3 * self.noise[0.3]
        _, chosen = e10.evaluate_2025(self.labels, dict(R=same, T=same, M=same, E=same), self.rule)
        self.assertEqual(chosen, 'R')

    def test_largest_ap_among_accepted_wins(self):
        scores = dict(R=self.labels + self.noise[1.0], T=self.labels + 0.6 * self.noise[0.6], M=self.labels + 0.3 * self.noise[0.3],
                      E=self.labels + 0.45 * self.noise[1.0])
        results, chosen = e10.evaluate_2025(self.labels, scores, self.rule)
        accepted = [name for name in ('T', 'M', 'E') if results[name]['accepted']]
        self.assertEqual(accepted[0], 'T')
        self.assertEqual(chosen, max(accepted, key=lambda name: results[name]['ap']))
        self.assertNotEqual(chosen, 'T')


if __name__ == '__main__':
    unittest.main()
