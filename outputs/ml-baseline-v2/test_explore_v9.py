import unittest

import numpy as np

import explore_v7 as e7
import explore_v9 as e9


def moment(text):
    return np.datetime64(text, 'us')


class SiblingTest(unittest.TestCase):
    def test_siblings_share_parent_and_exclude_self(self):
        table = e9.sibling_table(np.array(['5', '5', '7', '', '5'], dtype=object))
        self.assertEqual(sorted(table[0][table[0] >= 0]), [1, 4])
        self.assertEqual(sorted(table[4][table[4] >= 0]), [0, 1])
        self.assertTrue((table[2] < 0).all())
        self.assertTrue((table[3] < 0).all())


class CountTest(unittest.TestCase):
    def setUp(self):
        times = ['2025-03-01T10:00', '2025-03-01T20:00', '2025-02-27T12:00', '2025-02-10T00:00', '2025-03-02T01:00']
        objects = np.array([1, 1, 1, 1, 1])
        self.keys = np.sort(e7.keys(objects, np.array([moment(t) for t in times])))
        self.has_pumps = np.array([0.0, 1.0, 0.0])

    def test_windows_look_only_backward(self):
        counts = e9.window_counts(self.keys, np.array([1]), np.array([moment('2025-03-02T00:00')]), (24, 168, 720))
        self.assertEqual([int(c[0]) for c in counts], [2, 3, 4])

    def test_event_on_window_start_is_excluded(self):
        keys = np.sort(e7.keys(np.array([1, 1]), np.array([moment('2025-03-01T00:00'), moment('2025-03-02T00:00')])))
        counts = e9.window_counts(keys, np.array([1]), np.array([moment('2025-03-02T00:00')]), (24,))
        self.assertEqual(int(counts[0][0]), 1)

    def test_other_object_is_not_counted(self):
        counts = e9.window_counts(self.keys, np.array([2]), np.array([moment('2025-03-02T00:00')]), (24,))
        self.assertEqual(int(counts[0][0]), 0)

    def test_flap_ratio(self):
        block = e9.own_block(self.keys, self.has_pumps, np.array([1, 0]), np.array([moment('2025-03-02T00:00')] * 2))
        self.assertAlmostEqual(block[0, 3], 2 / (4 / 30))
        self.assertEqual(block[0, 4], 1.0)
        self.assertEqual(block[1].tolist(), [0, 0, 0, 0, 0])

    def test_sibling_sums_and_floods(self):
        siblings = e9.sibling_table(np.array(['5', '5', '5'], dtype=object))
        flood = np.sort(e7.keys(np.array([2, 2, 0]), np.array([moment('2025-03-01T12:00'), moment('2025-02-26T00:00'), moment('2025-03-01T12:00')])))
        features = e9.pump_features(self.keys, self.has_pumps, flood, siblings, np.array([0, 1]), np.array([moment('2025-03-02T00:00')] * 2))
        self.assertEqual(features.shape[1], 12)
        self.assertEqual(features[0, 5:8].tolist(), [2, 3, 4])
        self.assertEqual(features[0, 9], 1.0)
        self.assertEqual(features[0, 10:].tolist(), [1, 2])
        self.assertEqual(features[1, 10:].tolist(), [2, 3])
        self.assertEqual(features[1, 5:8].tolist(), [0, 0, 0])


class AcceptanceTest(unittest.TestCase):
    def test_gain_and_precision_rule(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(2000) < 0.2).astype(np.int64)
        strong = labels + rng.normal(0, 0.3, 2000)
        weak = labels + rng.normal(0, 1.0, 2000)
        rule = np.zeros(2000)
        self.assertTrue(e9.compare(labels, weak, strong, rule)['accepted'])
        self.assertFalse(e9.compare(labels, strong, weak, rule)['accepted'])
        self.assertFalse(e9.compare(labels, strong, strong, rule)['accepted'])

    def test_higher_ap_with_lower_precision_at_half_recall_is_rejected(self):
        labels = np.array([1] * 5 + [0] * 20 + [1] * 5 + [0] * 70)
        scores = np.linspace(1, 0, 100)
        variant_scores = np.empty(100)
        positives, negatives = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
        ranked = np.r_[negatives[:1], positives, negatives[1:]]
        variant_scores[ranked] = scores
        result = e9.compare(labels, scores, variant_scores, np.zeros(100))
        self.assertGreater(result['ap_gain'], 0.1)
        self.assertLess(result['variant']['precision_at_recall_0_5'], result['reference']['precision_at_recall_0_5'])
        self.assertTrue(result['variant']['qualifies'])
        self.assertFalse(result['accepted'])


if __name__ == '__main__':
    unittest.main()
