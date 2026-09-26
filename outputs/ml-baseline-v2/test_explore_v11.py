import unittest

import numpy as np

import explore_v11 as e11


def moments(*texts):
    return np.array([np.datetime64(t, 'us') for t in texts])


class RhythmTest(unittest.TestCase):
    def setUp(self):
        times = moments('2025-03-09T23:00', '2025-03-09T01:00', '2025-03-08T12:00', '2025-03-06T12:00', '2025-03-03T05:00',
                        '2025-03-03T20:00', '2025-02-24T10:00', '2025-02-01T00:00', '2024-06-01T00:00', '2025-03-10T00:00')
        self.codes = np.array([1] * 10)
        self.times = times
        self.features = e11.rhythm_features(self.codes, times, np.array([1, 2]), np.array(['2025-03-10', '2025-03-10'], dtype='datetime64[D]'), 72)

    def test_long_counts_include_midnight_of_day(self):
        self.assertEqual(self.features[0, :2].tolist(), [9, 10])

    def test_same_window_one_and_two_weeks_back(self):
        self.assertEqual(self.features[0, 2:4].tolist(), [2, 1])

    def test_active_days_before_day(self):
        self.assertEqual(self.features[0, 4:7].tolist(), [4, 5, 6])

    def test_streak_counts_back_from_previous_day(self):
        self.assertEqual(self.features[0, 7], 2)

    def test_same_weekday(self):
        self.assertEqual(self.features[0, 8], 3)

    def test_other_object_sees_nothing(self):
        self.assertEqual(self.features[1].tolist(), [0] * 9)

    def test_future_starts_do_not_change_features(self):
        later = np.concatenate([self.times, moments('2025-03-10T00:00:01', '2025-03-12T00:00')])
        features = e11.rhythm_features(np.array([1] * 12), later, np.array([1]), np.array(['2025-03-10'], dtype='datetime64[D]'), 72)
        self.assertEqual(features[0].tolist(), self.features[0].tolist())


class PrecisionThresholdTest(unittest.TestCase):
    def test_highest_precision_at_recall_0_5(self):
        labels = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        scores = np.linspace(1, 0.1, 10)
        self.assertAlmostEqual(e11.precision_threshold(labels, scores), scores[3])


class CompareTest(unittest.TestCase):
    def test_equal_models_keep_reference_and_better_one_wins(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(3000) < 0.2).astype(np.int64)
        good = labels + 0.3 * rng.normal(0, 1, 3000)
        better = labels + 0.15 * rng.normal(0, 1, 3000)
        rule = np.zeros(3000)
        self.assertEqual(e11.compare(labels, good, good, rule)['chosen'], 'R')
        self.assertEqual(e11.compare(labels, labels + rng.normal(0, 1, 3000), better, rule)['chosen'], 'H')


if __name__ == '__main__':
    unittest.main()
