from datetime import datetime, timedelta
import unittest
import numpy as np
import object_predictor as op
import explore_v8 as e8
import experiment_v2 as ex


class HistoryRowTests(unittest.TestCase):
    def test_windows_age_and_fallbacks(self):
        t = datetime(2025, 3, 10)
        starts = [t, t - timedelta(hours=23, minutes=59), t - timedelta(hours=24), t - timedelta(days=6), t - timedelta(days=29, hours=23)]
        np.testing.assert_allclose(op.class_history_row(t, starts, True), [2, 4, 5, 0.0])
        np.testing.assert_allclose(op.class_history_row(t, starts[3:], True), [0, 1, 2, 144.0])
        np.testing.assert_allclose(op.class_history_row(t, [], True), [0, 0, 0, 720.0])
        np.testing.assert_allclose(op.class_history_row(t, [], False), [0, 0, 0, np.nan])

    def test_texts_follow_aggregate_layout(self):
        rng = np.random.default_rng(0)
        n, types = 12, ['a', 'b', 'c']
        columns = {name: rng.integers(0, 3, n).astype(np.float64) for name in ex.V2_FEATURES}
        columns['as_of'] = np.full(n, np.datetime64('2025-01-01', 'us'))
        _, matrix = e8.aggregate(columns, np.zeros(n, dtype=np.int64), rng.integers(0, 3, n), 3)
        texts = op.feature_texts(types)
        self.assertEqual(len(texts), matrix.shape[1] + 4)
        self.assertIn('«c»', texts[1 + len(e8.SUMS) + len(e8.MAXIMA) + len(e8.MINIMA) + 2 * len(e8.ACTIVE) + 2])
        self.assertIn('ведущий канал', texts[-4 - 2 - len(ex.V2_FEATURES)])
        self.assertIn('месяц', texts[-6])


if __name__ == '__main__':
    unittest.main()
