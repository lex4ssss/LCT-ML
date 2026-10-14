from pathlib import Path
import tempfile
import unittest
import duckdb
import numpy as np
import pandas as pd
import explore_v20 as run
import recurrence_v20 as r


class RecurrenceTests(unittest.TestCase):
    def setUp(self):
        self.now = np.datetime64("2025-01-01T00:00:00", "us")
        self.names = r.feature_names()

    def column(self, matrix, name):
        return matrix[:, self.names.index(name)]

    def test_schema_and_empty_history(self):
        matrix = r.object_features([], [self.now], 72)
        self.assertEqual(len(self.names), 30)
        self.assertEqual(len(set(self.names)), 30)
        self.assertEqual(matrix.shape, (1, 30))
        np.testing.assert_array_equal(matrix[0, :13], np.zeros(13))
        self.assertTrue(np.isnan(matrix[0, 13:28]).all())
        np.testing.assert_array_equal(matrix[0, 28:], np.zeros(2))

    def test_rolling_windows_have_exclusive_start_and_inclusive_end(self):
        for hours in (72, 168):
            for window in r.WINDOWS:
                lower = self.now - np.timedelta64(window, "h")
                starts = [lower, lower + np.timedelta64(1, "us"), self.now, self.now + np.timedelta64(1, "us")]
                matrix = r.object_features(starts, [self.now], hours)
                self.assertEqual(self.column(matrix, f"recorded_start_count_{window}h")[0], 2)

    def test_echo_windows_do_not_include_past_window_neighbors(self):
        for hours in (72, 168):
            for lag in r.LAGS:
                lower = self.now - np.timedelta64(lag, "h")
                upper = lower + np.timedelta64(hours, "h")
                starts = [lower, lower + np.timedelta64(1, "us"), upper, upper + np.timedelta64(1, "us")]
                matrix = r.object_features(starts, [self.now], hours)
                self.assertEqual(self.column(matrix, f"recorded_start_echo_{lag}h")[0], 2)

    def test_bursts_use_single_linkage_and_strict_900_second_boundary(self):
        base = self.now - np.timedelta64(1, "D")
        starts = [base, base, base + np.timedelta64(900, "s"), base + np.timedelta64(1800, "s"),
                  base + np.timedelta64(2700_000_001, "us")]
        bursts = r.burst_starts(starts).astype("datetime64[us]")
        np.testing.assert_array_equal(bursts, [base, starts[-1]])
        matrix = r.object_features(starts, [self.now], 72)
        self.assertEqual(self.column(matrix, "recorded_start_count_2160h")[0], 4)
        self.assertEqual(self.column(matrix, "recorded_burst_count_2160h")[0], 2)

    def test_gap_statistics_and_age_ratio(self):
        base = self.now - np.timedelta64(50, "h")
        starts = base + np.array([0, 10, 20, 40], dtype="timedelta64[h]")
        matrix = r.object_features(starts, [self.now], 72)
        self.assertEqual(self.column(matrix, "recorded_gap_last8_median_h")[0], 10)
        self.assertAlmostEqual(self.column(matrix, "recorded_gap_last8_mean_h")[0], 40 / 3)
        self.assertEqual(self.column(matrix, "recorded_gap_last8_min_h")[0], 10)
        self.assertEqual(self.column(matrix, "recorded_gap_last8_max_h")[0], 20)
        self.assertEqual(self.column(matrix, "recorded_burst_age_over_last8_median")[0], 1)
        self.assertEqual(self.column(matrix, "first_recorded_burst_age_h")[0], 50)

    def test_last_eight_and_thirty_two_intervals_are_distinct(self):
        starts = self.now - np.timedelta64(2000, "h") + (np.arange(41) ** 2).astype("timedelta64[h]")
        matrix = r.object_features(starts, [self.now], 168)
        self.assertEqual(self.column(matrix, "recorded_gap_last8_median_h")[0], 72)
        self.assertEqual(self.column(matrix, "recorded_gap_last32_median_h")[0], 48)

    def test_age_and_ratio_caps(self):
        base = self.now - np.timedelta64(1000, "D")
        matrix = r.object_features([base, base + np.timedelta64(1, "h")], [self.now], 72)
        self.assertEqual(self.column(matrix, "last_recorded_burst_age_h")[0], r.AGE_CAP)
        self.assertEqual(self.column(matrix, "recorded_burst_age_over_last8_median")[0], 20)

    def test_count_trend_uses_two_disjoint_windows(self):
        starts = self.now - np.array([1, 2, 3, 2200], dtype="timedelta64[h]")
        matrix = r.object_features(starts, [self.now], 168)
        self.assertAlmostEqual(self.column(matrix, "recorded_start_log_count_change_2160h")[0],
                               np.log1p(3) - np.log1p(1))

    def test_prefix_invariance_for_future_starts_and_unsorted_duplicates(self):
        for index in range(20):
            past = self.now - np.array([1, 5 + index, 24, 720, 9000], dtype="timedelta64[h]")
            future = self.now + np.array([1, 2 + index, 4000], dtype="timedelta64[h]")
            for hours in (72, 168):
                expected = r.object_features(past, [self.now], hours)
                actual = r.object_features(np.r_[future, past[::-1], past], [self.now], hours)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)

    def test_objects_and_input_row_order_remain_separate(self):
        history = pd.DataFrame({"object_id": ["a", "b", "unknown"],
                                "first_alarm_at": [self.now - np.timedelta64(1, "h"),
                                                   self.now - np.timedelta64(2, "h"),
                                                   self.now + np.timedelta64(1, "h")]})
        objects = np.array(["b", "a", "unknown", "a"])
        days = np.array([self.now, self.now, self.now, self.now - np.timedelta64(1, "D")])
        matrix = r.feature_matrix(history, objects, days, 72)
        np.testing.assert_array_equal(self.column(matrix, "recorded_start_count_2160h"), [1, 1, 0, 0])
        np.testing.assert_allclose(self.column(matrix, "last_recorded_start_age_h"), [2, 1, np.nan, np.nan],
                                   equal_nan=True)
        permutation = np.array([2, 0, 3, 1])
        np.testing.assert_allclose(r.feature_matrix(history, objects[permutation], days[permutation], 72),
                                   matrix[permutation], equal_nan=True)

    def test_future_label_columns_cannot_affect_features(self):
        history = pd.DataFrame({"object_id": ["a", "a"], "first_alarm_at": [self.now, self.now + np.timedelta64(1, "D")],
                                "future_label": [0, 1]})
        expected = r.feature_matrix(history, ["a"], [self.now], 72)
        history["future_label"] = [1, 0]
        np.testing.assert_allclose(r.feature_matrix(history, ["a"], [self.now], 72), expected, equal_nan=True)

    def test_invalid_horizons_nat_and_misaligned_inputs_fail(self):
        for horizon in (24, 336):
            with self.assertRaises(ValueError):
                r.object_features([], [self.now], horizon)
        for starts, days in (([np.datetime64("NaT", "us")], [self.now]), ([], [np.datetime64("NaT", "us")])):
            with self.assertRaises(ValueError):
                r.object_features(starts, days, 72)
        with self.assertRaises(ValueError):
            r.feature_matrix(pd.DataFrame(columns=["object_id", "first_alarm_at"]), ["a", "b"], [self.now], 72)

    def test_sql_history_preserves_unknown_mapping_and_excludes_future(self):
        with tempfile.TemporaryDirectory() as folder, duckdb.connect() as connection:
            path = Path(folder) / "episodes.parquet"
            connection.execute("CREATE TABLE e(channel_id BIGINT, first_alarm_at TIMESTAMP)")
            connection.execute("INSERT INTO e VALUES (1,'2025-01-01'),(1,'2025-01-01'),(99,'2024-12-31'),(2,'2030-01-01')")
            connection.execute("COPY e TO ? (FORMAT PARQUET)", [str(path)])
            mapping = pd.DataFrame({"channel_id": ["1", "1", "2"], "object_id": ["old", "a", "b"]})
            history = r.read_history(connection, path, mapping, self.now)
            self.assertEqual(set(history["object_id"]), {"a", "unknown"})
            self.assertEqual(len(history), 2)
            self.assertTrue((history["first_alarm_at"] <= self.now).all())

    def test_matched_toggles_differ_only_in_long_history(self):
        for candidate, baseline in (("H_LONG", "R"), ("C_LONG", "C_B"), ("C_TIME_TN_LONG", "C_TIME_TN")):
            a, b = run.toggles(candidate), run.toggles(baseline)
            self.assertEqual([name for name in a if a[name] != b[name]], ["long_recurrence"])
            self.assertTrue(a["long_recurrence"])

    def test_selection_requires_incumbent_improvement_in_every_measure(self):
        reference = dict(ap=0.8, precision=0.8, recall=0.6, f1=0.68, precision_at_recall_0_5=0.85)
        incumbent = dict(reference, ap=0.81)
        candidate = dict(incumbent, ap=0.816, precision=0.81, recall=0.61, f1=0.70,
                         precision_at_recall_0_5=0.86, qualifies=True)
        self.assertTrue(run.accepted(candidate, reference, incumbent))
        for field in ("precision", "recall", "f1", "precision_at_recall_0_5"):
            failed = dict(candidate, **{field: incumbent[field] - 0.001})
            self.assertFalse(run.accepted(failed, reference, incumbent))
        self.assertFalse(run.accepted(dict(candidate, ap=0.814), reference, incumbent))


if __name__ == "__main__":
    unittest.main()
