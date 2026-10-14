from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
import pandas as pd
import explore_v21 as run
import models_v12 as models
import shared_v21 as shared


class IntervalScores:
    def __init__(self, numeric_features=2):
        self.numeric_features = numeric_features

    def predict_proba(self, matrix):
        if isinstance(matrix, pd.DataFrame):
            expected = [f"f{index}" for index in range(self.numeric_features + 1)] + ["object_id"]
            if list(matrix.columns) != expected:
                raise ValueError("Wrong internal feature order")
            interval = matrix[f"f{self.numeric_features}"].to_numpy(dtype=int)
        else:
            interval = np.asarray(matrix)[:, -1].astype(int)
        probability = np.array([0.2, 0.25, 0.1])[interval]
        return np.column_stack([1 - probability, probability])


class SharedHazardTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array([[0, 10], [1, 11], [2, 12], [3, 13]], dtype=float)
        self.onsets = np.array([24, 72, 168, np.nan])
        self.objects = np.array(["a", "b", "a", "b"])
        self.days = np.repeat(np.datetime64("2024-01-01", "us"), 4)

    def expanded(self):
        return shared.expand_training(self.matrix, self.onsets, self.objects, self.days, 168)

    def test_risk_sets_include_one_positive_per_observed_onset(self):
        result = self.expanded()
        self.assertEqual([row["rows"] for row in result["training"]], [4, 3, 2])
        self.assertEqual([row["positives"] for row in result["training"]], [1, 1, 1])
        self.assertEqual(result["labels"].sum(), 3)
        self.assertEqual(np.bincount(result["source_rows"]).tolist(), [1, 2, 3, 3])
        for source in range(4):
            selected = result["source_rows"] == source
            self.assertEqual(result["labels"][selected].sum(), int(np.isfinite(self.onsets[source])))

    def test_expanded_features_contain_only_original_values_and_interval_code(self):
        result = self.expanded()
        np.testing.assert_array_equal(result["matrix"][:, :2], self.matrix[result["source_rows"]])
        np.testing.assert_array_equal(result["matrix"][:, 2], result["interval_codes"])
        self.assertEqual(result["matrix"].shape, (9, 3))
        np.testing.assert_array_equal(result["object_ids"], self.objects[result["source_rows"]])
        np.testing.assert_array_equal(result["interval_codes"][result["source_rows"] == 0], [0])

    def test_binary_loss_equals_discrete_time_event_likelihood(self):
        result = self.expanded()
        hazard = np.array([0.2, 0.25, 0.1])[result["interval_codes"]]
        labels = result["labels"]
        binary_loss = -(labels * np.log(hazard) + (1 - labels) * np.log1p(-hazard)).sum()
        event_likelihood = np.array([0.2, 0.8 * 0.25, 0.8 * 0.75 * 0.1, 0.8 * 0.75 * 0.9])
        self.assertAlmostEqual(binary_loss, -np.log(event_likelihood).sum(), places=13)

    def test_chronological_order_groups_source_rows_then_intervals(self):
        days = self.days + np.array([2, 0, 1, 0], dtype="timedelta64[D]")
        result = shared.expand_training(self.matrix, self.onsets, self.objects, days, 168)
        triples = list(zip(result["days"].astype(np.int64), result["source_rows"], result["interval_codes"]))
        self.assertEqual(triples, sorted(triples))

    def test_risk_set_boundaries_for_both_horizons(self):
        for hours in (72, 168):
            onsets = np.array([24, 24.000001, 72, np.nan])
            result = shared.expand_training(self.matrix, onsets, self.objects, self.days, hours)
            positives = [(int(source), int(interval)) for source, interval, label in
                         zip(result["source_rows"], result["interval_codes"], result["labels"]) if label]
            self.assertEqual(positives, [(0, 0), (1, 1), (2, 1)])

    def test_empty_conditional_interval_and_empty_input_are_rejected(self):
        with self.assertRaises(ValueError):
            shared.expand_training(self.matrix, np.repeat(24, 4), self.objects, self.days, 72)
        with self.assertRaises(ValueError):
            shared.expand_training(np.empty((0, 2)), [], [], [], 168)

    def test_invalid_times_values_and_shapes_are_rejected(self):
        for value in (0, -1, 169, np.inf, -np.inf):
            onsets = self.onsets.copy()
            onsets[0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                shared.expand_training(self.matrix, onsets, self.objects, self.days, 168)
        bad = self.matrix.copy()
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            shared.expand_training(bad, self.onsets, self.objects, self.days, 168)
        days = self.days.copy()
        days[0] = np.datetime64("NaT", "us")
        with self.assertRaises(ValueError):
            shared.expand_training(self.matrix, self.onsets, self.objects, days, 168)
        with self.assertRaises(ValueError):
            shared.expand_training(self.matrix, self.onsets[:-1], self.objects, self.days, 168)

    def test_numeric_prediction_is_conditional_product_and_monotone(self):
        model = shared.SharedHazardModel(IntervalScores(), 2, False, 168, [])
        curve = model.predict_curve(self.matrix)
        np.testing.assert_allclose(curve, np.tile([0.2, 0.4, 0.46], (4, 1)), rtol=1e-14)
        self.assertTrue((np.diff(curve, axis=1) >= 0).all())
        np.testing.assert_allclose(model.predict_proba(self.matrix).sum(axis=1), np.ones(4))

    def test_categorical_prediction_orders_named_features_without_mutating_input(self):
        model = shared.SharedHazardModel(IntervalScores(), 2, True, 168, [])
        frame = pd.DataFrame({"object_id": self.objects, "f1": self.matrix[:, 1], "f0": self.matrix[:, 0]})
        before = frame.copy(deep=True)
        np.testing.assert_allclose(model.predict_proba(frame)[:, 1], np.repeat(0.46, 4))
        pd.testing.assert_frame_equal(frame, before)

    def test_prediction_rejects_missing_features(self):
        numeric = shared.SharedHazardModel(IntervalScores(), 2, False, 72, [])
        categorical = shared.SharedHazardModel(IntervalScores(), 2, True, 72, [])
        with self.assertRaises(ValueError):
            numeric.predict_proba(np.ones((4, 3)))
        with self.assertRaises(ValueError):
            categorical.predict_proba(self.matrix)
        with self.assertRaises(ValueError):
            categorical.predict_proba(pd.DataFrame({"f0": [1], "object_id": ["a"]}))

    def test_empty_prediction_retains_curve_and_probability_shapes(self):
        model = shared.SharedHazardModel(IntervalScores(), 2, False, 168, [])
        self.assertEqual(model.predict_curve(np.empty((0, 2))).shape, (0, 3))
        self.assertEqual(model.predict_proba(np.empty((0, 2))).shape, (0, 2))

    def test_all_negative_training_uses_constant_model_with_list_input(self):
        model = shared.fit("H_SHARED_B", self.matrix.tolist(), np.full(4, np.nan), self.objects, ["a", "b"], self.days, 168)
        self.assertEqual(model.numeric_features, 2)
        np.testing.assert_array_equal(model.predict_proba(self.matrix)[:, 1], np.zeros(4))

    def test_fit_delegates_chronological_expansion_with_unchanged_family(self):
        with patch.object(shared.models, "fit_model", return_value=IntervalScores()) as fit:
            model = shared.fit("C_SHARED_B", self.matrix, self.onsets, self.objects, ["a", "b"], self.days, 168)
        variant, matrix, labels, objects, categories, days = fit.call_args.args
        self.assertEqual(variant, "C_SHARED_B")
        self.assertEqual(matrix.shape, (9, 3))
        self.assertEqual(labels.sum(), 3)
        self.assertEqual(categories, ["a", "b"])
        self.assertEqual(len(days), len(objects))
        frame = models.model_frame(self.matrix, self.objects, ["a", "b"])
        np.testing.assert_allclose(model.predict_proba(frame)[:, 1], np.repeat(0.46, 4))

    def test_real_hgb_and_catboost_save_load_preserve_predictions(self):
        matrix = np.arange(80, dtype=float).reshape(40, 2)
        onsets = np.array([72 if index % 2 else np.nan for index in range(40)])
        objects = np.array(["a" if index % 3 else "b" for index in range(40)])
        days = np.datetime64("2024-01-01", "us") + np.arange(40).astype("timedelta64[D]")
        with tempfile.TemporaryDirectory() as folder:
            for variant in ("H_SHARED_B", "C_SHARED_B"):
                model = shared.fit(variant, matrix, onsets, objects, ["a", "b"], days, 72)
                path = Path(folder) / f"{variant}.joblib"
                expected = models.predict_model(variant, model, matrix, objects, ["a", "b"])
                models.save_verified(model, path, variant, matrix, objects, ["a", "b"])
                actual = models.predict_model(variant, joblib.load(path), matrix, objects, ["a", "b"])
                np.testing.assert_array_equal(actual, expected)
                self.assertTrue(np.isfinite(actual).all())

    def test_matched_toggles_only_change_parameter_sharing(self):
        for variant in run.VARIANTS:
            current, old = run.toggles(variant), run.toggles(run.matched_variant(variant))
            self.assertEqual([key for key in current if current[key] != old[key]], ["shared_interval_parameters"])
            self.assertFalse(current["long_recurrence"])
        self.assertEqual(len(run.VARIANTS), 6)

    def test_selection_preserves_incumbent_without_eligible_candidate(self):
        reference = dict(ap=0.8, precision=0.8, recall=0.6, f1=0.68, precision_at_recall_0_5=0.85)
        incumbent = dict(reference, ap=0.81)
        candidate = dict(reference, qualifies=True, margin=0.1)
        results = {"R": {"validation": reference}, "S": {"validation": incumbent}}
        results.update({name: dict(validation=dict(candidate), features=79, model_count=1) for name in run.VARIANTS})
        self.assertEqual(run.select(results), "S")
        results["C_SHARED_TN"]["validation"].update(ap=0.82, precision=0.81, recall=0.61, f1=0.70,
                                                    precision_at_recall_0_5=0.86)
        self.assertEqual(run.select(results), "C_SHARED_TN")

    def test_search_rejects_test_rows_before_any_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / str(index) for index in range(5)]
            source_state = ({}, {}, {}, {"hgb_parameters": {}, "catboost_parameters": {}}, {}, paths[0], paths[1])
            with patch.object(run, "source_state", return_value=source_state), \
                 patch.object(run.joblib, "load", return_value={"days": np.array([np.datetime64("2026-01-01", "us")])}), \
                 patch.object(run.shared_hazard, "fit") as fit:
                with self.assertRaises(ValueError):
                    run.search(*paths)
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
