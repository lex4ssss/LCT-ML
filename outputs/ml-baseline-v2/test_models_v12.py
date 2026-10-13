from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
import models_v12 as models


class SplitTests(unittest.TestCase):
    def test_exact_horizon_boundaries(self):
        for hours in (72, 168):
            start = np.datetime64('2025-01-01', 'us')
            end = np.datetime64('2026-01-01', 'us')
            horizon = np.timedelta64(hours, 'h')
            days = np.array([start - horizon - np.timedelta64(1, 'us'), start - horizon,
                             start - np.timedelta64(1, 'us'), start, end - horizon,
                             end - horizon + np.timedelta64(1, 'us')])
            train, evaluation = models.training_masks(days, np.ones(6, bool), hours, 'H_ND')
            assert_array_equal(train, [True, False, False, False, False, False])
            assert_array_equal(evaluation, [False, False, False, True, True, False])

    def test_recent_history_and_unknown_labels(self):
        days = np.array(['2021-12-31', '2022-01-01', '2023-01-01', '2025-01-01'], dtype='datetime64[us]')
        train, evaluation = models.training_masks(days, np.array([True, True, False, False]), 72, 'C_ND_2022')
        assert_array_equal(train, [False, True, False, False])
        self.assertFalse(evaluation.any())

    def test_test_period_rejected(self):
        with self.assertRaisesRegex(ValueError, 'test rows'):
            models.training_masks(np.array(['2026-01-01'], dtype='datetime64[us]'), np.ones(1, bool), 72, 'R')


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.reference = dict(ap=0.8, precision_at_recall_0_5=0.85, qualifies=True, f1=0.73, margin=0.14)
        self.candidate = dict(ap=0.82, precision_at_recall_0_5=0.86, qualifies=True, f1=0.75, margin=0.15)

    def test_every_acceptance_gate(self):
        self.assertTrue(models.accepted(self.candidate, self.reference))
        for name, value in [('ap', 0.8099), ('precision_at_recall_0_5', 0.8499), ('qualifies', False), ('f1', 0.7299)]:
            self.assertFalse(models.accepted(dict(self.candidate, **{name: value}), self.reference))

    def test_reference_retained(self):
        rows = {'R': dict(validation=self.reference, features=77),
                'H_N': dict(validation=dict(self.candidate, ap=0.805), features=125)}
        self.assertEqual(models.winner(rows), 'R')

    def test_ap_then_margin_then_size(self):
        rows = {'R': dict(validation=self.reference, features=77),
                'H_N': dict(validation=self.candidate, features=125),
                'H_D': dict(validation=self.candidate, features=96)}
        self.assertEqual(models.winner(rows), 'H_D')
        rows['H_N']['validation'] = dict(self.candidate, margin=0.16)
        self.assertEqual(models.winner(rows), 'H_N')
        rows['H_D']['validation'] = dict(self.candidate, ap=0.83)
        self.assertEqual(models.winner(rows), 'H_D')

    def test_fixed_threshold_not_reselected(self):
        labels = np.array([1, 1, 0, 0, 0])
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        row = models.metrics(labels, scores, np.zeros(5), np.ones(5), 0.85)
        self.assertEqual(row['threshold'], 0.85)
        self.assertEqual(row['precision'], 1)
        self.assertEqual(row['recall'], 0.5)
        self.assertEqual(row['ap'], 1)


class FeatureTests(unittest.TestCase):
    def test_variant_features(self):
        base, numeric, domains = np.ones((3, 2)), np.full((3, 4), 2), np.full((3, 5), 3)
        expected = {'R': 2, 'H_N': 6, 'H_D': 7, 'H_ND': 11, 'C_B': 2,
                    'C_ND': 11, 'C_ND_2022': 11, 'L_ND_2022': 11, 'E': 11}
        for variant, width in expected.items():
            matrix = models.feature_matrix(variant, base, numeric, domains)
            self.assertEqual(matrix.shape, (3, width))
            assert_array_equal(matrix[:, :2], base)
            flags = models.toggles(variant)
            for name in ['class_weights', 'weather', 'synthetic', 'neural_network', 'online_retraining', 'calibration']:
                self.assertFalse(flags[name])

    def test_unknown_object_category_preserved(self):
        matrix = np.array([[np.nan], [1]])
        for lightgbm in [False, True]:
            frame = models.model_frame(matrix, ['a', 'unknown'], ['a', 'unknown'], lightgbm)
            self.assertEqual(frame['object_id'].tolist(), ['a', 'unknown'])
            self.assertTrue(np.isnan(frame.iloc[0, 0]))
        with self.assertRaisesRegex(ValueError, 'category absent'):
            models.model_frame(matrix, ['a', 'missing'], ['a', 'unknown'], True)


class ModelTests(unittest.TestCase):
    def test_serialization_and_reproducibility(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(600, 3))
        matrix[::17, 1] = np.nan
        labels = (matrix[:, 0] + np.nan_to_num(matrix[:, 1]) > 0).astype(int)
        objects = np.where(np.arange(600) % 3 == 0, 'unknown', 'a')
        categories = ['a', 'unknown']
        days = np.datetime64('2023-01-01') + rng.integers(0, 200, 600).astype('timedelta64[D]')
        for variant in ['H_ND', 'C_ND_2022', 'L_ND_2022']:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temporary:
                first = models.fit_model(variant, matrix, labels, objects, categories, days)
                second = models.fit_model(variant, matrix, labels, objects, categories, days)
                expected = models.predict_model(variant, first, matrix, objects, categories)
                actual = models.predict_model(variant, second, matrix, objects, categories)
                assert_array_equal(actual, expected)
                models.save_verified(first, Path(temporary) / 'model.joblib', variant, matrix, objects, categories)
                loaded = joblib.load(Path(temporary) / 'model.joblib')
                assert_allclose(models.predict_model(variant, loaded, matrix, objects, categories), expected, rtol=0, atol=0)
                self.assertTrue(np.isfinite(expected).all())
                self.assertGreater(expected[labels == 1].mean(), expected[labels == 0].mean())


if __name__ == '__main__':
    unittest.main()
