from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_equal
import explore_v17 as runner
import models_v17 as capacity


class RecordingModel:
    def fit(self, matrix, labels, **kwargs):
        self.rows = np.asarray(matrix)[:, 0].astype(float)
        self.labels = labels.copy()
        self.mean = labels.mean()
        self.kwargs = kwargs
        return self

    def predict_proba(self, matrix):
        scores = np.clip(self.mean + np.asarray(matrix)[:, 1].astype(float) / 100, 0.01, 0.99)
        return np.column_stack([1 - scores, scores])


class CapacityTests(unittest.TestCase):
    def test_inner_calendar_oracle_twenty_scenarios(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            days = np.arange(np.datetime64('2023-01-01'), np.datetime64('2026-01-01'), dtype='datetime64[D]').astype('datetime64[us]')
            rng.shuffle(days)
            present = rng.random(len(days)) > 0.1
            for hours in (72, 168):
                train, evaluation = capacity.inner_masks(days, present, hours)
                expected_train = [seen and moment + np.timedelta64(hours, 'h') < np.datetime64('2024-01-01') for moment, seen in zip(days, present)]
                expected_evaluation = [seen and np.datetime64('2024-01-01') <= moment and moment + np.timedelta64(hours, 'h') <= np.datetime64('2025-01-01') for moment, seen in zip(days, present)]
                assert_array_equal(train, expected_train)
                assert_array_equal(evaluation, expected_evaluation)

    def test_inner_exact_horizon_boundaries(self):
        for hours in (72, 168):
            start, end = np.datetime64('2024-01-01', 'us'), np.datetime64('2025-01-01', 'us')
            horizon = np.timedelta64(hours, 'h')
            days = np.array([start - horizon - np.timedelta64(1, 'us'), start - horizon,
                             start, end - horizon, end - horizon + np.timedelta64(1, 'us')])
            train, evaluation = capacity.inner_masks(days, np.ones(5, bool), hours)
            assert_array_equal(train, [True, False, False, False, False])
            assert_array_equal(evaluation, [False, False, True, True, False])
        for moment in ['NaT', '2026-01-01']:
            with self.assertRaises(ValueError):
                capacity.inner_masks(np.array([moment], dtype='datetime64[us]'), np.ones(1, bool), 72)

    def test_parameters_match_preregistration(self):
        expected_h = [(300, 0.05, 31, 200, 1), (300, 0.05, 31, 50, 1), (600, 0.05, 63, 50, 10), (600, 0.03, 31, 200, 10)]
        expected_c = [(600, 0.05, 6, 10), (600, 0.05, 8, 10), (1200, 0.03, 6, 10), (1200, 0.03, 8, 10)]
        for index, expected in enumerate(expected_h):
            actual = capacity.estimator(f'H{index}').get_params()
            self.assertEqual(tuple(actual[key] for key in ('max_iter', 'learning_rate', 'max_leaf_nodes', 'min_samples_leaf', 'l2_regularization')), expected)
            self.assertFalse(actual['early_stopping'])
            self.assertEqual(actual['random_state'], 0)
        for index, expected in enumerate(expected_c):
            actual = capacity.estimator(f'C{index}').get_params()
            self.assertEqual(tuple(actual[key] for key in ('iterations', 'learning_rate', 'depth', 'l2_leaf_reg')), expected)
            self.assertTrue(actual['has_time'])
            self.assertFalse(actual['allow_writing_files'])
            self.assertEqual(actual['thread_count'], 4)
        with self.assertRaises(ValueError):
            capacity.estimator('bad')

    def test_training_is_chronological_and_categorical_routing(self):
        matrix = np.array([[3, 4], [1, 2], [2, 3]])
        days = np.array(['2023-03-01', '2023-01-01', '2023-02-01'], dtype='datetime64[us]')
        labels = np.array([1, 0, 1])
        for name in ('H1', 'C1'):
            with patch.object(capacity, 'estimator', return_value=RecordingModel()):
                model = capacity.fit(name, matrix, labels, np.array(['x', 'x', 'y']), ['x', 'y'], days)
            assert_array_equal(model.rows, [1, 2, 3])
            assert_array_equal(model.labels, [0, 1, 1])
            self.assertEqual('cat_features' in model.kwargs, name.startswith('C'))
            self.assertEqual(capacity.predict(name, model, matrix, np.array(['x', 'x', 'y']), ['x', 'y']).shape, (3,))

    def test_selection_ties_and_incomplete_grid(self):
        inner = {f'{name}-inner': {'ap': 0.7} for name in capacity.grid()}
        self.assertEqual(capacity.choose(inner, 'H'), 'H0')
        self.assertEqual(capacity.choose(inner, 'C'), 'C0')
        inner['H3-inner']['ap'] = 0.8
        inner['C3-inner']['ap'] = 0.8
        self.assertEqual(capacity.choose(inner, 'H'), 'H3')
        self.assertEqual(capacity.choose(inner, 'C'), 'C3')
        with self.assertRaises(ValueError):
            capacity.choose({}, 'H')
        with self.assertRaises(ValueError):
            capacity.choose(inner, 'bad')

    def test_twenty_future_changes_leave_inner_selection_unchanged(self):
        days = np.arange(np.datetime64('2023-01-01'), np.datetime64('2026-01-01'), dtype='datetime64[D]').astype('datetime64[us]')
        base = np.column_stack([np.arange(len(days)), np.arange(len(days)) % 13])
        matrix = np.column_stack([base, np.zeros((len(days), 2))])
        labels = (np.arange(len(days)) % 5 == 0).astype(int)
        data = dict(present=np.ones(len(days), bool), labels=labels)
        shared = dict(days=days, object_ids=np.full(len(days), 'x'), categories=['x'])
        with tempfile.TemporaryDirectory() as temporary, patch.object(capacity, 'estimator', side_effect=lambda name: RecordingModel()), patch.object(runner.prior, 'emit'):
            first_output = Path(temporary) / 'first'
            first_output.mkdir()
            own = {}
            runner.inner_search(base, matrix, data, shared, 72, 'incident', first_output, own, {'targets': own}, first_output / 'search.json')
            original = own.copy()
            future = days >= np.datetime64('2025-01-01')
            for seed in range(20):
                altered = matrix.copy()
                altered[future] = 1000 + seed
                changed_labels = labels.copy()
                changed_labels[future] = 1 - changed_labels[future]
                output = Path(temporary) / str(seed)
                output.mkdir()
                changed = {}
                runner.inner_search(altered[:, :2], altered, dict(present=data['present'], labels=changed_labels), shared,
                                    72, 'incident', output, changed, {'targets': changed}, output / 'search.json')
                self.assertEqual(changed['selected'], original['selected'])
                for name in original['inner']:
                    self.assertEqual(changed['inner'][name]['ap'], original['inner'][name]['ap'])
                    self.assertEqual(changed['inner'][name]['score_digest'], original['inner'][name]['score_digest'])

    def test_completed_inner_grid_resumes_without_fitting(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(runner.prior, 'emit'), patch.object(capacity, 'fit') as fit:
            output = Path(temporary)
            inner = {}
            for name in ['R', *capacity.grid()]:
                path = output / f'incident-{name}-inner-scores.npy'
                np.save(path, [0.1, 0.2])
                inner[f'{name}-inner'] = {'ap': 0.7, 'score_digest': runner.adaptive.file_digest(path)}
            days = np.array(['2023-01-01', '2024-01-01', '2025-01-01'], dtype='datetime64[us]')
            own = {'inner': inner}
            runner.inner_search(None, None, dict(present=np.ones(3, bool)), dict(days=days), 72, 'incident', output, own, {}, output / 'report.json')
            fit.assert_not_called()
            self.assertEqual(own['selected'], {'H': 'H0', 'C': 'C0'})


if __name__ == '__main__':
    unittest.main()
