from pathlib import Path
from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import adaptive_v13 as adaptive
import explore_v8 as e8
import explore_v13
import models_v12 as metrics


class RecordingModel:
    def fit(self, matrix, labels, sample_weight=None):
        self.rows = matrix[:, 0].copy()
        self.labels = labels.copy()
        self.weights = sample_weight
        self.mean = np.average(labels, weights=sample_weight)
        return self

    def predict_proba(self, matrix):
        score = np.clip(self.mean + matrix[:, 1] * 0.01, 0.01, 0.99)
        return np.column_stack([1 - score, score])


def fixture():
    days = np.arange(np.datetime64('2024-12-01'), np.datetime64('2025-02-01')).astype('datetime64[us]')
    matrix = np.column_stack([np.arange(len(days)), np.sin(np.arange(len(days)))])
    labels = (np.arange(len(days)) % 3 == 0).astype(int)
    return matrix, labels, days, np.ones(len(days), bool)


class ScheduleTests(unittest.TestCase):
    def test_exact_purge_boundaries(self):
        start = np.datetime64('2025-01-01', 'us')
        for hours in [24, 72, 168, 336]:
            cutoff = start - np.timedelta64(hours, 'h')
            days = np.array([cutoff - np.timedelta64(1, 'us'), cutoff, start,
                             start + np.timedelta64(7, 'D') - np.timedelta64(1, 'us'), start + np.timedelta64(7, 'D')])
            blocks = list(adaptive.blocks(days, np.ones(5, bool), hours, 'W'))
            assert_array_equal(blocks[0][2], [0])
            assert_array_equal(blocks[0][3], [2, 3])
            assert_array_equal(blocks[1][3], [4])
            for block_start, _, train, _ in blocks:
                self.assertTrue(np.all(days[train] + np.timedelta64(hours, 'h') < block_start))

    def test_schedule_against_independent_calendar_oracle(self):
        _, _, days, present = fixture()
        present[::13] = False
        for hours in [72, 168]:
            actual = list(adaptive.blocks(days, present, hours, 'W'))
            for start, end, train, prediction in actual:
                expected_train = [i for i, day in enumerate(days) if present[i] and day + np.timedelta64(hours, 'h') < start]
                expected_prediction = [i for i, day in enumerate(days) if present[i] and start <= day < end]
                assert_array_equal(train, expected_train)
                assert_array_equal(prediction, expected_prediction)

    def test_full_year_single_prediction_and_end_horizon(self):
        days = np.arange(np.datetime64('2024-01-01'), np.datetime64('2026-01-01')).astype('datetime64[us]')
        for hours in [72, 168]:
            prediction = np.concatenate([row[3] for row in adaptive.blocks(days, np.ones(len(days), bool), hours, 'W')])
            expected = np.flatnonzero((days >= np.datetime64('2025-01-01')) &
                                     (days + np.timedelta64(hours, 'h') <= np.datetime64('2026-01-01')))
            assert_array_equal(prediction, expected)

    def test_frozen_variant_has_one_model(self):
        _, _, days, present = fixture()
        blocks = list(adaptive.blocks(days, present, 72, 'F365'))
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0][1], np.datetime64('2026-01-01'))

    def test_future_period_rejected(self):
        with self.assertRaisesRegex(ValueError, 'test rows'):
            list(adaptive.blocks(np.array(['2024-01-01', '2025-01-01', '2026-01-01'], dtype='datetime64[us]'), np.ones(3, bool), 72, 'W'))


class WeightTests(unittest.TestCase):
    def test_independent_half_life_oracle(self):
        cutoff = np.datetime64('2025-01-01', 'us')
        days = cutoff - np.array([0, 365, 730, 1095]).astype('timedelta64[D]')
        actual = adaptive.recency_weights(days, cutoff)
        expected = np.array([1, 0.5, 0.25, 0.125])
        assert_allclose(actual, expected / expected.mean(), rtol=1e-15)
        self.assertEqual(actual.mean(), 1)

    def test_invalid_ages_and_half_life(self):
        for days in [np.array([], dtype='datetime64[us]'), np.array(['2026-01-01'], dtype='datetime64[us]'),
                     np.array(['NaT'], dtype='datetime64[us]')]:
            with self.assertRaises(ValueError):
                adaptive.recency_weights(days, np.datetime64('2025-01-01'))
        with self.assertRaises(ValueError):
            adaptive.recency_weights(np.array(['2024-01-01'], dtype='datetime64[us]'), np.datetime64('2025-01-01'), 0)


class PredictionTests(unittest.TestCase):
    def test_future_labels_and_features_cannot_change_previous_predictions(self):
        matrix, labels, days, present = fixture()
        original, _ = adaptive.score_blocks(matrix, labels, days, present, 72, 'W365', model_factory=RecordingModel)
        changed_matrix, changed_labels = matrix.copy(), labels.copy()
        future = days >= np.datetime64('2025-01-15')
        changed_matrix[future, 1] = 100
        changed_labels[future] = 1 - changed_labels[future]
        changed, _ = adaptive.score_blocks(changed_matrix, changed_labels, days, present, 72, 'W365', model_factory=RecordingModel)
        evaluated_days = days[days >= np.datetime64('2025-01-01')]
        assert_array_equal(original[evaluated_days < np.datetime64('2025-01-15')], changed[evaluated_days < np.datetime64('2025-01-15')])

    def test_appended_future_rows_preserve_prediction_prefix(self):
        matrix, labels, days, present = fixture()
        prefix = days < np.datetime64('2025-01-15')
        before, _ = adaptive.score_blocks(matrix[prefix], labels[prefix], days[prefix], present[prefix], 168, 'W', model_factory=RecordingModel)
        after, _ = adaptive.score_blocks(matrix, labels, days, present, 168, 'W', model_factory=RecordingModel)
        assert_array_equal(before, after[:len(before)])

    def test_unknown_rows_excluded_from_training_and_predictions(self):
        matrix, labels, days, present = fixture()
        present[::3] = False
        models = []
        def factory():
            model = RecordingModel()
            models.append(model)
            return model
        scores, records = adaptive.score_blocks(matrix, labels, days, present, 72, 'W', model_factory=factory)
        self.assertEqual(len(scores), ((days >= np.datetime64('2025-01-01')) & present).sum())
        for model in models:
            self.assertTrue(present[model.rows.astype(int)].all())
        self.assertTrue(all(record['label_window_last_end'] < record['block_start'] for record in records))

    def test_resume_uses_saved_models_without_refitting(self):
        matrix, labels, days, present = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            first, records = adaptive.score_blocks(matrix, labels, days, present, 72, 'W365', temporary, 'same', RecordingModel)
            def forbidden():
                raise AssertionError('refit on resume')
            again, resumed_records = adaptive.score_blocks(matrix, labels, days, present, 72, 'W365', temporary, 'same', forbidden)
            assert_array_equal(first, again)
            self.assertEqual(records, resumed_records)
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                adaptive.score_blocks(matrix, labels, days, present, 72, 'W365', temporary, 'changed', forbidden)

    def test_corrupted_saved_model_rejected(self):
        matrix, labels, days, present = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            adaptive.score_blocks(matrix, labels, days, present, 72, 'W', temporary, 'same', RecordingModel)
            next(Path(temporary).glob('*.model.joblib')).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'model changed'):
                adaptive.score_blocks(matrix, labels, days, present, 72, 'W', temporary, 'same', RecordingModel)

    def test_reproducible_actual_hgb_and_serialization(self):
        rng = np.random.default_rng(12)
        days = np.array(['2024-01-01'] * 600 + ['2025-01-01'] * 40, dtype='datetime64[us]')
        matrix = rng.normal(size=(640, 3))
        matrix[::31, 1] = np.nan
        labels = (matrix[:, 0] > 0).astype(int)
        present = np.ones(640, bool)
        with tempfile.TemporaryDirectory() as temporary:
            first, _ = adaptive.score_blocks(matrix, labels, days, present, 72, 'F365', temporary, 'hgb')
            second, _ = adaptive.score_blocks(matrix, labels, days, present, 72, 'F365')
            assert_array_equal(first, second)
            self.assertTrue(np.isfinite(first).all())


class RunnerTests(unittest.TestCase):
    def test_search_and_resume_end_to_end(self):
        rng = np.random.default_rng(7)
        days = np.array(['2024-01-01'] * 600 + ['2025-01-01'] * 20 + ['2025-01-08'] * 20, dtype='datetime64[us]')
        labels = (np.arange(640) % 5 == 0).astype(int)
        present = np.ones(640, bool)
        data = dict(present=present, labels=labels, history=np.zeros((640, 4)), rule=np.zeros(640), strong_rule=np.zeros(640))
        shared = dict(base=rng.normal(size=(640, 73)), days=days, object_ids=np.array(['a'] * 640),
                      targets={'incident': data, 'neispraven': data})
        source = Path(adaptive.__file__).parent
        previous = dict(complete=True, packages={}, code={name: adaptive.file_digest(source / name) for name in
                        ['explore_v12.py', 'models_v12.py', 'telemetry_v12.py', 'spec_explore_v12.txt']}, targets={})
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            cache, manifest, output = directory / 'features.joblib', directory / 'previous.json', directory / 'output'
            joblib.dump(shared, cache)
            for target, hours in [('incident', 72), ('neispraven', 168)]:
                train, evaluated = e8.masks(days, present, hours)
                matrix = np.column_stack([shared['base'], data['history']])
                reference_model = RecordingModel().fit(matrix[train], labels[train])
                scores = reference_model.predict_proba(matrix[evaluated])[:, 1]
                np.save(directory / f'{target}-R-scores.npy', scores)
                reference = metrics.metrics(labels[evaluated], scores, data['rule'][evaluated], data['strong_rule'][evaluated])
                previous['targets'][target] = dict(winner='R', results={'R': dict(validation=reference)})
            manifest.write_text(json.dumps(previous))
            with patch.object(adaptive, 'ex') as experiment:
                experiment.make_model.side_effect = lambda kind: RecordingModel()
                explore_v13.search(cache, manifest, output)
            report = json.loads((output / 'search.json').read_text())
            self.assertTrue(report['complete'])
            self.assertFalse(report['test_evaluation'])
            self.assertFalse(report['service_changed'])
            for target in ['incident', 'neispraven']:
                self.assertEqual(list(report['targets'][target]['results']), list(adaptive.VARIANTS))
            with patch.object(adaptive, 'score_blocks', side_effect=AssertionError('refit after completed search')):
                explore_v13.search(cache, manifest, output)
            shared['base'][0, 0] += 1
            joblib.dump(shared, cache)
            with self.assertRaisesRegex(ValueError, 'fingerprint changed'):
                explore_v13.search(cache, manifest, output)


if __name__ == '__main__':
    unittest.main()
