from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import duckdb
import joblib
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
import explore_v18 as runner
import models_v12 as models
import time_v18 as timing


def calculate(snapshots, episodes, hours):
    with tempfile.TemporaryDirectory() as temporary, duckdb.connect() as connection:
        paths = [Path(temporary) / name for name in ('examples.parquet', 'episodes.parquet')]
        for frame, path in zip((snapshots, episodes), paths):
            if path.name == 'episodes.parquet':
                frame = frame.copy()
                frame['left_boundary'] = frame['left_boundary'].astype('string')
            connection.register('fixture', frame)
            connection.execute('COPY fixture TO ? (FORMAT PARQUET)', [str(path)])
        mapping = pd.DataFrame(dict(channel_id=['a', 'b', 'c'], object_code=[0, 0, 1]))
        return timing.label_table(connection, *paths, mapping, 2, hours)


def oracle(snapshots, episodes, hours):
    mapping = {'a': 0, 'b': 0, 'c': 1}
    rows = {}
    for snapshot in snapshots.itertuples():
        if pd.isna(snapshot.label) or snapshot.split not in ('train', 'validation') or snapshot.as_of >= pd.Timestamp('2026-01-01'):
            continue
        key = mapping.get(snapshot.channel_id, 2) * 1_000_000 + np.datetime64(snapshot.as_of, 'D').astype(int)
        row = rows.setdefault(key, dict(source=0, eligible=0, onsets=[]))
        row['source'] += 1
        future = [event for event in episodes.itertuples() if event.channel_id == snapshot.channel_id and
                  snapshot.as_of < event.first_alarm_at <= snapshot.as_of + pd.Timedelta(hours=hours)]
        if any(event.left_boundary != 'observed_quiet_gap' for event in future):
            continue
        row['eligible'] += 1
        row['onsets'] += [(event.first_alarm_at - snapshot.as_of).total_seconds() / 3600 for event in future]
    return rows


class RecordingModel:
    def fit(self, matrix, labels, **kwargs):
        self.mean = float(labels.mean())
        self.rows = np.asarray(matrix)[:, 0].astype(float)
        self.labels = labels.copy()
        return self

    def predict_proba(self, matrix):
        values = np.full(len(matrix), self.mean)
        return np.column_stack([1 - values, values])


def fake_fit(variant, matrix, labels, object_ids, categories, days):
    return RecordingModel().fit(matrix, labels)


class OnsetLabelTests(unittest.TestCase):
    def compare(self, snapshots, episodes, hours):
        actual = calculate(snapshots, episodes, hours)
        expected = oracle(snapshots, episodes, hours)
        assert_array_equal(actual['object_key'], sorted(expected))
        for row in actual.itertuples():
            own = expected[row.object_key]
            self.assertEqual(row.present, own['eligible'] > 0)
            self.assertEqual(row.label, bool(own['onsets']))
            self.assertEqual(row.source_channels, own['source'])
            self.assertEqual(row.eligible_channels, own['eligible'])
            self.assertEqual(row.excluded_channels, own['source'] - own['eligible'])
            assert_allclose(row.next_hours, min(own['onsets']) if own['onsets'] else np.nan, atol=1e-9, rtol=0, equal_nan=True)
        return actual

    def test_exact_boundaries_unknown_after_known_orphans_and_censoring(self):
        moment = pd.Timestamp('2025-01-01')
        snapshots = pd.DataFrame(dict(channel_id=['a', 'b', 'c', 'orphan', 'ignored', 'excluded'], as_of=[moment] * 6,
                                     label=[0, 0, 0, 0, None, 0], split=['validation'] * 5 + ['test']))
        for hours in (72, 168):
            events = [('a', moment, 'observed_quiet_gap'), ('a', moment + pd.Timedelta(hours=1), 'observed_quiet_gap'),
                      ('a', moment + pd.Timedelta(hours=hours), 'first_observation'),
                      ('b', moment + pd.Timedelta(hours=24), 'observed_quiet_gap'),
                      ('b', moment + pd.Timedelta(hours=24), 'observed_quiet_gap'),
                      ('c', moment + pd.Timedelta(hours=hours, microseconds=1), 'observed_quiet_gap'),
                      ('orphan', moment + pd.Timedelta(hours=hours), 'observed_quiet_gap')]
            episodes = pd.DataFrame(events, columns=['channel_id', 'first_alarm_at', 'left_boundary'])
            actual = self.compare(snapshots, episodes, hours)
            first = actual[actual['object_key'] // 1_000_000 == 0].iloc[0]
            self.assertEqual(first['next_hours'], 24)
            self.assertEqual(first['eligible_channels'], 1)

    def test_random_raw_oracle_twenty_scenarios(self):
        moment = pd.Timestamp('2025-02-01')
        for seed in range(20):
            rng = np.random.default_rng(seed)
            snapshots = pd.DataFrame([(channel, moment - pd.Timedelta(days=day), 0 if rng.random() > 0.1 else None, 'validation')
                                      for channel in ['a', 'b', 'c', 'orphan'] for day in range(6)],
                                     columns=['channel_id', 'as_of', 'label', 'split'])
            events = [(str(rng.choice(['a', 'b', 'c', 'orphan'])), moment + pd.Timedelta(hours=int(rng.integers(-200, 200))),
                       ['observed_quiet_gap', 'first_observation', None][int(rng.integers(0, 3))]) for _ in range(60)]
            episodes = pd.DataFrame(events, columns=['channel_id', 'first_alarm_at', 'left_boundary'])
            for hours in (72, 168):
                self.compare(snapshots, episodes, hours)

    def test_twenty_all_unknown_schema_smokes(self):
        for seed in range(20):
            moment = pd.Timestamp('2025-02-01') + pd.Timedelta(days=seed)
            snapshots = pd.DataFrame(dict(channel_id=['a', 'b', 'c', 'orphan'], as_of=[moment] * 4, label=[0] * 4, split=['train'] * 4))
            episodes = pd.DataFrame(dict(channel_id=['a', 'b', 'c', 'orphan'],
                                        first_alarm_at=[moment + pd.Timedelta(hours=seed + 1)] * 4, left_boundary=[None] * 4))
            for hours in (72, 168):
                actual = self.compare(snapshots, episodes, hours)
                self.assertFalse(actual['present'].any())
                self.assertTrue(actual['next_hours'].isna().all())

    def test_alignment_and_union_identity_guards(self):
        shared = dict(categories=['x', 'y'], object_ids=np.array(['y', 'x', 'x']),
                      days=np.array(['2025-01-01', '2024-01-01', '2025-01-01'], dtype='datetime64[us]'))
        day = shared['days'].astype('datetime64[D]').astype(int)
        table = pd.DataFrame(dict(object_key=[day[1], day[2], 1_000_000 + day[0]], present=[True, True, True],
                                  label=[0, 1, 0], next_hours=[np.nan, 24, np.nan], source_channels=[1] * 3,
                                  eligible_channels=[1] * 3, excluded_channels=[0] * 3))
        derived = timing.align(table, shared)
        assert_array_equal(derived['label'], [0, 0, 1])
        data = dict(present=np.ones(3, bool), labels=np.array([0, 0, 1]))
        runner.verify_labels(derived, data, shared['days'], 72)
        with self.assertRaises(AssertionError):
            runner.verify_labels(derived, dict(present=data['present'], labels=np.array([1, 0, 1])), shared['days'], 72)
        with self.assertRaises(ValueError):
            timing.align(pd.concat([table, table.iloc[:1]]), shared)


class HazardTests(unittest.TestCase):
    def test_risk_set_calendar_oracle_twenty_scenarios(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            for hours in (72, 168):
                values = rng.choice([np.nan, 1, 24, 24.000001, 72] + ([100, 168] if hours == 168 else []), 100)
                lower = 0
                for lower_actual, upper, at_risk, labels in timing.risk_sets(values, hours):
                    self.assertEqual(lower_actual, lower)
                    assert_array_equal(at_risk, [np.isnan(value) or value > lower for value in values])
                    assert_array_equal(labels, [int(not np.isnan(value) and lower < value <= upper) for value in values])
                    lower = upper

    def test_conditional_factorization_recovers_empirical_union(self):
        for hours, values in [(72, [12, 24, 36, 72, np.nan]), (168, [12, 24, 36, 72, 100, 168, np.nan, np.nan])]:
            values = np.array(values)
            matrix = np.column_stack([np.arange(len(values)), np.ones(len(values))])
            days = np.arange(len(values)).astype('datetime64[D]').astype('datetime64[us]')
            with patch.object(timing.models, 'fit_model', side_effect=fake_fit):
                model = timing.fit('H_TIME_B', matrix, values, np.full(len(values), 'x'), ['x'], days, hours)
            scores = model.predict_proba(matrix)[:, 1]
            assert_allclose(scores, np.isfinite(values).mean(), atol=1e-15, rtol=0)
            curve = model.predict_curve(matrix)
            self.assertTrue(np.all(np.diff(curve, axis=1) >= 0))
            self.assertEqual(model.training[0]['rows'], len(values))
            self.assertEqual(model.training[1]['rows'], int((np.isnan(values) | (values > 24)).sum()))

    def test_constants_invalid_inputs_and_empty_risk_sets(self):
        matrix = np.ones((3, 2))
        days = np.array(['2023-01-01'] * 3, dtype='datetime64[us]')
        for variant in ('H_TIME_B', 'C_TIME_B'):
            model = timing.fit(variant, matrix, np.full(3, np.nan), np.full(3, 'x'), ['x'], days, 168)
            assert_array_equal(models.predict_model(variant, model, matrix, np.full(3, 'x'), ['x']), np.zeros(3))
        with self.assertRaisesRegex(ValueError, 'empty conditional'):
            timing.fit('H_TIME_B', matrix, np.array([1, 2, 3]), np.full(3, 'x'), ['x'], days, 72)
        for value in [0, -1, np.inf, -np.inf, 73]:
            with self.assertRaises(ValueError):
                list(timing.risk_sets([value], 72))
        for values in [np.array([[np.nan, 0]]), np.array([[1.1, 0]]), np.array([[-0.1, 0]]), np.zeros((2, 1)), np.ones(3)]:
            with self.assertRaises(ValueError):
                timing.cumulative_risk(values)

    def test_twenty_future_mutations_do_not_change_frozen_training(self):
        days = np.arange(np.datetime64('2023-01-01'), np.datetime64('2026-01-01'), dtype='datetime64[D]').astype('datetime64[us]')
        matrix = np.column_stack([np.arange(len(days)), np.ones(len(days))])
        shared = dict(days=days, object_ids=np.full(len(days), 'x'), categories=['x'])
        data = dict(present=np.ones(len(days), bool))
        future = days >= np.datetime64('2025-01-01')
        for hours in (72, 168):
            values = np.resize(np.array([12, 36, np.nan] if hours == 72 else [12, 36, 100, np.nan]), len(days))
            with patch.object(timing.models, 'fit_model', side_effect=fake_fit):
                original = runner.fit_training('H_TIME_B', matrix, data, shared, values, hours)
                for seed in range(20):
                    changed_matrix = matrix.copy()
                    changed_matrix[future] = 1000 + seed
                    changed_values = values.copy()
                    changed_values[future] = -1 - seed
                    changed = runner.fit_training('H_TIME_B', changed_matrix, data, shared, changed_values, hours)
                    self.assertEqual(changed.training, original.training)
                    assert_array_equal(changed.predict_proba(matrix[:20]), original.predict_proba(matrix[:20]))

    def test_native_model_round_trip(self):
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(900, 3))
        onsets = np.where(matrix[:, 0] > 0.4, 12, np.where(matrix[:, 1] > 0.4, 36, np.nan))
        days = np.full(len(matrix), np.datetime64('2023-01-01', 'us'))
        model = timing.fit('H_TIME_B', matrix, onsets, np.full(len(matrix), 'x'), ['x'], days, 72)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.joblib'
            models.save_verified(model, path, 'H_TIME_B', matrix, np.full(len(matrix), 'x'), ['x'])
            assert_array_equal(joblib.load(path).predict_proba(matrix), model.predict_proba(matrix))

    def test_error_audit_reconciles_intervals_and_cohorts(self):
        labels = np.array([1, 1, 0, 0])
        onsets = np.array([12, 72, np.nan, np.nan])
        history = np.array([[0, 0, 0, 0], [1, 1, 1, 0], [1, 1, 2, 0], [0, 0, 1, 0]])
        channels = dict(source_channels=np.array([2, 1, 3, 1]), eligible_channels=np.array([1, 1, 2, 1]), excluded_channels=np.array([1, 0, 1, 0]))
        audit = timing.error_audit(labels, [0.9, 0.1, 0.8, 0.2], 0.5, onsets, history, channels, 72)
        self.assertEqual(audit['confusion'], dict(tp=1, fp=1, fn=1, tn=1))
        self.assertEqual(sum(row['false_negatives'] for row in audit['onset_intervals']), 1)
        self.assertEqual(audit['source_channels'], audit['eligible_channels'] + audit['excluded_channels'])
        for variant in runner.VARIANTS:
            flags = runner.toggles(variant)
            self.assertFalse(flags['calibration'])
            self.assertFalse(flags['retraining'])


if __name__ == '__main__':
    unittest.main()
