from pathlib import Path
import tempfile
import unittest
import duckdb
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
import families_v14 as families


def write_parquet(connection, rows, path):
    connection.register('fixture_rows', rows)
    connection.execute('COPY fixture_rows TO ? (FORMAT PARQUET)', [str(path)])


def mapping():
    return pd.DataFrame(dict(channel_id=['s', 'w', 'g', 'a', 'z'], object_code=[0, 0, 0, 0, 0], head=[0, 1, 2, 3, 4]))


def oracle(snapshots, episodes, channel_map, unknown_code, hours=72):
    result = {}
    objects = {row.channel_id: (row.object_code, row.head) for row in channel_map.itertuples()}
    for row in snapshots.itertuples():
        if pd.isna(row.label) or row.split not in ['train', 'validation'] or row.as_of >= pd.Timestamp('2026-01-01'):
            continue
        code, head = objects.get(row.channel_id, (unknown_code, 4))
        key = code * 1_000_000 + np.datetime64(row.as_of, 'D').astype(int)
        record = result.setdefault(key, dict(present=False, label=0, heads=np.zeros(5, int)))
        future = [event for event in episodes.itertuples() if event.channel_id == row.channel_id and
                  row.as_of < event.first_alarm_at <= row.as_of + pd.Timedelta(hours=hours)]
        if any(event.left_boundary != 'observed_quiet_gap' for event in future):
            continue
        record['present'] = True
        positive = int(any(event.left_boundary == 'observed_quiet_gap' for event in future))
        record['label'] = max(record['label'], positive)
        record['heads'][head] = max(record['heads'][head], positive)
    return result


class LabelTests(unittest.TestCase):
    def calculate(self, snapshots, episodes, hours=72):
        with tempfile.TemporaryDirectory() as temporary, duckdb.connect() as connection:
            examples, starts = Path(temporary) / 'examples.parquet', Path(temporary) / 'episodes.parquet'
            write_parquet(connection, snapshots, examples)
            write_parquet(connection, episodes, starts)
            return families.label_table(connection, examples, starts, mapping(), 1, hours)

    def assert_oracle(self, snapshots, episodes, hours=72):
        expected = oracle(snapshots, episodes, mapping(), 1, hours)
        actual = self.calculate(snapshots, episodes, hours)
        assert_array_equal(actual['object_key'], sorted(expected))
        for row in actual.itertuples():
            record = expected[row.object_key]
            self.assertEqual(row.present, record['present'])
            self.assertEqual(row.label, record['label'])
            heads = np.array([getattr(row, f'family_{index}') for index in range(5)])
            assert_array_equal(heads, record['heads'])
            self.assertEqual(heads.max(), row.label)

    def test_exact_boundaries_unknown_precedence_and_orphans(self):
        moment = pd.Timestamp('2025-01-01')
        snapshots = pd.DataFrame(dict(channel_id=['s', 'w', 'g', 'a', 'z', 'orphan', 'ignored'],
            as_of=[moment] * 7, label=[0, 0, 0, 0, 0, 0, None], split=['validation'] * 7))
        episodes = pd.DataFrame(dict(channel_id=['s', 'w', 'g', 'a', 'a', 'z', 'orphan'],
            first_alarm_at=[moment, moment + pd.Timedelta(hours=72), moment + pd.Timedelta(hours=73),
                            moment + pd.Timedelta(hours=1), moment + pd.Timedelta(hours=2),
                            moment + pd.Timedelta(hours=100), moment + pd.Timedelta(hours=3)],
            left_boundary=['observed_quiet_gap', 'observed_quiet_gap', 'observed_quiet_gap',
                           'observed_quiet_gap', 'unknown', 'unknown', 'observed_quiet_gap']))
        self.assert_oracle(snapshots, episodes)

    def test_random_channel_oracle_cases(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            origin = pd.Timestamp('2025-03-01')
            channels = ['s', 'w', 'g', 'a', 'z', 'orphan']
            snapshots = pd.DataFrame([dict(channel_id=channel, as_of=origin + pd.Timedelta(days=day), label=0, split='validation')
                                      for channel in channels for day in [0, 1, 2]])
            episodes = pd.DataFrame([dict(channel_id=channel, first_alarm_at=origin + pd.Timedelta(hours=int(hour)),
                                           left_boundary='observed_quiet_gap' if index % 4 else 'unknown')
                                      for channel in channels for index, hour in enumerate(rng.integers(-50, 150, 10))])
            self.assert_oracle(snapshots, episodes)
            self.assert_oracle(snapshots, episodes, 168)

    def test_unknown_only_object_is_not_negative_eligible_example(self):
        snapshots = pd.DataFrame(dict(channel_id=['orphan'], as_of=[pd.Timestamp('2025-01-01')], label=[0], split=['validation']))
        episodes = pd.DataFrame(dict(channel_id=['orphan'], first_alarm_at=[pd.Timestamp('2025-01-02')], left_boundary=['unknown']))
        result = self.calculate(snapshots, episodes)
        self.assertFalse(result.present.iloc[0])
        self.assert_oracle(snapshots, episodes)

    def test_test_snapshots_are_excluded(self):
        snapshots = pd.DataFrame(dict(channel_id=['s', 'w'], as_of=[pd.Timestamp('2025-01-01'), pd.Timestamp('2026-01-01')],
                                       label=[0, 0], split=['validation', 'test']))
        episodes = pd.DataFrame(dict(channel_id=['w'], first_alarm_at=[pd.Timestamp('2026-01-02')], left_boundary=['observed_quiet_gap']))
        self.assert_oracle(snapshots, episodes)


class HistoryTests(unittest.TestCase):
    def calculate(self, episodes, moments):
        days = np.asarray(moments).astype('datetime64[D]').astype(int)
        keys = np.r_[days, 1_000_000 + days]
        with tempfile.TemporaryDirectory() as temporary, duckdb.connect() as connection:
            path = Path(temporary) / 'episodes.parquet'
            write_parquet(connection, episodes, path)
            return families.family_history(connection, path, mapping(), keys, 1)

    def test_windows_against_independent_episode_scan(self):
        moment = pd.Timestamp('2025-02-01')
        episodes = pd.DataFrame([dict(channel_id=channel, first_alarm_at=moment - pd.Timedelta(hours=hours))
                                 for channel in ['s', 'w', 'g', 'a', 'z', 'orphan'] for hours in [-1, 0, 24, 25, 168, 169, 720, 800]])
        actual, names = self.calculate(episodes, [moment])
        for head, name in enumerate(families.FAMILIES):
            for window in [24, 168, 720]:
                expected = sum(0 <= age < window for age in [-1, 0, 24, 25, 168, 169, 720, 800])
                self.assertEqual(actual[0, names.index(f'{name}_starts_{window}h')], expected)
            self.assertEqual(actual[0, names.index(f'{name}_last_start_age')], 0)
        self.assertEqual(actual[1, names.index('other_starts_24h')], 1)

    def test_missing_family_is_nan_history_and_future_does_not_change_it(self):
        moment = pd.Timestamp('2025-02-01')
        episodes = pd.DataFrame(dict(channel_id=['s'], first_alarm_at=[moment - pd.Timedelta(hours=900)]))
        before, names = self.calculate(episodes, [moment])
        extended = pd.concat([episodes, pd.DataFrame(dict(channel_id=['w'], first_alarm_at=[moment + pd.Timedelta(hours=1)]))])
        after, _ = self.calculate(extended, [moment])
        assert_allclose(before, after, equal_nan=True, rtol=0, atol=0)
        self.assertTrue(np.isnan(before[0, names.index('water_last_start_age')]))
        self.assertEqual(before[0, names.index('smoke_temperature_last_start_age')], 720)


class HeadTests(unittest.TestCase):
    def test_partition_is_disjoint_and_complete(self):
        names = set().union(*families.TYPES)
        self.assertEqual(sum(len(group) for group in families.TYPES), len(names))
        for head, group in enumerate(families.TYPES):
            for name in group:
                self.assertEqual(families.family_code(name), head)
        self.assertEqual(families.family_code('unregistered'), 4)

    def test_combinations_against_independent_scalar_formula(self):
        scores = np.random.default_rng(0).uniform(size=(20, 5))
        assert_allclose(families.combine(scores, 'HEAD_MAX'), [max(row) for row in scores])
        expected = []
        for row in scores:
            product = 1
            for value in row:
                product *= 1 - value
            expected.append(1 - product)
        assert_allclose(families.combine(scores, 'HEAD_OR'), expected)

    def test_constant_heads_and_serialization(self):
        matrix = np.ones((10, 3))
        for value in [0, 1]:
            model = families.fit_head(matrix, np.full(10, value))
            assert_array_equal(model.predict_proba(matrix)[:, 1], np.full(10, value))
            with tempfile.TemporaryDirectory() as temporary:
                families.save_verified(model, Path(temporary) / 'model.joblib', matrix)

    def test_invalid_combinations_are_rejected(self):
        for values in [np.zeros((2, 4)), np.full((2, 5), np.nan), np.full((2, 5), -0.1), np.full((2, 5), 1.1)]:
            with self.assertRaises(ValueError):
                families.combine(values, 'HEAD_MAX')

    def test_invalid_head_classes_are_rejected(self):
        for values in [[], [2, 2], [0, 2], [np.nan], [0.5]]:
            with self.assertRaisesRegex(ValueError, 'binary'):
                families.fit_head(np.ones((len(values), 2)), np.array(values))


if __name__ == '__main__':
    unittest.main()
