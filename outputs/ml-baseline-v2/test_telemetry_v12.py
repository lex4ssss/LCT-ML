from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
import duckdb
import numpy as np
import pandas as pd
import telemetry_v12 as telemetry


def channel_registry():
    return pd.DataFrame([
        dict(channel_id='t', object_id='a', sensor_type='Датчик температуры', picket='1'),
        dict(channel_id='g', object_id='a', sensor_type='Газовый датчик', picket='1'),
        dict(channel_id='b', object_id='b', sensor_type='Газовый датчик', picket=None),
    ])


def daily_from_events(events, cutoff):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'observations.parquet'
        with duckdb.connect() as connection:
            frame = pd.DataFrame(events, columns=['channel_id', 'occurred_at', 'raw_value'])
            connection.register('input_events', frame)
            connection.execute('COPY input_events TO ? (FORMAT PARQUET)', [str(path)])
            return telemetry.numeric_daily(connection, path, channel_registry(), cutoff)


def numeric_oracle(events, moment, window, sensor):
    start = moment - timedelta(days=window)
    values = []
    for channel, occurred_at, raw in events:
        if channel != sensor or not start < occurred_at <= moment:
            continue
        try:
            value = float(raw.replace(',', '.'))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return (len(values), np.mean(values), np.std(values), min(values), max(values)) if values else (0, np.nan, np.nan, np.nan, np.nan)


class NumericFeaturesTests(unittest.TestCase):
    def test_exact_windows_against_independent_raw_oracle(self):
        moment = datetime(2025, 3, 2)
        events = []
        for age, value in [(0, 30), (0.5, 20), (1, 10), (6.5, 40), (7, 50), (29.5, -5), (30, 99)]:
            events.append(('t', moment - timedelta(days=age), str(value)))
        events += [('g', moment - timedelta(hours=2), '0,5'), ('g', moment, '1'), ('t', moment + timedelta(seconds=1), '999')]
        daily = daily_from_events(events, moment)
        values, names = telemetry.numeric_matrix(daily, ['a'], np.array([moment], dtype='datetime64[us]'))
        for window in [1, 7, 30]:
            for prefix, sensor in [('temp', 't'), ('gas', 'g')]:
                expected = numeric_oracle(events, moment, window, sensor)
                found = [values[0, names.index(f'{prefix}_{stat}_{window}d')] for stat in ['n', 'mean', 'std', 'min', 'max']]
                np.testing.assert_allclose(found, expected, equal_nan=True)

    def test_future_records_do_not_change_past_features(self):
        moment = datetime(2025, 2, 1)
        events = [('t', moment - timedelta(hours=1), '12'), ('g', moment, '0')]
        future = events + [('t', moment + timedelta(hours=1), '1000'), ('g', moment + timedelta(days=100), '1000')]
        first, names = telemetry.numeric_matrix(daily_from_events(events, moment), ['a'], np.array([moment], dtype='datetime64[us]'))
        second, later_names = telemetry.numeric_matrix(daily_from_events(future, moment + timedelta(days=101)), ['a'], np.array([moment], dtype='datetime64[us]'))
        self.assertEqual(names, later_names)
        np.testing.assert_allclose(first, second, equal_nan=True)

    def test_missing_history_is_not_zero_temperature(self):
        moment = datetime(2025, 2, 1)
        events = [('b', moment, '3')]
        matrix, names = telemetry.numeric_matrix(daily_from_events(events, moment), ['a'], np.array([moment], dtype='datetime64[us]'))
        self.assertTrue(np.isnan(matrix[0, names.index('temp_mean_1d')]))
        self.assertTrue(np.isnan(matrix[0, names.index('gas_mean_1d')]))

    def test_infinite_values_and_text_do_not_enter_numeric_values(self):
        moment = datetime(2025, 2, 1)
        events = [('g', moment, raw) for raw in ['nan', 'inf', '-inf', 'Обнаружен газ', '0', '2,5']]
        matrix, names = telemetry.numeric_matrix(daily_from_events(events, moment), ['a'], np.array([moment], dtype='datetime64[us]'))
        self.assertEqual(matrix[0, names.index('gas_n_1d')], 2)
        self.assertEqual(matrix[0, names.index('gas_mean_1d')], 1.25)
        self.assertEqual(matrix[0, names.index('gas_positive_share_1d')], 0.5)

    def test_negative_temperature_and_clock_are_features_not_labels(self):
        moment = datetime(2025, 2, 1)
        events = [('t', moment, '-10'), ('g', moment, '01.01.1970 03:00:00/01')]
        daily = daily_from_events(events, moment)
        self.assertNotIn('label', daily.columns)
        matrix, names = telemetry.numeric_matrix(daily, ['a'], np.array([moment], dtype='datetime64[us]'))
        self.assertEqual(matrix[0, names.index('temp_mean_1d')], -10)
        self.assertEqual(matrix[0, names.index('negative_share_1d')], 0.5)
        self.assertEqual(matrix[0, names.index('clock_share_1d')], 0.5)

    def test_object_statistics_remain_separate(self):
        moment = datetime(2025, 2, 1)
        events = [('g', moment, '1'), ('b', moment, '10')]
        matrix, names = telemetry.numeric_matrix(daily_from_events(events, moment), ['b', 'a'], np.array([moment, moment], dtype='datetime64[us]'))
        np.testing.assert_equal(matrix[:, names.index('gas_mean_1d')], [10, 1])

    def test_irregular_observation_days_preserve_calendar_windows(self):
        moment = datetime(2025, 2, 1)
        events = [('t', moment - timedelta(days=40), '100'), ('t', moment - timedelta(days=6), '10')]
        matrix, names = telemetry.numeric_matrix(daily_from_events(events, moment), ['a'], np.array([moment], dtype='datetime64[us]'))
        self.assertTrue(np.isnan(matrix[0, names.index('temp_mean_1d')]))
        self.assertEqual(matrix[0, names.index('temp_mean_7d')], 10)
        self.assertEqual(matrix[0, names.index('temp_mean_30d')], 10)

    def test_random_oracle_cases(self):
        moment = datetime(2025, 2, 1)
        for seed in range(20):
            rng = np.random.default_rng(seed)
            events = [(sensor, moment - timedelta(hours=int(age)), str(float(value)))
                      for sensor in ['t', 'g'] for age, value in zip(rng.integers(-50, 900, size=30), rng.normal(size=30))]
            daily = daily_from_events(events, moment + timedelta(days=3))
            matrix, names = telemetry.numeric_matrix(daily, ['a'], np.array([moment], dtype='datetime64[us]'))
            for window in [1, 7, 30]:
                for prefix, sensor in [('temp', 't'), ('gas', 'g')]:
                    expected = numeric_oracle(events, moment, window, sensor)
                    found = [matrix[0, names.index(f'{prefix}_{stat}_{window}d')] for stat in ['n', 'mean', 'std', 'min', 'max']]
                    np.testing.assert_allclose(found, expected, atol=1e-12, equal_nan=True)


class DomainTests(unittest.TestCase):
    def columns(self, ages, alarms24, alarms7=None, alarms30=None):
        count = len(ages)
        return dict(channel_id=np.array(['t', 'g'][:count]), as_of=np.array(['2025-02-01'] * count, dtype='datetime64[us]'),
                    last_event_age_hours=np.asarray(ages, float), alarm_count_24h=np.asarray(alarms24, float),
                    alarm_count_7d=np.asarray(alarms24 if alarms7 is None else alarms7, float),
                    alarm_count_30d=np.asarray(alarms24 if alarms30 is None else alarms30, float))

    def calculate(self, columns, mapping=None):
        keys = np.array([np.datetime64('2025-02-01', 'D').astype(np.int64)])
        return telemetry.domain_matrix(columns, channel_registry() if mapping is None else mapping, keys, {'a': 0, 'b': 1, 'unknown': 2})

    def test_same_picket_confirms_two_observed_alarms(self):
        matrix, names = self.calculate(self.columns([1, 2], [3, 4]))
        self.assertEqual(matrix[0, names.index('domain_confirmed_24h')], 1)
        self.assertEqual(matrix[0, names.index('domain_max_alarm_channels_24h')], 2)
        self.assertEqual(matrix[0, names.index('domain_max_alarm_share_24h')], 1)

    def test_silent_neighbor_is_not_treated_as_healthy(self):
        matrix, names = self.calculate(self.columns([1, 100], [1, 0]))
        self.assertEqual(matrix[0, names.index('domain_singleton_24h')], 0)
        self.assertEqual(matrix[0, names.index('domain_ambiguous_24h')], 1)

    def test_left_window_boundary_is_excluded(self):
        matrix, names = self.calculate(self.columns([1, 24], [1, 0]))
        self.assertEqual(matrix[0, names.index('domain_singleton_24h')], 0)
        self.assertEqual(matrix[0, names.index('domain_ambiguous_24h')], 1)

    def test_different_pickets_do_not_confirm_each_other(self):
        mapping = channel_registry()
        mapping.loc[mapping['channel_id'] == 'g', 'picket'] = '2'
        matrix, names = self.calculate(self.columns([1, 1], [1, 1]), mapping)
        self.assertEqual(matrix[0, names.index('domain_confirmed_24h')], 0)
        for window in telemetry.DOMAIN_WINDOWS:
            self.assertTrue(np.isnan(matrix[0, names.index(f'domain_max_alarm_share_{window}h')]))

    def test_windows_are_independent(self):
        matrix, names = self.calculate(self.columns([1, 100], [1, 0], [1, 1], [1, 1]))
        self.assertEqual(matrix[0, names.index('domain_confirmed_24h')], 0)
        self.assertEqual(matrix[0, names.index('domain_confirmed_168h')], 1)

    def test_zero_age_is_observed(self):
        matrix, names = self.calculate(self.columns([0, 0], [1, 0]))
        self.assertEqual(matrix[0, names.index('domain_singleton_24h')], 1)

    def test_alarm_without_observation_cannot_confirm_domain(self):
        for age in [24, 25, np.nan, -1]:
            matrix, names = self.calculate(self.columns([1, age], [1, 1]))
            self.assertEqual(matrix[0, names.index('domain_confirmed_24h')], 0)
            self.assertEqual(matrix[0, names.index('domain_ambiguous_24h')], 1)
            self.assertEqual(matrix[0, names.index('domain_max_alarm_share_24h')], 1)

    def test_picket_normalization_and_ambiguous_names(self):
        self.assertEqual(telemetry.parse_picket('ДД ПК 010,00'), '10')
        self.assertEqual(telemetry.parse_picket('пикет 10.0 ПК10'), '10')
        self.assertIsNone(telemetry.parse_picket('ПК10 и ПК11'))
        self.assertIsNone(telemetry.parse_picket('Датчик без пикета'))


if __name__ == '__main__':
    unittest.main()
