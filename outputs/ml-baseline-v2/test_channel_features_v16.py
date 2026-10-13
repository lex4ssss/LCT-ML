from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import duckdb
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
import channel_features_v16 as features
import explore_v16 as runner


MAPPING = {'a': ('x', 'Датчик температуры'), 'b': ('x', 'Газовый датчик'),
           'c': ('y', 'Датчик температуры'), 'd': ('x', 'Контактный датчик')}


def calculate(events, objects, moments, cutoff=None):
    with tempfile.TemporaryDirectory() as temporary, duckdb.connect() as connection:
        frame = pd.DataFrame(events, columns=['channel_id', 'occurred_at', 'raw_value', 'is_alarm'])
        frame['occurred_at'] = pd.to_datetime(frame['occurred_at'])
        frame['is_alarm'] = frame['is_alarm'].astype('boolean')
        mapping = pd.DataFrame([dict(channel_id=channel, object_id=values[0], sensor_type=values[1]) for channel, values in MAPPING.items()])
        path = Path(temporary) / 'journal.parquet'
        connection.register('fixture', frame)
        connection.execute('COPY fixture TO ? (FORMAT PARQUET)', [str(path)])
        daily = features.daily_channels(connection, [path], mapping, max(moments) if cutoff is None else cutoff)
        return features.trajectory_matrix(daily, objects, moments), features.numeric_matrix(daily, objects, moments), daily


def code(value, alarm):
    if alarm is None:
        return 18
    if not alarm:
        return 16
    return features.STATES.index(value) if value in features.STATES else 17


def day_end(moment):
    return (moment - pd.Timedelta(microseconds=1)).floor('D') + pd.Timedelta(days=1)


def oracle(events, object_id, moment):
    records = [event for event in events if MAPPING.get(event[0], ('unknown', None))[0] == object_id and event[1] <= moment]
    totals = np.zeros((3, 11))
    state_counts = np.zeros((2, 16))
    last_counts = np.zeros(19)
    ages = []
    for channel in sorted({event[0] for event in records}):
        own = [event for event in records if event[0] == channel]
        buckets = {}
        for event in own:
            buckets.setdefault(day_end(event[1]), []).append(event)
        previous = None
        previous_previous = None
        previous_switch = False
        for end in sorted(buckets):
            daily = buckets[end]
            first_at = min(event[1] for event in daily)
            last_at = max(event[1] for event in daily)
            first_codes = {code(event[2], event[3]) for event in daily if event[1] == first_at}
            last_codes = {code(event[2], event[3]) for event in daily if event[1] == last_at}
            first = next(iter(first_codes)) if len(first_codes) == 1 else 18
            last = next(iter(last_codes)) if len(last_codes) == 1 else 18
            valid = previous is not None and last != 18 and previous[0] != 18 and last_at - previous[1] <= pd.Timedelta(hours=720)
            switch = valid and last != previous[0]
            reversal = switch and previous_switch and previous_previous is not None and last == previous_previous[0]
            old_alarm = previous is not None and previous[0] < 16
            flags = [switch, reversal, len({code(event[2], event[3]) for event in daily}) > 1,
                     first != last and first != 18 and last != 18, switch and last < 16,
                     switch and old_alarm and last == 16, switch and old_alarm and last < 16,
                     switch and last == 0, switch and 7 <= last < 16, switch and last == 17, first == 18 or last == 18]
            for index, window in enumerate([1, 7, 30]):
                if end > moment - pd.Timedelta(days=window):
                    totals[index] += flags
            if switch and end > moment - pd.Timedelta(days=7):
                if last < 16:
                    state_counts[0, last] += 1
                if old_alarm:
                    state_counts[1, previous[0]] += 1
            previous_previous, previous = previous, (last, last_at)
            previous_switch = switch
        if previous is not None and moment - previous[1] < pd.Timedelta(hours=720):
            last_counts[previous[0]] += 1
            ages.append((moment - previous[1]).total_seconds() / 3600)
    shares = last_counts / last_counts.sum() if last_counts.sum() else np.full(19, np.nan)
    trajectory = np.concatenate([totals.ravel(), state_counts.ravel(), last_counts, shares,
                                 [min(ages), np.mean(ages), max(ages)] if ages else [np.nan] * 3])
    numeric = []
    for sensor_type in ['Датчик температуры', 'Газовый датчик']:
        for window in [1, 7]:
            recent_channels = 0
            scores = []
            for channel, (_, kind) in MAPPING.items():
                if kind != sensor_type:
                    continue
                values = []
                for event in records:
                    if event[0] != channel:
                        continue
                    try:
                        value = np.longdouble(event[2].replace(',', '.'))
                    except ValueError:
                        continue
                    if np.isfinite(value):
                        values.append((event[1], value))
                recent = [value for at, value in values if moment - pd.Timedelta(days=window) < at <= moment]
                if not recent:
                    continue
                recent_channels += 1
                baseline = [(at, value) for at, value in values if moment - pd.Timedelta(days=30) < at <= moment - pd.Timedelta(days=window)]
                if len(baseline) < 10 or len({day_end(at) for at, _ in baseline}) < 7:
                    continue
                base_values = np.array([value for _, value in baseline], dtype=np.longdouble)
                base_mean = base_values.mean()
                variance = ((base_values - base_mean) ** 2).mean()
                difference = np.mean(recent, dtype=np.longdouble) - base_mean
                score = np.sign(difference) * 20 if variance == 0 else np.clip(difference / np.sqrt(variance), -20, 20)
                scores.append(float(score))
            numeric.extend([recent_channels, len(scores), recent_channels - len(scores)])
            if scores:
                signed = np.array(scores)
                absolute = np.abs(signed)
                numeric.extend([absolute.max(), absolute.mean(), np.quantile(absolute, 0.9), signed.max(), signed.min(), (signed > 3).mean(), (signed < -3).mean()])
            else:
                numeric.extend([np.nan] * 7)
    return trajectory, numeric


class ChannelFeatureTests(unittest.TestCase):
    def compare(self, events, objects, moments, tolerance=1e-10):
        trajectory, numeric, daily = calculate(events, objects, moments)
        for index, (object_id, moment) in enumerate(zip(objects, moments)):
            expected_t, expected_n = oracle(events, object_id, moment)
            assert_allclose(trajectory[index], expected_t, rtol=0, atol=1e-10, equal_nan=True)
            assert_allclose(numeric[index], expected_n, rtol=0, atol=tolerance, equal_nan=True)
        return trajectory, numeric, daily

    def test_random_raw_oracles_twenty_seeds(self):
        moment = pd.Timestamp('2025-03-01')
        for seed in range(20):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(seed)
                events = [(str(rng.choice(['a', 'b', 'c', 'd', 'orphan'])), moment - pd.Timedelta(hours=int(rng.integers(-24, 1500))),
                           str(rng.choice([*features.STATES, '1,25', '-3', '4', '10', 'unknown', 'NaN', 'inf'])),
                           [False, True, None][int(rng.integers(0, 3))]) for _ in range(600)]
                self.compare(events, ['x', 'y', 'unknown', 'empty'], [moment] * 4)

    def test_all_codes_and_endpoint_ties(self):
        moment = pd.Timestamp('2025-03-01')
        events = [(f'orphan{index}', moment, state, True) for index, state in enumerate(features.STATES)]
        events += [('orphan16', moment, features.STATES[0], False), ('orphan17', moment, 'other', True),
                   ('orphan18', moment, 'other', None), ('a', moment, features.STATES[0], True), ('a', moment, features.STATES[1], True)]
        _, _, daily = self.compare(events, ['unknown', 'x'], [moment] * 2)
        own = daily[daily['channel_id'] == 'a'].iloc[0]
        self.assertEqual((own['first_code'], own['last_code']), (18, 18))

    def test_reversals_gaps_stale_status_and_exact_boundaries(self):
        moment = pd.Timestamp('2025-03-01')
        events = [('a', moment - pd.Timedelta(days=days), state, alarm) for days, state, alarm in
                  [(62, '0', False), (32, features.STATES[0], True), (2, '0', False), (1, features.STATES[0], True), (0, '0', False)]]
        events += [('d', moment - pd.Timedelta(hours=720), features.STATES[0], True),
                   ('b', moment - pd.Timedelta(hours=720, microseconds=-1), 'other', True)]
        trajectory, _, _ = self.compare(events, ['x', 'x'], [moment, moment - pd.Timedelta(days=1)])
        self.assertEqual(trajectory[0, 1], 1)
        self.assertEqual(trajectory[0, 65:84].sum(), 2)

    def test_numeric_baselines_minimums_constants_commas_and_large_values(self):
        moment = pd.Timestamp('2025-03-01')
        for seed in range(20):
            with self.subTest(seed=seed):
                anchor = 1e12 if seed % 4 == 0 else 10.0
                events = [('a', moment - pd.Timedelta(days=day, hours=offset), str(anchor + (day % 3 if seed % 2 else 0)), False)
                          for day in range(1, 31) for offset in [0, 1]]
                events += [('a', moment, str(anchor + seed), False), ('b', moment, '1,5', False),
                           ('b', moment - pd.Timedelta(days=1), 'inf', False), ('c', moment, '-3,25', False)]
                self.compare(events, ['x', 'y', 'empty'], [moment] * 3, tolerance=2e-7)

    def test_daily_variance_preserves_reading_weights(self):
        moment = pd.Timestamp('2025-03-01')
        events = [('a', moment - pd.Timedelta(days=day, minutes=offset), str(day % 3 + offset), False)
                  for day in range(1, 30) for offset in range(day % 5 + 1)]
        events += [('a', moment, '25', False)]
        self.compare(events, ['x'], [moment])

    def test_twenty_overflow_and_constant_large_value_smokes(self):
        moment = pd.Timestamp('2025-03-01')
        for seed in range(20):
            with self.subTest(seed=seed):
                magnitude = 10.0 ** (155 + 7 * seed)
                events = [('a', moment - pd.Timedelta(days=day, hours=offset), str(sign * magnitude), False)
                          for day in range(1, 30) for offset, sign in [(0, 1), (1, -1)]]
                events += [('a', moment, '0', False)]
                _, numeric, _ = calculate(events, ['x'], [moment])
                self.assertEqual(numeric[0, 0], 1)
                self.assertEqual(numeric[0, 1], 0)
                self.assertTrue(np.isnan(numeric[0, 3:10]).all())
                events = [('a', moment - pd.Timedelta(days=day, hours=offset), '1e308', False)
                          for day in range(30) for offset in [0, 1]]
                _, numeric, _ = calculate(events, ['x'], [moment])
                self.assertEqual(numeric[0, 1], 1)
                self.assertEqual(numeric[0, 3], 0)

    def test_future_records_leave_past_features_unchanged_twenty_scenarios(self):
        moment = pd.Timestamp('2025-03-01')
        for seed in range(20):
            events = [('a', moment - pd.Timedelta(days=day), str(1 + day % 3), False) for day in range(31)]
            before = calculate(events, ['x', 'y'], [moment] * 2)
            future = events + [('a', moment + pd.Timedelta(hours=1), str(1000 + seed), True),
                               ('a', moment + pd.Timedelta(days=1), features.STATES[0], True),
                               ('c', moment + pd.Timedelta(days=1), '100', False)]
            after = calculate(future, ['x', 'y'], [moment] * 2, moment + pd.Timedelta(days=2))
            for original, changed in zip(before[:2], after[:2]):
                assert_allclose(original, changed, rtol=0, atol=0, equal_nan=True)

    def test_numeric_baseline_excludes_recent_window(self):
        moment = pd.Timestamp('2025-03-01')
        events = [('a', moment - pd.Timedelta(days=day, hours=1), '10', False) for day in range(7, 30)]
        events += [('a', moment - pd.Timedelta(days=day), '20', False) for day in range(7)]
        _, numeric, _ = self.compare(events, ['x'], [moment])
        self.assertEqual(numeric[0, 13], 20)

    def test_duplicate_registry_last_wins_and_bad_query(self):
        for moments in [[pd.Timestamp('2025-03-01 01:00')], [np.datetime64('NaT', 'us')]]:
            with self.assertRaises(ValueError):
                features.query_index(['x'], moments)
        with self.assertRaises(ValueError):
            features.query_index([], [pd.Timestamp('2025-03-01')])
        with tempfile.TemporaryDirectory() as temporary, duckdb.connect() as connection:
            path = Path(temporary) / 'journal.parquet'
            connection.register('fixture', pd.DataFrame(dict(channel_id=['a'], occurred_at=[pd.Timestamp('2025-03-01')], raw_value=['1'], is_alarm=[False])))
            connection.execute('COPY fixture TO ? (FORMAT PARQUET)', [str(path)])
            mapping = pd.DataFrame(dict(channel_id=['a', 'a'], object_id=['wrong', 'x'], sensor_type=['wrong', 'Датчик температуры']))
            daily = features.daily_channels(connection, path, mapping, pd.Timestamp('2025-03-01'))
            self.assertEqual(len(daily), 1)
            self.assertEqual(daily['object_id'].iloc[0], 'x')

    def test_feature_names_and_flags(self):
        names = features.feature_names()
        self.assertEqual(len(names['trajectories']), 106)
        self.assertEqual(len(names['numeric']), 40)
        for variant in runner.VARIANTS:
            flags = runner.toggles(variant)
            self.assertFalse(flags['retraining'])
            self.assertFalse(flags['recency_weights'])
            self.assertFalse(flags['calibration'])
        with self.assertRaises(ValueError):
            runner.toggles('bad')


class RunnerGuardsTests(unittest.TestCase):
    def test_twenty_saved_score_integrity_smokes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / 'incident-H_TN-scores.npy'
            for index in range(20):
                np.save(path, [0.1, 0.2, 0.3])
                results = {'H_TN': {'score_digest': runner.adaptive.file_digest(path)}}
                runner.verify_scores(results, output, 'incident')
                np.save(path, [0.1, 0.2, 0.4 + index / 100])
                with self.assertRaisesRegex(ValueError, 'saved scores changed'):
                    runner.verify_scores(results, output, 'incident')

    def test_twenty_resume_fingerprint_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'report.json'
            for index in range(20):
                path.write_text('{"fingerprint": {"input": "original"}}')
                with self.assertRaisesRegex(ValueError, 'resume fingerprint'):
                    runner.restore_report(path, {'input': str(index)}, {})
            report = runner.restore_report(Path(temporary) / 'new.json', {'input': 'original'}, {})
            self.assertFalse(report['test_evaluation'])
            self.assertFalse(report['service_changed'])

    def test_source_input_and_code_guards(self):
        source = Path('/source')
        previous = dict(complete=True, code={'old.py': 'original'}, inputs={'/journal': 'original'})
        with patch.object(runner.adaptive, 'file_digest', return_value='changed'):
            with self.assertRaisesRegex(ValueError, 'source implementation'):
                runner.fingerprint(source, Path('/cache'), Path('/manifest'), previous)
        with patch.object(runner.adaptive, 'file_digest', side_effect=lambda path: 'original' if path.name == 'old.py' else 'changed'):
            with self.assertRaisesRegex(ValueError, 'source input'):
                runner.fingerprint(source, Path('/cache'), Path('/manifest'), previous)


if __name__ == '__main__':
    unittest.main()
