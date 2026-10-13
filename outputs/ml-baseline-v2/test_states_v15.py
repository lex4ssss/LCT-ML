from pathlib import Path
import tempfile
import unittest
import duckdb
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
import states_v15 as states


def registry():
    return pd.DataFrame(dict(channel_id=['a', 'b', 'c'], object_id=['x', 'x', 'y']))


def events_frame(events):
    frame = pd.DataFrame(events, columns=['channel_id', 'occurred_at', 'raw_value', 'is_alarm'])
    frame['occurred_at'] = pd.to_datetime(frame['occurred_at'])
    frame['is_alarm'] = frame['is_alarm'].astype(bool)
    frame['channel_id'] = frame['channel_id'].astype(str)
    frame['raw_value'] = frame['raw_value'].astype(str)
    return frame


def calculate(events, object_ids, moments, cutoff=None):
    with tempfile.TemporaryDirectory() as temporary, duckdb.connect() as connection:
        path = Path(temporary) / 'observations.parquet'
        connection.register('fixture', events_frame(events))
        connection.execute('COPY fixture TO ? (FORMAT PARQUET)', [str(path)])
        daily = states.daily_states(connection, [path], registry(), max(moments) if cutoff is None else cutoff)
        return states.state_matrix(daily, object_ids, moments)


def oracle(events, object_id, moment):
    objects = {'a': 'x', 'b': 'x', 'c': 'y'}
    selected = [event for event in events if event[3] and event[2] in states.STATES and
                objects.get(event[0], 'unknown') == object_id and event[1] <= moment]
    primary, context = [], []
    for state in states.STATES:
        own = [event for event in selected if event[2] == state]
        counts = [sum(moment - pd.Timedelta(days=window) < event[1] <= moment for event in own) for window in [1, 7, 30]]
        age = min((moment - max(event[1] for event in own)).total_seconds() / 3600, 720) if own else np.nan
        primary.extend(counts + [age])
        for window in [1, 7, 30]:
            history = [event for event in selected if event[1] > moment - pd.Timedelta(days=window)]
            in_state = [event for event in history if event[2] == state]
            channel_days = len({(event[0], (event[1] - pd.Timedelta(microseconds=1)).floor('D')) for event in in_state})
            context.extend([len(in_state) / len(history) if history else np.nan, channel_days])
    return primary, context


class StateTests(unittest.TestCase):
    def test_all_states_and_exact_boundaries_against_raw_oracle(self):
        moment = pd.Timestamp('2025-02-01')
        events = [(channel, moment - pd.Timedelta(hours=hours), state, True)
                  for state in states.STATES for channel in ['a', 'b'] for hours in [0, 24, 24.000001, 168, 168.000001, 720, 721, -1]]
        events += [('a', moment, states.STATES[0], False), ('a', moment, 'not-an-alarm-state', True)]
        primary, context, names = calculate(events, ['x'], [moment])
        expected_primary, expected_context = oracle(events, 'x', moment)
        assert_allclose(primary[0], expected_primary, equal_nan=True, rtol=0, atol=1e-9)
        assert_allclose(context[0], expected_context, equal_nan=True, rtol=0, atol=1e-12)
        self.assertEqual(len(names['primary']), 64)
        self.assertEqual(len(names['context']), 96)

    def test_random_irregular_histories_and_multiple_objects(self):
        moment = pd.Timestamp('2025-05-01')
        for seed in range(20):
            rng = np.random.default_rng(seed)
            events = [(str(rng.choice(['a', 'b', 'c', 'orphan'])), moment - pd.Timedelta(hours=int(rng.integers(-50, 1000))),
                       str(rng.choice(states.STATES)), bool(rng.integers(0, 2))) for _ in range(200)]
            primary, context, _ = calculate(events, ['x', 'y', 'unknown'], [moment] * 3)
            for index, object_id in enumerate(['x', 'y', 'unknown']):
                expected_primary, expected_context = oracle(events, object_id, moment)
                assert_allclose(primary[index], expected_primary, equal_nan=True, rtol=0, atol=1e-9)
                assert_allclose(context[index], expected_context, equal_nan=True, rtol=0, atol=1e-12)

    def test_future_observations_cannot_change_existing_features(self):
        moment = pd.Timestamp('2025-02-01')
        events = [('a', moment - pd.Timedelta(hours=1), states.STATES[0], True)]
        before = calculate(events, ['x', 'unknown'], [moment] * 2)
        future = events + [('a', moment + pd.Timedelta(hours=1), states.STATES[1], True),
                           ('orphan', moment + pd.Timedelta(days=1), states.STATES[0], True)]
        after = calculate(future, ['x', 'unknown'], [moment] * 2, moment + pd.Timedelta(days=2))
        for original, changed in zip(before[:2], after[:2]):
            assert_allclose(original, changed, equal_nan=True, rtol=0, atol=0)

    def test_no_alarm_states_preserve_nan_age_and_share(self):
        moment = pd.Timestamp('2025-02-01')
        primary, context, _ = calculate([('a', moment, 'normal', False)], ['x'], [moment])
        self.assertTrue(np.isnan(primary[:, 3::4]).all())
        self.assertTrue(np.isnan(context[:, 0::2]).all())
        assert_array_equal(primary[:, 0::4], 0)
        assert_array_equal(context[:, 1::2], 0)

    def test_event_counts_are_distinct_from_channel_days(self):
        moment = pd.Timestamp('2025-02-01')
        events = [('a', moment - pd.Timedelta(hours=1), states.STATES[0], True)] * 5
        primary, context, _ = calculate(events, ['x'], [moment])
        self.assertEqual(primary[0, 0], 5)
        self.assertEqual(context[0, 1], 1)

    def test_very_old_state_has_capped_age_and_zero_rolling_counts(self):
        moment = pd.Timestamp('2025-02-01')
        primary, _, _ = calculate([('a', moment - pd.Timedelta(days=100), states.STATES[0], True)], ['x'], [moment])
        assert_array_equal(primary[0, :3], [0, 0, 0])
        self.assertEqual(primary[0, 3], 720)

    def test_subdaily_and_missing_moments_are_rejected(self):
        empty = pd.DataFrame(dict(object_id=[], as_of=[], state=[], events=[], channel_days=[], last_at=[]))
        for moment in [np.datetime64('2025-01-01T01:00'), np.datetime64('NaT')]:
            with self.assertRaisesRegex(ValueError, 'midnight'):
                states.state_matrix(empty, ['x'], [moment])


if __name__ == '__main__':
    unittest.main()
