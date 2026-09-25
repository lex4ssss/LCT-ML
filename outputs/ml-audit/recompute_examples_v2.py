import json
from pathlib import Path
import sys
import duckdb
import numpy as np

prepared, sample_size, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
DAY = 86400 * 10**6
connection = duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'})
years = [str(p) for p in sorted(prepared.glob('20??.parquet'))]
examples = str(prepared / 'training-examples-v2.parquet')
channels = [r[0] for r in connection.execute('SELECT DISTINCT channel_id FROM read_parquet(?) ORDER BY hash(channel_id||?) LIMIT ?', [examples, str(seed), sample_size]).fetchall()]
connection.execute('CREATE TABLE o AS SELECT channel_id,epoch_us(occurred_at) t,is_alarm FROM read_parquet(?) WHERE channel_id IN (SELECT unnest(?))', [years, channels])
connection.execute('CREATE TABLE e AS SELECT channel_id,epoch_us(first_alarm_at) s FROM read_parquet(?) WHERE channel_id IN (SELECT unnest(?))', [str(prepared / 'episodes-900.parquet'), channels])


def count_in(sorted_times, grid, length):
    return np.searchsorted(sorted_times, grid, 'right') - np.searchsorted(sorted_times, grid - length, 'right')


mismatch, compared, missing_days, extra_days = {}, 0, 0, 0
for channel in channels:
    data = connection.execute('SELECT t,is_alarm FROM o WHERE channel_id=? ORDER BY t', [channel]).fetchnumpy()
    times, alarms = data['t'].astype(np.int64), data['is_alarm'].astype(bool)
    alarm_times = times[alarms]
    starts = np.sort(connection.execute('SELECT s FROM e WHERE channel_id=?', [channel]).fetchnumpy()['s'].astype(np.int64))
    buckets = np.unique(((times - 1) // DAY + 1) * DAY)
    candidates = np.unique((buckets[:, None] + np.arange(31)[None, :] * DAY).ravel())
    expected_grid = candidates[count_in(times, candidates, 30 * DAY) > 0]
    actual = connection.execute('SELECT epoch_us(as_of),event_count_24h,alarm_count_24h,event_count_7d,alarm_count_7d,event_count_30d,alarm_count_30d,active_days_30d,episode_starts_7d,episode_starts_30d,last_event_age_hours,last_alarm_age_hours_30d FROM read_parquet(?) WHERE channel_id=? ORDER BY as_of', [examples, channel]).fetchnumpy()
    grid = actual['epoch_us(as_of)'].astype(np.int64)
    missing_days += len(np.setdiff1d(expected_grid, grid))
    extra_days += len(np.setdiff1d(grid, expected_grid))
    grid = np.intersect1d(grid, expected_grid)
    keep = np.isin(actual['epoch_us(as_of)'], grid)
    expected = {'event_count_24h': count_in(times, grid, DAY), 'alarm_count_24h': count_in(alarm_times, grid, DAY),
                'event_count_7d': count_in(times, grid, 7 * DAY), 'alarm_count_7d': count_in(alarm_times, grid, 7 * DAY),
                'event_count_30d': count_in(times, grid, 30 * DAY), 'alarm_count_30d': count_in(alarm_times, grid, 30 * DAY),
                'active_days_30d': count_in(buckets, grid, 30 * DAY),
                'episode_starts_7d': count_in(starts, grid, 7 * DAY), 'episode_starts_30d': count_in(starts, grid, 30 * DAY)}
    last_event = times[np.searchsorted(times, grid, 'right') - 1]
    expected['last_event_age_hours'] = (grid - last_event) / 3.6e9
    alarm_index = np.searchsorted(alarm_times, grid, 'right') - 1
    last_alarm = np.where(alarm_index >= 0, np.append(alarm_times, 0)[np.maximum(alarm_index, 0)], -(2**62))
    expected['last_alarm_age_hours_30d'] = np.where(grid - last_alarm < 30 * DAY, (grid - last_alarm) / 3.6e9, np.nan)
    for name, values in expected.items():
        got = np.ma.filled(actual[name][keep].astype(float), np.nan)
        bad = ~np.isclose(got, values.astype(float), rtol=0, atol=1e-9, equal_nan=True)
        mismatch[name] = mismatch.get(name, 0) + int(bad.sum())
    compared += len(grid)
print(json.dumps(dict(channels=len(channels), rows_compared=compared, missing_days=missing_days, extra_days=extra_days, field_mismatches=mismatch)))
