import json
from pathlib import Path
import sys
import duckdb
import numpy as np

prepared, config_path, sample_size, seed = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
config = json.loads(config_path.read_text())
DAY = 86400 * 10**6
GAP = config['gap_seconds'] * 10**6
to_us = lambda text: int(np.datetime64(text, 'us').astype(np.int64))
validation_start, test_start = to_us(config['validation_start_utc']), to_us(config['test_start_utc'])
log_gaps = [(to_us(g['start_at']), to_us(g['end_at'])) for g in config['gap_intervals_utc']]
connection = duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'})
years = [str(p) for p in sorted(prepared.glob('20??.parquet'))]
connection.execute('CREATE TABLE bounds AS SELECT channel_id,min(occurred_at) first_at,max(occurred_at) last_at FROM read_parquet(?) GROUP BY channel_id', [years])
global_first, global_last = connection.execute('SELECT epoch_us(min(first_at)),epoch_us(max(last_at)) FROM bounds').fetchone()
channels = [r[0] for r in connection.execute('SELECT channel_id FROM bounds ORDER BY hash(channel_id||?) LIMIT ?', [str(seed), sample_size]).fetchall()]
connection.execute('CREATE TABLE o AS SELECT channel_id,epoch_us(occurred_at) t,is_alarm FROM read_parquet(?) WHERE channel_id IN (SELECT unnest(?))', [years, channels])


def expected_rows(channel):
    data = connection.execute('SELECT t,is_alarm FROM o WHERE channel_id=? ORDER BY t', [channel]).fetchnumpy()
    times, alarms = data['t'].astype(np.int64), data['is_alarm'].astype(bool)
    alarm_times = times[alarms]
    starts, unknown = [], []
    for i, t in enumerate(alarm_times):
        if i == 0 or t - alarm_times[i - 1] > GAP:
            starts.append(t)
            unknown.append(i == 0 and t - times[0] <= GAP)
    starts, unknown = np.array(starts, dtype=np.int64), np.array(unknown, dtype=bool)
    buckets = ((times - 1) // DAY + 1) * DAY
    rows = []
    for t in np.unique(buckets):
        in_window = (times > t - DAY) & (times <= t)
        window_alarms = times[in_window & alarms]
        last_alarm = window_alarms.max() if window_alarms.size else None
        future = (starts > t) & (starts <= t + DAY)
        split = 'train' if t < validation_start else 'validation' if t < test_start else 'test'
        checks = [(t - DAY < global_first, 'global_history'), (t - DAY < times[0], 'channel_history'),
                  (t + DAY > global_last, 'future_window'),
                  ((split == 'train' and t + DAY >= validation_start) or (split == 'validation' and t + DAY >= test_start), 'split_boundary'),
                  (any(a <= t + DAY and b > t - DAY for a, b in log_gaps), 'known_log_gap'),
                  (last_alarm is not None and last_alarm >= t - GAP, 'active_alarm'),
                  (bool((future & unknown).any()), 'uncertain_first_episode')]
        reason = next((name for flag, name in checks if flag), None)
        label = None if reason else int((future & ~unknown).any())
        rows.append((channel, int(t), int(in_window.sum()), int(window_alarms.size), (t - times[in_window].max()) / 3.6e9,
                     None if last_alarm is None else (t - last_alarm) / 3.6e9, split, label, reason))
    return rows


mismatches, compared = [], 0
for channel in channels:
    expected = expected_rows(channel)
    actual = connection.execute('SELECT channel_id,epoch_us(as_of),event_count_24h,alarm_count_24h,last_event_age_hours,last_alarm_age_hours_24h,split,label,exclusion_reason FROM read_parquet(?) WHERE channel_id=? ORDER BY as_of', [str(prepared / 'training-examples.parquet'), channel]).fetchall()
    compared += len(expected)
    if len(actual) != len(expected):
        mismatches.append(dict(channel=channel, expected_rows=len(expected), actual_rows=len(actual)))
        continue
    for e, a in zip(expected, actual):
        same = e[:4] == a[:4] and e[6:] == a[6:] and abs(e[4] - a[4]) < 1e-9 and ((e[5] is None and a[5] is None) or (e[5] is not None and a[5] is not None and abs(e[5] - a[5]) < 1e-9))
        if not same:
            mismatches.append(dict(expected=[str(v) for v in e], actual=[str(v) for v in a]))
print(json.dumps(dict(channels=len(channels), rows_compared=compared, mismatches=len(mismatches), first_mismatches=mismatches[:5]), indent=2))
