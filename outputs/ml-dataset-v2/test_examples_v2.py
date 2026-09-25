from datetime import datetime, timedelta
from pathlib import Path
import random
import sys
import unittest
import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-dataset'))
from examples import build_examples
from examples_v2 import build_examples_v2

DAY = timedelta(days=1)
VALIDATION = datetime(2025, 1, 1)
TEST = datetime(2026, 1, 1)
T = datetime(2024, 6, 3)
HOUR = timedelta(hours=1)


def reference(observations, episodes, gaps, gap=900):
    if not observations:
        return []
    first, last = min(r[1] for r in observations), max(r[1] for r in observations)
    result = []
    for channel in sorted({r[0] for r in observations}):
        own = [r for r in observations if r[0] == channel]
        channel_first = min(r[1] for r in own)
        start = channel_first.replace(hour=0, minute=0, second=0, microsecond=0) - DAY
        for k in range((max(r[1] for r in own) - start).days + 32):
            t = start + k * DAY
            window = lambda length: [r for r in own if t - length < r[1] <= t]
            past30 = window(30 * DAY)
            if not past30:
                continue
            alarms30 = [r[1] for r in past30 if r[2]]
            starts = lambda length: sum(1 for e in episodes if e[0] == channel and t - length < e[1] <= t)
            future = [e for e in episodes if e[0] == channel and t < e[1] <= t + DAY]
            split = 'train' if t < VALIDATION else 'validation' if t < TEST else 'test'
            reasons = [(t - 30 * DAY < first, 'global_history'), (t - DAY < channel_first, 'channel_history'),
                       (t + DAY > last, 'future_window'),
                       ((split == 'train' and t + DAY >= VALIDATION) or (split == 'validation' and t + DAY >= TEST), 'split_boundary'),
                       (any(a <= t + DAY and b > t - DAY for a, b in gaps), 'known_log_gap'),
                       (bool(alarms30) and max(alarms30) >= t - timedelta(seconds=gap), 'active_alarm'),
                       (any(e[2] == 'unknown' for e in future), 'uncertain_first_episode')]
            reason = next((name for flag, name in reasons if flag), None)
            label = None if reason else int(any(e[2] == 'observed_quiet_gap' for e in future))
            overlap = sum(max(timedelta(0), min(b, t) - max(a, t - 30 * DAY)) / HOUR for a, b in gaps)
            counts = [len(window(DAY)), sum(r[2] for r in window(DAY)), len(window(7 * DAY)), sum(r[2] for r in window(7 * DAY)),
                      len(past30), len(alarms30), len({(r[1] - timedelta(microseconds=1)).date() for r in past30}), starts(7 * DAY), starts(30 * DAY)]
            result.append((channel, t, *counts, (t - max(r[1] for r in past30)) / HOUR,
                           (t - max(alarms30)) / HOUR if alarms30 else None, (t - channel_first) / DAY,
                           float(overlap), split, label, reason, 'recorded_alarm_episode'))
    return result


def load(c, observations, episodes, gaps):
    c.execute('CREATE TABLE observations(channel_id VARCHAR,occurred_at TIMESTAMP,is_alarm BOOLEAN)')
    c.execute('CREATE TABLE episodes(channel_id VARCHAR,first_alarm_at TIMESTAMP,left_boundary VARCHAR)')
    c.execute('CREATE TABLE gaps(start_at TIMESTAMP,end_at TIMESTAMP)')
    for table, rows in [('observations', observations), ('episodes', episodes), ('gaps', gaps)]:
        if rows:
            c.executemany('INSERT INTO ' + table + ' VALUES (' + ','.join(['?'] * len(rows[0])) + ')', rows)


class ExamplesV2Tests(unittest.TestCase):
    def run_case(self, observations, episodes=(), gaps=()):
        with duckdb.connect() as c:
            load(c, list(observations), list(episodes), list(gaps))
            self.assertIsNone(build_examples_v2(c, 900, VALIDATION, TEST))
            rows = c.execute('SELECT * FROM training_examples ORDER BY channel_id,as_of').fetchall()
        expected = reference(list(observations), list(episodes), list(gaps))
        self.assertEqual(len(rows), len(expected))
        for actual, wanted in zip(rows, expected):
            self.assertEqual(actual[:11] + actual[15:], wanted[:11] + wanted[15:])
            for a, w in zip(actual[11:15], wanted[11:15]):
                self.assertTrue(a == w if w is None else abs(a - w) < 1e-9, (actual, wanted))
        return rows

    def base(self):
        return [('01', T - 40 * DAY, False), ('01', T - 12 * HOUR, False), ('01', T + 2 * DAY, False)]

    def test_snapshot_existence_edges(self):
        for age in (29 * DAY, 30 * DAY - timedelta(microseconds=1), 30 * DAY, 30 * DAY + timedelta(microseconds=1)):
            with self.subTest(age=age):
                rows = self.run_case([('00', T - 60 * DAY, False), ('01', T - age, True), ('00', T + 5 * DAY, False)])
                self.assertEqual(any(r[0] == '01' and r[1] == T for r in rows), age < 30 * DAY)

    def test_island_separation(self):
        for days in (29, 30, 31, 32):
            with self.subTest(days=days):
                rows = self.run_case([('00', T - 90 * DAY, False), ('01', T - 40 * DAY, False), ('01', T - (40 - days) * DAY, False), ('00', T + 5 * DAY, False)])
                self.assertTrue(all(r[6] > 0 for r in rows))

    def test_window_and_episode_edges(self):
        for offset in (6 * DAY, 7 * DAY - timedelta(microseconds=1), 7 * DAY, 29 * DAY, 30 * DAY):
            with self.subTest(offset=offset):
                self.run_case(self.base() + [('01', T - offset, True)], [('01', T - offset, 'observed_quiet_gap')])

    def test_gap_overlap_and_labels(self):
        gaps = [(T - 31 * DAY, T - 29 * DAY), (T - 3 * DAY, T - 2 * DAY + HOUR), (T + DAY, T + DAY + HOUR)]
        self.run_case(self.base(), [('01', T + HOUR, 'observed_quiet_gap'), ('01', T + 5 * DAY, 'unknown')], gaps)

    def test_seeded_reference(self):
        rng = random.Random(2002)
        for trial in range(20):
            observations, episodes = [], []
            for channel in ('01', '02', '03'):
                for _ in range(25):
                    timestamp = T + timedelta(seconds=rng.randrange(-45 * 86400, 10 * 86400))
                    alarm = rng.random() < 0.3
                    observations.append((channel, timestamp, alarm))
                    if alarm:
                        episodes.append((channel, timestamp, rng.choice(['unknown', 'observed_quiet_gap', 'observed_quiet_gap'])))
            with self.subTest(trial=trial):
                self.run_case(observations, episodes, [(T - 10 * DAY, T - 10 * DAY + 2 * HOUR)])

    def test_v1_rows_are_subset_with_same_values(self):
        rng = random.Random(77)
        observations = [('00', T - 90 * DAY, False)] + [(c, T + timedelta(seconds=rng.randrange(-20 * 86400, 20 * 86400)), rng.random() < 0.3) for c in ('01', '02') for _ in range(60)]
        episodes = [(r[0], r[1], 'observed_quiet_gap') for r in observations if r[2]]
        with duckdb.connect() as c:
            load(c, observations, episodes, [])
            build_examples(c, 900, VALIDATION, TEST)
            v1 = c.execute('SELECT channel_id,as_of,event_count_24h,alarm_count_24h,last_event_age_hours,split,label,exclusion_reason FROM training_examples ORDER BY ALL').fetchall()
            build_examples_v2(c, 900, VALIDATION, TEST)
            v2 = c.execute('SELECT channel_id,as_of,event_count_24h,alarm_count_24h,last_event_age_hours,split,label,exclusion_reason FROM training_examples WHERE event_count_24h>0 ORDER BY ALL').fetchall()
        self.assertEqual(v1, v2)
        self.assertGreater(len(v1), 20)

    def test_future_rows_do_not_change_features(self):
        before = self.run_case(self.base())
        after = self.run_case(self.base() + [('01', T + HOUR, True)], [('01', T + HOUR, 'observed_quiet_gap')])
        pick = lambda rows: next(r[:15] for r in rows if r[1] == T)
        self.assertEqual(pick(before), pick(after))


if __name__ == '__main__':
    unittest.main()
