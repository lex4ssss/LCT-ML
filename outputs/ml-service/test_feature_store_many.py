from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import tempfile
import unittest
import duckdb
from feature_store import FeatureStore, InsufficientData

STATES = ['Не замкнут', 'Затоплен']
START = datetime(2024, 1, 1)


def build_journal(root, rng):
    rows, channels = [], [f'c{i}' for i in range(12)]
    for channel in channels:
        first = START + timedelta(days=rng.randint(0, 50))
        moment = first
        while moment < START + timedelta(days=75):
            alarm = rng.random() < 0.3
            state = rng.choice(STATES + ['Неисправен']) if alarm else 'Норма'
            rows.append((f'e{len(rows)}', channel, moment, alarm, state))
            moment += timedelta(seconds=rng.choice([60, 300, 800, 950, 3600, 20000, 90000]))
    rows.append(('last', 'c0', START + timedelta(days=75), True, 'Не замкнут'))
    with duckdb.connect() as c:
        c.execute('CREATE TABLE o(event_id VARCHAR, channel_id VARCHAR, occurred_at TIMESTAMP, is_alarm BOOLEAN, raw_value VARCHAR)')
        c.executemany('INSERT INTO o VALUES (?, ?, ?, ?, ?)', rows)
        c.execute(f"COPY o TO '{root / '2024.parquet'}' (FORMAT PARQUET)")
    config = root / 'config.json'
    config.write_text(json.dumps(dict(gap_intervals_utc=[dict(start_at='2024-02-20T10:00:00', end_at='2024-02-20T12:00:00')])))
    return rows, channels, config


class ManyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.rows, self.channels, config = build_journal(root, random.Random(9))
        self.store = FeatureStore(root, config, STATES)

    def tearDown(self):
        self.store.connection.close()
        self.tmp.cleanup()

    def moments(self):
        return [START + timedelta(days=d, hours=h) for d in (31, 40, 52, 60, 74) for h in (0, 7)] + [datetime(2024, 2, 21, 0)]

    def test_batch_equals_single_channel_features(self):
        served = 0
        for t in self.moments():
            try:
                _, many = self.store.features_many(self.channels, t)
            except InsufficientData as error:
                self.assertEqual(error.reason, 'known_log_gap')
                continue
            for channel in self.channels:
                try:
                    expected = self.store.features(channel, t)[1]
                except InsufficientData:
                    expected = None
                self.assertEqual(many.get(channel), expected, (channel, t))
                served += expected is not None
        self.assertGreater(served, 40)

    def test_class_history_matches_brute_force(self):
        for t in self.moments():
            starts, seen = self.store.class_history(self.channels, t)
            expected, first = [], {}
            for channel in self.channels:
                times = sorted(r[2] for r in self.rows if r[1] == channel and r[3] and r[4] in STATES)
                expected += [(channel, at) for i, at in enumerate(times)
                             if t - timedelta(days=30) < at <= t and (i == 0 or at - times[i - 1] > timedelta(seconds=900))]
                if times and times[0] <= t:
                    first[channel] = True
            self.assertEqual(sorted(starts), sorted(expected), t)
            self.assertEqual(seen, set(first))


if __name__ == '__main__':
    unittest.main()
