from pathlib import Path
import tempfile
import unittest
import duckdb
import numpy as np
import eval_units as eu


def write_directory(path, pairs):
    path.write_text('"ид_канала_данных","тип_датчика","ид_объект"\n' + ''.join(f'{channel},Датчик дыма,{obj}\n' for channel, obj in pairs))


def write_episodes(path, rows):
    with duckdb.connect() as c:
        c.execute('CREATE TABLE e(channel_id VARCHAR, first_alarm_at TIMESTAMP, left_boundary VARCHAR)')
        c.executemany('INSERT INTO e VALUES (?, ?::TIMESTAMP, ?)', [(channel, moment, 'observed_quiet_gap') for channel, moment in rows])
        c.execute(f"COPY e TO '{path}' (FORMAT PARQUET)")


class ObjectDayTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(1)
        n = 3000
        codes = rng.integers(0, 7, n)
        as_of = np.datetime64('2025-01-01', 'us') + rng.integers(0, 40, n) * np.timedelta64(86400 * 10**6, 'us')
        labels, scores, alarms = (rng.random(n) < 0.1).astype(np.int64), rng.random(n), rng.integers(0, 3, n).astype(float) * (rng.random(n) < 0.3)
        table = eu.object_days(codes, as_of, labels, scores, alarms)
        days = as_of.astype('datetime64[D]')
        for position, key in enumerate(table['keys']):
            rows = (codes * 1_000_000 + days.astype(np.int64)) == key
            self.assertEqual(table['label'][position], labels[rows].max())
            self.assertEqual(table['score'][position], scores[rows].max())
            self.assertEqual(table['rule'][position], float((alarms[rows] > 0).any()))
        self.assertEqual(len(table['keys']), len(set(zip(codes, days))))


class IncidentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.directory, self.episodes = root / 'channels.csv', root / 'episodes.parquet'
        write_directory(self.directory, [('1', 'A'), ('2', 'A'), ('3', 'B')])
        write_episodes(self.episodes, [
            ('1', '2025-03-01 10:00:00'), ('2', '2025-03-01 10:59:00'), ('1', '2025-03-01 11:58:00'),
            ('2', '2025-03-01 13:00:00'),
            ('3', '2025-03-01 10:30:00'),
            ('9', '2025-03-02 00:00:00'),
            ('1', '2025-03-03 23:30:00'), ('2', '2025-03-04 00:20:00'),
            ('1', '2024-12-31 23:00:00')])

    def tearDown(self):
        self.tmp.cleanup()

    def incidents(self):
        with duckdb.connect() as c:
            _, code = eu.object_codes(c, self.directory, np.array(['1']))
            keys, sizes = eu.incident_keys(c, self.episodes, self.directory, code, '2025-01-01', '2026-01-01')
        return code, keys, sizes

    def key(self, code, obj, day):
        return code[obj] * 1_000_000 + np.datetime64(day, 'D').astype(np.int64)

    def test_chains_within_gap_and_splits_after(self):
        code, keys, sizes = self.incidents()
        found = sorted(zip(keys.tolist(), sizes.tolist()))
        expected = sorted([(self.key(code, 'A', '2025-03-01'), 3), (self.key(code, 'A', '2025-03-01'), 1), (self.key(code, 'B', '2025-03-01'), 1),
                           (self.key(code, 'unknown', '2025-03-01'), 1), (self.key(code, 'A', '2025-03-03'), 2)])
        self.assertEqual(found, expected)

    def test_report_counts_alerted_and_eligible(self):
        code, keys, sizes = self.incidents()
        table = dict(keys=np.array([self.key(code, 'A', '2025-03-01'), self.key(code, 'B', '2025-03-01')]), score=np.array([0.9, 0.1]))
        report = eu.incident_report(keys, sizes, table, 0.5)
        self.assertEqual((report['incidents'], report['episodes'], report['with_eligible_object_day'], report['in_alerted_object_day']), (5, 8, 3, 2))
        self.assertAlmostEqual(report['recall'], 2 / 5)
        self.assertAlmostEqual(report['recall_among_eligible'], 2 / 3)
        self.assertAlmostEqual(report['single_channel_share'], 3 / 5)


if __name__ == '__main__':
    unittest.main()
