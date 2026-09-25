from pathlib import Path
import tempfile
import unittest
import duckdb
import object_features as of

ROWS = [
    ('1', '2025-03-01', 'train', 0, 2, 5, 1, 3),
    ('2', '2025-03-01', 'train', 1, 0, 1, 0, 1),
    ('3', '2025-03-01', 'train', None, 4, 4, 2, 2),
    ('4', '2025-03-01', 'train', 0, 1, 1, 1, 1),
    ('1', '2025-03-02', 'train', 0, 0, 3, 0, 3),
    ('9', '2025-03-01', 'train', 0, 7, 7, 1, 1),
]
DIRECTORY = [('1', 'A', 'Датчик дыма', 'Пожарная охрана'), ('2', 'A', 'Датчик дыма', 'Пожарная охрана'),
             ('3', 'A', 'КД Дверь', 'Охранная подсистема'), ('4', 'B', 'Датчик дыма', 'Пожарная охрана')]


class ObjectFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        examples, directory, output = root / 'examples.parquet', root / 'channels.csv', root / 'features.parquet'
        directory.write_text('"ид_канала_данных","тип_инж_системы","тип_датчика","ид_объект"\n' + ''.join(f'{c},{s},{t},{o}\n' for c, o, t, s in DIRECTORY))
        with duckdb.connect() as c:
            c.execute('CREATE TABLE e(channel_id VARCHAR, as_of TIMESTAMP, split VARCHAR, label INTEGER, alarm_count_24h DOUBLE, alarm_count_7d DOUBLE, episode_starts_7d DOUBLE, episode_starts_30d DOUBLE)')
            c.executemany('INSERT INTO e VALUES (?, ?::TIMESTAMP, ?, ?, ?, ?, ?, ?)', ROWS)
            c.execute(f"COPY e TO '{examples}' (FORMAT PARQUET)")
            of.build(c, examples, directory, output)
            cursor = c.execute(f"SELECT * FROM read_parquet('{output}')")
            names = [d[0] for d in cursor.description]
            cls.rows = {(r[0], str(r[1])[:10]): dict(zip(names, r)) for r in cursor.fetchall()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_other_channels_exclude_self_and_other_days(self):
        row = self.rows[('1', '2025-03-01')]
        self.assertEqual([row[name] for name in of.OBJECT_FEATURES], [2, 1, 4, 5, 2, 3, 1])

    def test_excluded_rows_still_feed_neighbours(self):
        row = self.rows[('2', '2025-03-01')]
        self.assertEqual((row['obj_other_alarm_24h'], row['obj_other_alarm_channels_24h'], row['type_other_alarm_7d']), (6, 2, 5))

    def test_single_channel_object_and_next_day(self):
        self.assertEqual([self.rows[('4', '2025-03-01')][name] for name in of.OBJECT_FEATURES], [0] * 7)
        self.assertEqual(self.rows[('1', '2025-03-02')]['obj_channels_30d'], 0)

    def test_unknown_channel_gets_nulls_and_keeps_label(self):
        row = self.rows[('9', '2025-03-01')]
        self.assertTrue(all(row[name] is None for name in of.OBJECT_FEATURES))
        self.assertIsNone(row['sensor_type'])
        self.assertEqual((row['label'], len(self.rows)), (0, 6))

    def test_categories_copied(self):
        self.assertEqual((self.rows[('3', '2025-03-01')]['sensor_type'], self.rows[('3', '2025-03-01')]['system_type']), ('КД Дверь', 'Охранная подсистема'))


if __name__ == '__main__':
    unittest.main()
