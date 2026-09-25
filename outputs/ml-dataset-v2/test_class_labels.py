from pathlib import Path
import tempfile
import unittest
import duckdb
import class_labels as cl

OBSERVATIONS = [
    ('1', '2025-03-01 00:00:00', False, 'Норма'),
    ('1', '2025-03-01 10:00:00', True, 'Неисправен'),
    ('1', '2025-03-01 10:05:00', True, 'Обнаружен дым'),
    ('1', '2025-03-01 10:10:00', True, 'Неисправен'),
    ('1', '2025-03-01 12:00:00', True, 'Обнаружен дым'),
    ('2', '2025-03-01 00:00:00', False, 'Норма'),
    ('2', '2025-03-01 00:10:00', True, 'Не замкнут'),
    ('2', '2025-03-02 05:00:00', True, 'Обесточен'),
    ('3', '2025-03-01 00:00:00', False, 'Норма'),
    ('3', '2025-03-01 09:00:00', False, 'Неисправен'),
]
EXAMPLES = [('1', '2025-03-01', 'train', 1), ('1', '2025-02-28', 'train', 0), ('2', '2025-02-28', 'train', 0), ('2', '2025-03-01', 'train', 1),
            ('2', '2025-03-02', 'train', None), ('3', '2025-03-01', 'train', 0)]


class ClassLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        observations, examples = root / '2025.parquet', root / 'examples.parquet'
        paths = {name: root / f'{name}.parquet' for name in cl.CLASSES}
        labels = root / 'labels.parquet'
        with duckdb.connect() as c:
            c.execute('CREATE TABLE o(channel_id VARCHAR, occurred_at TIMESTAMP, is_alarm BOOLEAN, raw_value VARCHAR)')
            c.executemany('INSERT INTO o VALUES (?, ?::TIMESTAMP, ?, ?)', OBSERVATIONS)
            c.execute(f"COPY o TO '{observations}' (FORMAT PARQUET)")
            c.execute('CREATE TABLE e(channel_id VARCHAR, as_of TIMESTAMP, split VARCHAR, label INTEGER)')
            c.executemany('INSERT INTO e VALUES (?, ?::TIMESTAMP, ?, ?)', EXAMPLES)
            c.execute(f"COPY e TO '{examples}' (FORMAT PARQUET)")
            for name, states in cl.CLASSES.items():
                cl.build_episodes(c, [str(observations)], states, 900, paths[name])
            cl.build_labels(c, examples, paths['failure'], paths['incident'], labels)
            cls.episodes = {name: c.execute(f"SELECT channel_id, first_alarm_at::VARCHAR, alarm_row_count, left_boundary FROM read_parquet('{path}') ORDER BY 1, 2").fetchall()
                            for name, path in paths.items()}
            cls.labels = {(r[0], r[1][:10]): r[2:] for r in c.execute(f"SELECT channel_id, as_of::VARCHAR, label, label_failure, label_incident FROM read_parquet('{labels}')").fetchall()}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_failure_episodes_ignore_incident_alarms_between(self):
        self.assertEqual(self.episodes['failure'], [('1', '2025-03-01 10:00:00', 2, 'observed_quiet_gap'), ('2', '2025-03-02 05:00:00', 1, 'observed_quiet_gap')])

    def test_incident_episodes_split_by_gap_and_unknown_boundary(self):
        self.assertEqual(self.episodes['incident'], [('1', '2025-03-01 10:05:00', 1, 'observed_quiet_gap'), ('1', '2025-03-01 12:00:00', 1, 'observed_quiet_gap'),
                                                     ('2', '2025-03-01 00:10:00', 1, 'unknown')])

    def test_non_alarm_rows_with_failure_text_are_ignored(self):
        self.assertFalse(any(row[0] == '3' for rows in self.episodes.values() for row in rows))

    def test_labels_per_class(self):
        self.assertEqual(self.labels[('1', '2025-03-01')], (1, 1, 1))
        self.assertEqual(self.labels[('1', '2025-02-28')], (0, 0, 0))
        self.assertEqual(self.labels[('2', '2025-02-28')], (0, 0, 0))
        self.assertEqual(self.labels[('2', '2025-03-01')], (1, 0, None))
        self.assertEqual(self.labels[('2', '2025-03-02')], (None, None, None))
        self.assertEqual(self.labels[('3', '2025-03-01')], (0, 0, 0))
        self.assertEqual(len(self.labels), len(EXAMPLES))


if __name__ == '__main__':
    unittest.main()
