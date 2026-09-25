import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from event_reader import iter_rows, normalize_event


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'input.csv'
        self.columns = {'event_id': 'identifier', 'channel_id': 'channel'}
        self.row = dict(event_id='001', channel_id='002', date='2026-08-01', time='00:00:00', alarm='true', value='-100')

    def rows(self, content):
        self.path.write_text(content, encoding='utf-8')
        return iter_rows(self.path, self.columns)

    def test_csv_valid_cases(self):
        cases = [
            ('identifier,channel\n001,002\n', [{'event_id': '001', 'channel_id': '002'}]),
            ('\ufeffidentifier,channel\n1,2\n', [{'event_id': '1', 'channel_id': '2'}]),
            ('channel,extra,identifier\n2,x,1\n', [{'event_id': '1', 'channel_id': '2'}]),
            ('identifier,channel\n"a,b",2\n', [{'event_id': 'a,b', 'channel_id': '2'}]),
            ('identifier,channel\n"a\nb",2\n', [{'event_id': 'a\nb', 'channel_id': '2'}]),
            ('identifier,channel\n"a""b",2\n', [{'event_id': 'a"b', 'channel_id': '2'}]),
            ('identifier,channel\n\n1,2\n\n', [{'event_id': '1', 'channel_id': '2'}]),
            ('identifier,channel\n', []),
            ('identifier,channel\r\n1,2\r\n', [{'event_id': '1', 'channel_id': '2'}]),
            ('identifier,channel\n,\n', [{'event_id': '', 'channel_id': ''}]),
        ]
        for content, expected in cases:
            with self.subTest(content=content):
                self.assertEqual(list(self.rows(content)), expected)
                self.assertEqual(self.path.read_bytes(), content.encode())

    def test_csv_structure_errors(self):
        for content in ['', 'identifier,identifier,channel\n', 'identifier\n',
                        'identifier,channel\n1\n', 'identifier,channel\n1,2,3\n',
                        'identifier,channel\n"unclosed,2\n',
                        'identifier,channel\n"a"x,2\n']:
            with self.subTest(content=content):
                with self.assertRaisesRegex(ValueError, '^csv_structure$'):
                    list(self.rows(content))

    def test_generator_is_lazy(self):
        rows = self.rows('identifier,channel\n1,2\n"unclosed,3\n')
        self.assertEqual(next(rows), {'event_id': '1', 'channel_id': '2'})
        with self.assertRaisesRegex(ValueError, '^csv_structure$'):
            next(rows)

    def test_normalization_preserves_identifiers_and_raw_values(self):
        output = normalize_event(self.row, 'Europe/Moscow')
        self.assertEqual(output, dict(event_id='001', channel_id='002', occurred_at=datetime(2026, 7, 31, 21, tzinfo=timezone.utc), is_alarm=True, raw_value='-100'))
        for raw in ['255', '01.01.1970 03:00:00', 'nan', '0', '', '  text\nvalue  ']:
            with self.subTest(raw=raw):
                self.assertEqual(normalize_event(dict(self.row, value=raw), 'UTC')['raw_value'], raw)

    def test_boolean_variants(self):
        for value, expected in [('true', True), ('t', True), (' TRUE ', True), ('T', True),
                                ('false', False), ('f', False), (' FALSE ', False), ('F', False)]:
            with self.subTest(value=value):
                self.assertIs(normalize_event(dict(self.row, alarm=value), 'UTC')['is_alarm'], expected)
        for value in ['', '1', 'yes', 'null']:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, '^alarm$'):
                    normalize_event(dict(self.row, alarm=value), 'UTC')

    def test_invalid_ids(self):
        for key in ['event_id', 'channel_id']:
            for value in ['', '   ']:
                with self.subTest(key=key, value=value):
                    with self.assertRaisesRegex(ValueError, '^missing_id$'):
                        normalize_event(dict(self.row, **{key: value}), 'UTC')
        self.assertEqual(normalize_event(dict(self.row, event_id=' 001 ', channel_id=' 002 '), 'UTC')['channel_id'], '002')

    def test_timestamp_cases(self):
        valid = [('0001-01-01', '00:00:00', 'UTC'), ('9999-12-31', '23:59:59', 'UTC'), ('2024-02-29', '23:59:59', 'UTC'), ('2026-08-01', '00:00:00', 'Europe/Moscow'),
                 ('2026-01-01', '12:00:00', 'America/New_York'), ('2026-07-01', '12:00:00', 'America/New_York')]
        for date, clock, zone in valid:
            with self.subTest(date=date, clock=clock, zone=zone):
                self.assertEqual(normalize_event(dict(self.row, date=date, time=clock), zone)['occurred_at'].tzinfo, timezone.utc)
        invalid = [('2026-02-29', '12:00:00', 'UTC'), ('2026-08-01', '24:00:00', 'UTC'),
                   ('bad', '12:00:00', 'UTC'), ('2026-08-01', '23:59:60', 'UTC'),
                   ('2026-8-1', '00:00:00', 'UTC'), ('2026-08-01', '0:0:0', 'UTC'),
                   ('0001-01-01', '00:00:00', 'Etc/GMT-3'), ('9999-12-31', '23:59:59', 'Etc/GMT+3'),
                   ('2026-03-08', '02:30:00', 'America/New_York'), ('2026-11-01', '01:30:00', 'America/New_York')]
        for date, clock, zone in invalid:
            with self.subTest(date=date, clock=clock, zone=zone):
                with self.assertRaisesRegex(ValueError, '^timestamp$'):
                    normalize_event(dict(self.row, date=date, time=clock), zone)
        with self.assertRaisesRegex(ValueError, '^timezone$'):
            normalize_event(self.row, 'Invalid/Zone')


if __name__ == '__main__':
    unittest.main()
