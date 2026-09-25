import csv
import tempfile
import unittest
from pathlib import Path

from audit import profile


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.events = self.root / 'events.csv'
        self.channels = self.root / 'channels.csv'
        self.facilities = self.root / 'facilities.csv'
        self.columns = {name: {key: key for key in keys} for name, keys in {
            'events': ['event_id', 'channel_id', 'date', 'time', 'alarm', 'value'],
            'channels': ['channel_id', 'facility_id'], 'facilities': ['facility_id']}.items()}
        self.write(self.facilities, 'facilities', [['F']])
        self.write(self.channels, 'channels', [['001', 'F']])
        self.event = ['e1', '001', '2026-08-01', '00:00:00', 'true', '255']

    def write(self, path, kind, rows):
        with path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns[kind])
            writer.writerows(rows)

    def run_profile(self, rows, zone='Europe/Moscow'):
        self.write(self.events, 'events', rows)
        before = [p.read_bytes() for p in [self.events, self.channels, self.facilities]]
        result = profile(self.events, self.channels, self.facilities, zone, self.columns)
        self.assertEqual(before, [p.read_bytes() for p in [self.events, self.channels, self.facilities]])
        return result

    def test_empty(self):
        result = self.run_profile([])
        for key in ['total_rows', 'valid_rows', 'rejected_rows', 'alarm_rows', 'normal_rows', 'observed_channels', 'unmapped_rows', 'unmapped_channels']:
            self.assertEqual(result[key], 0)
        for key in ['first_timestamp_utc', 'last_timestamp_utc', 'span_seconds']:
            self.assertIsNone(result[key])

    def test_one_row(self):
        result = self.run_profile([self.event])
        expected = dict(total_rows=1, valid_rows=1, rejected_rows=0, rejection_reasons={}, alarm_rows=1,
                        normal_rows=0, observed_channels=1, unmapped_rows=0, unmapped_channels=0,
                        catalogue_channels=1, catalogue_facilities=1,
                        first_timestamp_utc='2026-07-31T21:00:00+00:00', last_timestamp_utc='2026-07-31T21:00:00+00:00',
                        span_seconds=0.0, source_timezone='Europe/Moscow', duplicate_event_ids='not_checked', observation_coverage='not_verified')
        self.assertEqual(result, expected)

    def test_mixed_rows_unknown_channel_repeated_ids_and_unsorted_time(self):
        rows = [self.event, ['e1', 'unknown', '2026-07-31', '23:00:00', 'false', 'text'],
                ['e1', 'unknown', '2026-08-02', '00:00:00', 't', '']]
        result = self.run_profile(rows)
        for key, expected in dict(total_rows=3, valid_rows=3, alarm_rows=2, normal_rows=1,
                                  observed_channels=2, unmapped_rows=2, unmapped_channels=1, span_seconds=90000.0).items():
            self.assertEqual(result[key], expected)
        self.assertEqual(result['duplicate_event_ids'], 'not_checked')

    def test_bad_records_counted_without_false_normal_labels(self):
        rows = [self.event, ['', '001', '2026-08-01', '00:00:00', 'true', ''],
                ['e3', '001', 'invalid', '00:00:00', 'true', ''],
                ['e4', '001', '2026-08-01', '00:00:00', 'yes', '']]
        result = self.run_profile(rows)
        self.assertEqual(result['total_rows'], 4)
        self.assertEqual(result['valid_rows'], 1)
        self.assertEqual(result['rejected_rows'], 3)
        self.assertEqual(result['normal_rows'], 0)
        self.assertEqual(result['rejection_reasons'], dict(missing_id=1, timestamp=1, alarm=1))

    def test_catalogue_errors(self):
        for kind, rows in [('facilities', [['F'], ['F']]), ('facilities', [['']]),
                           ('channels', [['001', 'F'], ['001', 'F']]), ('channels', [['', 'F']]),
                           ('channels', [['001', '']]), ('channels', [['001', 'missing']])]:
            with self.subTest(kind=kind, rows=rows):
                self.write(self.facilities, 'facilities', [['F']])
                self.write(self.channels, 'channels', [['001', 'F']])
                self.write(getattr(self, kind), kind, rows)
                with self.assertRaisesRegex(ValueError, '^catalogue$'):
                    self.run_profile([self.event])

    def test_invalid_timezone_with_empty_input(self):
        with self.assertRaisesRegex(ValueError, '^timezone$'):
            self.run_profile([], 'Invalid/Zone')

    def test_structural_errors_are_fatal(self):
        self.write(self.events, 'events', [self.event, ['too', 'short']])
        with self.assertRaisesRegex(ValueError, '^csv_structure$'):
            profile(self.events, self.channels, self.facilities, 'UTC', self.columns)


if __name__ == '__main__':
    unittest.main()
