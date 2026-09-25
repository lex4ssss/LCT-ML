import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.script = Path(__file__).with_name('profile_data.py')
        self.columns = json.loads(self.script.with_name('columns.json').read_text(encoding='utf-8'))
        self.paths = {kind: self.root / (kind + '.csv') for kind in self.columns}
        self.write('facilities', [['F']])
        self.write('channels', [['001', 'F']])
        self.write('events', [['e1', '001', '2026-08-01', '00:00:00', 't', '-100']])

    def write(self, kind, rows):
        with self.paths[kind].open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns[kind].values())
            writer.writerows(rows)

    def invoke(self, *extra):
        command = [sys.executable, str(self.script)]
        for kind, path in self.paths.items():
            command.extend(['--' + kind, str(path)])
        return subprocess.run(command + list(extra), text=True, capture_output=True, timeout=10,
                              env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))

    def test_timezone_required(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, '')
        self.assertIn('--source-timezone', result.stderr)

    def test_stdout_report(self):
        result = self.invoke('--source-timezone', 'Europe/Moscow')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['first_timestamp_utc'], '2026-07-31T21:00:00+00:00')
        self.assertEqual(report['run']['training'], False)
        self.assertEqual(report['run']['inference'], False)
        self.assertEqual(report['run']['remove_duplicates'], False)
        self.assertEqual(result.stderr, '')

    def test_file_report_and_prevent_overwrite(self):
        output = self.root / 'report.json'
        result = self.invoke('--source-timezone', 'UTC', '--output', str(output))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '')
        saved = output.read_bytes()
        result = self.invoke('--source-timezone', 'UTC', '--output', str(output))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(saved, output.read_bytes())
        for path in self.paths.values():
            before = path.read_bytes()
            result = self.invoke('--source-timezone', 'UTC', '--output', str(path))
            self.assertEqual(result.returncode, 1)
            self.assertEqual(before, path.read_bytes())

    def test_invalid_row_and_unmatched_channel_return_attention(self):
        for row, key in [(['e1', '001', 'bad', '00:00:00', 't', ''], 'rejected_rows'),
                         (['e1', 'unknown', '2026-08-01', '00:00:00', 't', ''], 'unmapped_rows')]:
            with self.subTest(key=key):
                self.write('events', [row])
                result = self.invoke('--source-timezone', 'UTC')
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stdout)[key], 1)

    def test_fatal_error_creates_no_report(self):
        for kind in ['missing', 'structure', 'timezone']:
            with self.subTest(kind=kind):
                self.write('events', [['e1', '001', '2026-08-01', '00:00:00', 't', '']])
                zone = 'UTC'
                if kind == 'missing':
                    self.paths['events'].unlink()
                elif kind == 'structure':
                    self.write('events', [['too', 'short']])
                else:
                    zone = 'Invalid/Zone'
                output = self.root / (kind + '.json')
                result = self.invoke('--source-timezone', zone, '--output', str(output))
                self.assertEqual(result.returncode, 1)
                self.assertFalse(output.exists())
                self.assertEqual(result.stdout, '')
                self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
