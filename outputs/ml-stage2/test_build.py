import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_episodes import build


HEADER = ["ид_события", "ид_канала_данных", "дата", "время", "тревожное", "значение_датчика"]


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "events.csv"
        self.output = self.root / "episodes.jsonl"

    def write_rows(self, rows):
        with self.source.open("w", encoding="utf-8-sig", newline="") as destination:
            writer = csv.writer(destination)
            writer.writerow(HEADER)
            writer.writerows(rows)

    def test_unsorted_input_and_duplicate_ids(self):
        self.write_rows([["1", "02", "2026-08-01", "00:02:00", "true", "x"],
                         ["1", "01", "2026-08-01", "00:01:00", "true", "x"],
                         ["1", "01", "2026-08-01", "00:00:00", "true", "x"],
                         ["2", "01", "2026-08-01", "00:02:01", "false", "x"]])
        report = build(self.source, self.output, "Europe/Moscow", 60)
        rows = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(report["input_rows"], 4)
        self.assertEqual(report["alarm_rows"], 3)
        self.assertEqual(report["episodes"], 2)
        self.assertFalse(report["deduplication"])
        self.assertEqual(rows[0]["channel_id"], "01")
        self.assertEqual(rows[0]["alarm_row_count"], 2)
        self.assertEqual(rows[0]["first_alarm_at"], "2026-07-31T21:00:00+00:00")
        self.assertEqual(rows[0]["right_boundary"], "observed_quiet_gap")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["episodes.jsonl", "events.csv"])

    def test_invalid_late_row_does_not_publish(self):
        for alarm in ("maybe", "", "2"):
            with self.subTest(alarm=alarm):
                self.write_rows([["1", "01", "2026-08-01", "00:00:00", "true", "x"],
                                 ["2", "01", "2026-08-01", "00:01:00", alarm, "x"]])
                with self.assertRaises(ValueError):
                    build(self.source, self.output, "UTC", 60)
                self.assertFalse(self.output.exists())
                self.assertEqual(len(list(self.root.iterdir())), 1)

    def test_existing_output_is_preserved(self):
        self.output.write_text("existing")
        with self.assertRaises(FileExistsError):
            build(self.source, self.output, "UTC", 60)
        self.assertEqual(self.output.read_text(), "existing")

    def test_output_created_during_processing_is_preserved(self):
        import os
        link = os.link
        self.write_rows([])
        def competing_write(source, destination):
            Path(destination).write_text("other process")
            return link(source, destination)
        with patch("build_episodes.os.link", competing_write):
            with self.assertRaises(FileExistsError):
                build(self.source, self.output, "UTC", 60)
        self.assertEqual(self.output.read_text(), "other process")
        self.assertEqual(len(list(self.root.iterdir())), 2)

    def test_invalid_timezone_on_empty_input(self):
        self.write_rows([])
        with self.assertRaises(ValueError):
            build(self.source, self.output, "invalid/timezone", 60)
        self.assertFalse(self.output.exists())

    def test_empty_input(self):
        self.write_rows([])
        report = build(self.source, self.output, "UTC", 60)
        self.assertEqual(report["episodes"], 0)
        self.assertEqual(self.output.read_bytes(), b"")

    def test_cli_success_and_failure(self):
        self.write_rows([["1", "01", "2026-08-01", "00:00:00", "true", "x"]])
        command = [sys.executable, str(Path(__file__).with_name("build_episodes.py")),
                   "--events", str(self.source), "--output", str(self.output),
                   "--source-timezone", "UTC", "--gap-seconds", "60"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["episodes"], 1)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
