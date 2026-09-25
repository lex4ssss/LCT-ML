import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ml-stage1"))
from event_reader import iter_rows, normalize_event
from episodes import iter_episodes


def build(events_path, output_path, source_timezone, gap_seconds):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    try:
        ZoneInfo(source_timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise ValueError("timezone") from None
    list(iter_episodes([], gap_seconds))
    columns = json.loads((Path(__file__).resolve().parents[1] / "ml-stage1" / "columns.json").read_text())["events"]
    rows_count = alarms_count = episodes_count = 0
    with tempfile.TemporaryDirectory(dir=output_path.parent) as directory:
        with closing(sqlite3.connect(str(Path(directory) / "events.sqlite"))) as connection:
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("CREATE TABLE events(channel TEXT, timestamp TEXT, event_id TEXT, alarm INTEGER)")
            for row in iter_rows(events_path, columns):
                event = normalize_event(row, source_timezone)
                connection.execute("INSERT INTO events VALUES (?, ?, ?, ?)",
                                   (event["channel_id"], event["occurred_at"].isoformat(), event["event_id"], event["is_alarm"]))
                rows_count += 1
                alarms_count += event["is_alarm"]
            connection.commit()
            cursor = connection.execute("SELECT channel, timestamp, event_id, alarm FROM events ORDER BY channel, timestamp, rowid")
            events = (dict(channel_id=c, occurred_at=datetime.fromisoformat(t), event_id=i, is_alarm=bool(a))
                      for c, t, i, a in cursor)
            temporary_output = Path(directory) / "episodes.jsonl"
            with temporary_output.open("w", encoding="utf-8") as destination:
                for episode in iter_episodes(events, gap_seconds):
                    episode["first_alarm_at"] = episode["first_alarm_at"].isoformat()
                    episode["last_alarm_at"] = episode["last_alarm_at"].isoformat()
                    destination.write(json.dumps(episode, ensure_ascii=False) + "\n")
                    episodes_count += 1
            os.link(temporary_output, output_path)
    return dict(schema_version=1, algorithm="consecutive_alarm_gap_v1", gap_seconds=gap_seconds,
                source_timezone=source_timezone, input_rows=rows_count, alarm_rows=alarms_count,
                episodes=episodes_count, disk_sort=True, deduplication=False, recovery_inference=False,
                coverage_verified=False, boundary_scope="input_rows_only", training=False, inference=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-timezone", required=True)
    parser.add_argument("--gap-seconds", required=True, type=int)
    args = parser.parse_args()
    try:
        result = build(args.events, args.output, args.source_timezone, args.gap_seconds)
    except (ValueError, OSError, sqlite3.Error) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
