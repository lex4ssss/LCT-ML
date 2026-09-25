import csv
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def iter_rows(path, columns):
    required = set(columns.values())
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, strict=True)
            header = next(reader)
            if len(set(header)) != len(header) or required - set(header):
                raise ValueError("csv_structure")
            indexes = {name: header.index(src) for name, src in columns.items()}
            for row in reader:
                if not row:
                    continue
                if len(row) != len(header):
                    raise ValueError("csv_structure")
                yield {name: row[index] for name, index in indexes.items()}
    except (csv.Error, StopIteration):
        raise ValueError("csv_structure") from None


def _parse_timestamp(value, tz):
    try:
        naive = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        raise ValueError("timestamp") from None
    if naive.isoformat(sep=" ") != value:
        raise ValueError("timestamp")
    candidates = set()
    try:
        for fold in (0, 1):
            utc = naive.replace(tzinfo=tz, fold=fold).astimezone(timezone.utc)
            if utc.astimezone(tz).replace(tzinfo=None) == naive:
                candidates.add(utc)
    except OverflowError:
        raise ValueError("timestamp") from None
    if len(candidates) != 1:
        raise ValueError("timestamp")
    return candidates.pop()


def _parse_alarm(value):
    token = value.strip().lower()
    if token in ("true", "t"):
        return True
    if token in ("false", "f"):
        return False
    raise ValueError("alarm")


def normalize_event(row, timezone_name):
    event_id = row["event_id"].strip()
    channel_id = row["channel_id"].strip()
    if not event_id or not channel_id:
        raise ValueError("missing_id")
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise ValueError("timezone") from None
    occurred_at = _parse_timestamp(row["date"] + " " + row["time"], tz)
    is_alarm = _parse_alarm(row["alarm"])
    return {
        "event_id": event_id,
        "channel_id": channel_id,
        "occurred_at": occurred_at,
        "is_alarm": is_alarm,
        "raw_value": row["value"],
    }
