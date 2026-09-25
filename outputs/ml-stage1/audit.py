from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from event_reader import iter_rows, normalize_event


def profile(events_path, channels_path, facilities_path, timezone_name, columns):
    try:
        ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise ValueError("timezone")

    facilities = set()
    channels = {}
    for row in iter_rows(Path(facilities_path), columns["facilities"]):
        facility_id = str(row.get("facility_id", "")).strip()
        if not facility_id:
            raise ValueError("catalogue")
        if facility_id in facilities:
            raise ValueError("catalogue")
        facilities.add(facility_id)

    for row in iter_rows(Path(channels_path), columns["channels"]):
        channel_id = str(row.get("channel_id", "")).strip()
        facility_id = str(row.get("facility_id", "")).strip()
        if not channel_id or not facility_id:
            raise ValueError("catalogue")
        if channel_id in channels:
            raise ValueError("catalogue")
        if facility_id not in facilities:
            raise ValueError("catalogue")
        channels[channel_id] = facility_id

    total_rows = 0
    valid_rows = 0
    rejected_rows = 0
    rejection_reasons = {}
    alarm_rows = 0
    normal_rows = 0
    observed_channels = set()
    unmatched_channel_set = set()
    unmatched_rows = 0
    first_timestamp_utc = None
    last_timestamp_utc = None

    for row in iter_rows(Path(events_path), columns["events"]):
        total_rows += 1
        try:
            event = normalize_event(row, timezone_name)
        except ValueError as exc:
            code = str(exc)
            if code in ("missing_id", "timestamp", "alarm"):
                rejected_rows += 1
                rejection_reasons[code] = rejection_reasons.get(code, 0) + 1
                continue
            else:
                raise

        valid_rows += 1
        channel_id = event["channel_id"]
        if channel_id:
            observed_channels.add(channel_id)
            if channel_id not in channels:
                unmatched_channel_set.add(channel_id)
                unmatched_rows += 1

        alarm = event["is_alarm"]
        if alarm:
            alarm_rows += 1
        else:
            normal_rows += 1

        ts = event["occurred_at"]
        if ts is not None:
            if first_timestamp_utc is None or ts < first_timestamp_utc:
                first_timestamp_utc = ts
            if last_timestamp_utc is None or ts > last_timestamp_utc:
                last_timestamp_utc = ts

    span_seconds = None
    if first_timestamp_utc is not None and last_timestamp_utc is not None:
        span_seconds = float((last_timestamp_utc - first_timestamp_utc).total_seconds())

    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "rejected_rows": rejected_rows,
        "rejection_reasons": rejection_reasons,
        "alarm_rows": alarm_rows,
        "normal_rows": normal_rows,
        "observed_channels": len(observed_channels),
        "unmapped_rows": unmatched_rows,
        "unmapped_channels": len(unmatched_channel_set),
        "catalogue_channels": len(channels),
        "catalogue_facilities": len(facilities),
        "first_timestamp_utc": first_timestamp_utc.isoformat() if first_timestamp_utc else None,
        "last_timestamp_utc": last_timestamp_utc.isoformat() if last_timestamp_utc else None,
        "span_seconds": span_seconds,
        "source_timezone": timezone_name,
        "duplicate_event_ids": "not_checked",
        "observation_coverage": "not_verified",
    }
