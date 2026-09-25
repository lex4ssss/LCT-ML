from datetime import datetime, timezone

def _beyond(delta, gap):
    seconds = delta.days * 86400 + delta.seconds
    return seconds > gap or (seconds == gap and delta.microseconds > 0)

def iter_episodes(events, gap_seconds):
    if type(gap_seconds) is not int or gap_seconds <= 0:
        raise ValueError("gap")
    channel = first_time = previous = episode = None
    seen_alarm = False
    for row in events:
        if not isinstance(row, dict):
            raise ValueError("event")
        try:
            timestamp = row["occurred_at"]
            if not isinstance(timestamp, datetime) or timestamp.utcoffset() is None:
                raise ValueError("event")
            utc = timestamp.astimezone(timezone.utc)
        except (KeyError, ValueError, OverflowError, TypeError):
            raise ValueError("event") from None
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("channel_id", "event_id")):
            raise ValueError("event")
        if type(row.get("is_alarm")) is not bool:
            raise ValueError("event")
        key = (row["channel_id"], utc)
        if previous is not None and key < previous:
            raise ValueError("order")
        previous = key
        if row["channel_id"] != channel:
            if episode is not None:
                yield episode
            channel, first_time, seen_alarm, episode = row["channel_id"], utc, False, None
        if episode is not None and _beyond(utc - episode["last_alarm_at"], gap_seconds):
            episode["right_boundary"] = "observed_quiet_gap"
            yield episode
            episode = None
        if not row["is_alarm"]:
            continue
        if episode is None:
            episode = dict(channel_id=channel, first_alarm_at=utc, last_alarm_at=utc,
                           alarm_row_count=0, right_boundary="unknown",
                           left_boundary="observed_quiet_gap" if seen_alarm or _beyond(utc - first_time, gap_seconds) else "unknown")
            seen_alarm = True
        episode["last_alarm_at"] = utc
        episode["alarm_row_count"] += 1
    if episode is not None:
        yield episode
