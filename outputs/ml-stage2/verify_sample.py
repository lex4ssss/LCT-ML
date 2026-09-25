import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile

from build_episodes import build, iter_rows, normalize_event


def verify(source, source_timezone):
    columns = json.loads((Path(__file__).resolve().parents[1] / "ml-stage1" / "columns.json").read_text())["events"]
    channels = defaultdict(lambda: dict(first=None, last=None, alarms=[], normal_rows=0))
    previous = None
    inversions = rows_count = 0
    for row in iter_rows(source, columns):
        rows_count += 1
        if rows_count > 200000:
            raise ValueError("This verification harness is limited to 200000 fixture rows")
        event = normalize_event(row, source_timezone)
        timestamp = event["occurred_at"]
        inversions += previous is not None and timestamp < previous
        previous = timestamp
        channel = channels[event["channel_id"]]
        channel["first"] = min(channel["first"], timestamp) if channel["first"] else timestamp
        channel["last"] = max(channel["last"], timestamp) if channel["last"] else timestamp
        if event["is_alarm"]:
            channel["alarms"].append(timestamp)
        else:
            channel["normal_rows"] += 1
    results = []
    with tempfile.TemporaryDirectory() as directory:
        for gap in (300, 900, 3600):
            expected = []
            for channel_id, channel in sorted(channels.items()):
                alarms = sorted(channel["alarms"])
                cuts = [0] + [i for i in range(1, len(alarms)) if (alarms[i] - alarms[i - 1]).total_seconds() > gap]
                for start, end in zip(cuts, cuts[1:] + [len(alarms)]):
                    group = alarms[start:end]
                    if not group:
                        continue
                    left = start > 0 or (group[0] - channel["first"]).total_seconds() > gap
                    right = (channel["last"] - group[-1]).total_seconds() > gap
                    expected.append(dict(channel_id=channel_id, first_alarm_at=group[0].isoformat(),
                                         last_alarm_at=group[-1].isoformat(), alarm_row_count=len(group),
                                         left_boundary="observed_quiet_gap" if left else "unknown",
                                         right_boundary="observed_quiet_gap" if right else "unknown"))
            output = Path(directory) / str(gap)
            run = build(source, output, source_timezone, gap)
            actual = [json.loads(line) for line in output.read_text().splitlines()]
            if actual != expected or sum(x["alarm_row_count"] for x in actual) != run["alarm_rows"]:
                raise AssertionError("Independent reference mismatch")
            results.append(dict(run=run, independent_reference_equal=True,
                                left_unknown=sum(x["left_boundary"] == "unknown" for x in actual),
                                right_unknown=sum(x["right_boundary"] == "unknown" for x in actual)))
    digest = hashlib.sha256()
    with open(source, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return dict(source_sha256=digest.hexdigest(), source_rows=rows_count, timestamp_order_inversions=inversions,
                same_channel_alarm_timestamp_repeats=sum(len(c["alarms"]) - len(set(c["alarms"])) for c in channels.values()),
                alarm_only_channels=sum(bool(c["alarms"]) and c["normal_rows"] == 0 for c in channels.values()),
                results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--source-timezone", required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.events, args.source_timezone), ensure_ascii=False, indent=2))
