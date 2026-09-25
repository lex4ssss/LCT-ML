import copy
import random
import unittest
from datetime import datetime, timedelta, timezone

from episodes import iter_episodes


BASE = datetime(2026, 8, 1, tzinfo=timezone.utc)


def event(second, alarm=True, channel="01", event_id="1"):
    return dict(channel_id=channel, event_id=event_id,
                occurred_at=BASE + timedelta(seconds=second), is_alarm=alarm)


def reference(rows, gap):
    result = []
    for channel in sorted({row["channel_id"] for row in rows}):
        channel_rows = [row for row in rows if row["channel_id"] == channel]
        alarm_times = [row["occurred_at"].astimezone(timezone.utc)
                       for row in channel_rows if row["is_alarm"]]
        cuts = [0] + [i for i in range(1, len(alarm_times))
                      if (alarm_times[i] - alarm_times[i - 1]).total_seconds() > gap]
        for start, end in zip(cuts, cuts[1:] + [len(alarm_times)]):
            group = alarm_times[start:end]
            if not group:
                continue
            left = start > 0 or (group[0] - channel_rows[0]["occurred_at"]).total_seconds() > gap
            right = (channel_rows[-1]["occurred_at"] - group[-1]).total_seconds() > gap
            result.append(dict(channel_id=channel, first_alarm_at=group[0],
                               last_alarm_at=group[-1], alarm_row_count=len(group),
                               left_boundary="observed_quiet_gap" if left else "unknown",
                               right_boundary="observed_quiet_gap" if right else "unknown"))
    return result


class EpisodeTests(unittest.TestCase):
    def test_strict_gap_and_chaining(self):
        rows = [event(t) for t in (0, 60, 120, 181)]
        result = list(iter_episodes(rows, 60))
        self.assertEqual([x["alarm_row_count"] for x in result], [3, 1])
        self.assertEqual(result, reference(rows, 60))

    def test_normal_rows_do_not_mean_recovery(self):
        rows = [event(0), event(10, False), event(60), event(121, False)]
        result = list(iter_episodes(rows, 60))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["alarm_row_count"], 2)
        self.assertEqual(result[0]["right_boundary"], "observed_quiet_gap")

    def test_window_boundaries(self):
        for first_alarm in (59, 60, 61):
            for last_row_offset in (59, 60, 61):
                with self.subTest(first=first_alarm, tail=last_row_offset):
                    rows = [event(0, False), event(first_alarm),
                            event(first_alarm + last_row_offset, False)]
                    self.assertEqual(list(iter_episodes(rows, 60)), reference(rows, 60))

    def test_channels_and_normal_only_channel(self):
        rows = [event(0, channel="01"), event(900, False, "02"), event(0, channel="03")]
        result = list(iter_episodes(rows, 60))
        self.assertEqual([x["channel_id"] for x in result], ["01", "03"])
        self.assertTrue(all(x["right_boundary"] == "unknown" for x in result))

    def test_duplicates_count_and_input_immutable(self):
        rows = [event(0), event(0), event(0, event_id="2"), event(61)]
        original = copy.deepcopy(rows)
        result = list(iter_episodes(rows, 60))
        self.assertEqual([x["alarm_row_count"] for x in result], [3, 1])
        result[0]["alarm_row_count"] = 100
        self.assertEqual(result[1]["alarm_row_count"], 1)
        self.assertEqual(rows, original)

    def test_utc_comparison(self):
        first = event(0)
        second = event(60)
        second["occurred_at"] = second["occurred_at"].astimezone(timezone(timedelta(hours=3)))
        result = list(iter_episodes([first, second], 60))
        self.assertEqual(result[0]["alarm_row_count"], 2)
        self.assertEqual(result[0]["last_alarm_at"].tzinfo, timezone.utc)

    def test_gap_rejected_before_consumption(self):
        def source():
            raise AssertionError("Input must not be consumed")
            yield
        for gap in (0, -1, True, False, 1.0, "60", None, float("nan"), float("inf")):
            with self.subTest(gap=gap), self.assertRaisesRegex(ValueError, "^gap$"):
                list(iter_episodes(source(), gap))

    def test_invalid_events(self):
        cases = [None, {}, [], "event"]
        for key, value in (("channel_id", ""), ("channel_id", 1),
                           ("event_id", ""), ("event_id", None),
                           ("occurred_at", BASE.replace(tzinfo=None)),
                           ("occurred_at", "2026-08-01"),
                           ("occurred_at", datetime.min.replace(tzinfo=timezone(timedelta(hours=1)))),
                           ("is_alarm", 1), ("is_alarm", "false")):
            row = event(0, False)
            row[key] = value
            cases.append(row)
        for row in cases:
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, "^event$"):
                list(iter_episodes([row], 60))

    def test_unsorted_alarm_and_normal_rows(self):
        for rows in ([event(1), event(0)], [event(1, False), event(0, False)],
                     [event(1, channel="02"), event(2, channel="01")],
                     [event(1), event(10, channel="02"), event(20)]):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, "^order$"):
                list(iter_episodes(rows, 60))

    def test_empty_and_all_normal(self):
        self.assertEqual(list(iter_episodes([], 60)), [])
        self.assertEqual(list(iter_episodes([event(t, False) for t in range(10)], 60)), [])

    def test_extreme_timestamps(self):
        rows = [event(0), event(0)]
        rows[0]["occurred_at"] = datetime.min.replace(tzinfo=timezone.utc)
        rows[1]["occurred_at"] = datetime.max.replace(tzinfo=timezone.utc)
        self.assertEqual(len(list(iter_episodes(rows, 60))), 2)
        self.assertEqual(len(list(iter_episodes(rows, 10 ** 30))), 1)

    def test_iterator_error_propagates(self):
        def source():
            yield event(0)
            raise RuntimeError("source failed")
        with self.assertRaisesRegex(RuntimeError, "source failed"):
            list(iter_episodes(source(), 60))

    def test_large_gap_keeps_microsecond_precision(self):
        rows = [event(0), event(0)]
        rows[0]["occurred_at"] = datetime(1600, 1, 1, tzinfo=timezone.utc)
        rows[1]["occurred_at"] = datetime(2600, 1, 1, tzinfo=timezone.utc)
        gap = (rows[1]["occurred_at"] - rows[0]["occurred_at"]).days * 86400
        self.assertEqual(len(list(iter_episodes(rows, gap))), 1)
        rows[1]["occurred_at"] += timedelta(microseconds=1)
        self.assertEqual(len(list(iter_episodes(rows, gap))), 2)

    def test_bounded_lookahead(self):
        def source():
            yield event(0)
            yield event(61)
            raise RuntimeError("later")
        iterator = iter_episodes(source(), 60)
        self.assertEqual(next(iterator)["alarm_row_count"], 1)
        with self.assertRaisesRegex(RuntimeError, "later"):
            list(iterator)

    def test_seeded_reference_equivalence(self):
        rng = random.Random(342)
        for trial in range(200):
            rows = []
            for channel in ("01", "02", "10"):
                second = 0
                for i in range(rng.randrange(1, 100)):
                    second += rng.randrange(0, 100)
                    rows.append(event(second, rng.choice([True, False]), channel, str(i)))
            gap = rng.choice([1, 59, 60, 61, 300])
            with self.subTest(trial=trial):
                self.assertEqual(list(iter_episodes(iter(rows), gap)), reference(rows, gap))


if __name__ == "__main__":
    unittest.main()
