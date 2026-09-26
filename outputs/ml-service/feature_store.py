from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import duckdb

DAY = timedelta(days=1)
EPISODE_GAP = timedelta(seconds=900)
FEATURE_SQL = """
WITH w AS (
    SELECT occurred_at, is_alarm FROM observations
    WHERE channel_id = $channel AND occurred_at > $t - INTERVAL '30 days' - INTERVAL '900 seconds' AND occurred_at <= $t
), alarms AS (
    SELECT occurred_at, lag(occurred_at) OVER (ORDER BY occurred_at) AS previous_alarm FROM w WHERE is_alarm
), recent AS (
    SELECT * FROM w WHERE occurred_at > $t - INTERVAL '30 days'
)
SELECT
    count(*) FILTER (WHERE occurred_at > $t - INTERVAL '1 day'),
    count(*) FILTER (WHERE is_alarm AND occurred_at > $t - INTERVAL '1 day'),
    count(*) FILTER (WHERE occurred_at > $t - INTERVAL '7 days'),
    count(*) FILTER (WHERE is_alarm AND occurred_at > $t - INTERVAL '7 days'),
    count(*),
    count(*) FILTER (WHERE is_alarm),
    count(DISTINCT date_diff('microseconds', occurred_at, $t) // 86400000000),
    (SELECT count(*) FROM alarms WHERE occurred_at > $t - INTERVAL '7 days' AND (previous_alarm IS NULL OR occurred_at - previous_alarm > INTERVAL '900 seconds')),
    (SELECT count(*) FROM alarms WHERE occurred_at > $t - INTERVAL '30 days' AND (previous_alarm IS NULL OR occurred_at - previous_alarm > INTERVAL '900 seconds')),
    date_diff('microseconds', max(occurred_at), $t) / 3600000000.0,
    date_diff('microseconds', max(occurred_at) FILTER (WHERE is_alarm), $t) / 3600000000.0
FROM recent
"""
MANY_SQL = """
WITH w AS (
    SELECT channel_id, occurred_at, is_alarm FROM observations
    WHERE channel_id IN (SELECT unnest($channels)) AND occurred_at > $t - INTERVAL '30 days' - INTERVAL '900 seconds' AND occurred_at <= $t
), alarms AS (
    SELECT channel_id, occurred_at, lag(occurred_at) OVER (PARTITION BY channel_id ORDER BY occurred_at) AS previous_alarm FROM w WHERE is_alarm
), starts AS (
    SELECT channel_id, count(*) FILTER (WHERE occurred_at > $t - INTERVAL '7 days') AS s7, count(*) AS s30 FROM alarms
    WHERE occurred_at > $t - INTERVAL '30 days' AND (previous_alarm IS NULL OR occurred_at - previous_alarm > INTERVAL '900 seconds') GROUP BY channel_id
), recent AS (
    SELECT * FROM w WHERE occurred_at > $t - INTERVAL '30 days'
), counts AS (
    SELECT channel_id,
        count(*) FILTER (WHERE occurred_at > $t - INTERVAL '1 day') AS c1, count(*) FILTER (WHERE is_alarm AND occurred_at > $t - INTERVAL '1 day') AS c2,
        count(*) FILTER (WHERE occurred_at > $t - INTERVAL '7 days') AS c3, count(*) FILTER (WHERE is_alarm AND occurred_at > $t - INTERVAL '7 days') AS c4,
        count(*) AS c5, count(*) FILTER (WHERE is_alarm) AS c6, count(DISTINCT date_diff('microseconds', occurred_at, $t) // 86400000000) AS c7,
        date_diff('microseconds', max(occurred_at), $t) / 3600000000.0 AS c10,
        date_diff('microseconds', max(occurred_at) FILTER (WHERE is_alarm), $t) / 3600000000.0 AS c11
    FROM recent GROUP BY channel_id
)
SELECT c.channel_id, c1, c2, c3, c4, c5, c6, c7, coalesce(s7, 0), coalesce(s30, 0), c10, c11 FROM counts c LEFT JOIN starts USING (channel_id)
"""
CLASS_STARTS_SQL = """
SELECT channel_id, occurred_at FROM (
    SELECT channel_id, occurred_at, lag(occurred_at) OVER (PARTITION BY channel_id ORDER BY occurred_at) AS previous_alarm FROM observations
    WHERE channel_id IN (SELECT unnest($channels)) AND is_alarm AND raw_value IN (SELECT unnest($states))
      AND occurred_at > $t - INTERVAL '30 days' - INTERVAL '900 seconds' AND occurred_at <= $t)
WHERE occurred_at > $t - INTERVAL '30 days' AND (previous_alarm IS NULL OR occurred_at - previous_alarm > INTERVAL '900 seconds')
"""
COUNT_NAMES = ['event_count_24h', 'alarm_count_24h', 'event_count_7d', 'alarm_count_7d', 'event_count_30d', 'alarm_count_30d', 'active_days_30d',
               'episode_starts_7d', 'episode_starts_30d', 'last_event_age_hours', 'last_alarm_age_hours_30d']


class InsufficientData(Exception):
    def __init__(self, reason, detail):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


def to_utc_naive(moment):
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment


class FeatureStore:
    def __init__(self, prepared_dir, config_path, class_states=()):
        config = json.loads(Path(config_path).read_text())
        self.gaps = [(datetime.fromisoformat(g['start_at']), datetime.fromisoformat(g['end_at'])) for g in config['gap_intervals_utc']]
        files = sorted(str(p) for p in Path(prepared_dir).glob('20??.parquet'))
        if not files:
            raise FileNotFoundError(str(prepared_dir))
        self.connection = duckdb.connect(config={'threads': 4, 'memory_limit': '2GB'})
        self.connection.execute('CREATE VIEW observations AS SELECT channel_id, occurred_at, is_alarm, raw_value FROM read_parquet(' + repr(files) + ')')
        self.connection.execute('CREATE TABLE channels AS SELECT channel_id, min(occurred_at) AS first_at FROM observations GROUP BY channel_id')
        self.global_first, self.global_last = self.connection.execute('SELECT min(first_at), (SELECT max(occurred_at) FROM observations) FROM channels').fetchone()
        self.class_states = list(class_states)
        self.connection.execute('CREATE TABLE class_first AS SELECT channel_id, min(occurred_at) AS first_at FROM observations '
                                'WHERE is_alarm AND raw_value IN (SELECT unnest(?)) GROUP BY channel_id', [self.class_states])

    def check_journal(self, t):
        if t - 30 * DAY < self.global_first:
            raise InsufficientData('global_history', 'journal starts at ' + self.global_first.isoformat())
        if t > self.global_last + timedelta(hours=1):
            raise InsufficientData('stale_journal', 'last journal record at ' + self.global_last.isoformat())
        if any(a <= t and b > t - DAY for a, b in self.gaps):
            raise InsufficientData('known_log_gap', 'journal-wide gap in the previous 24 hours')

    def features_many(self, channels, moment):
        t = to_utc_naive(moment)
        self.check_journal(t)
        first = dict(self.connection.execute('SELECT channel_id, first_at FROM channels WHERE channel_id IN (SELECT unnest(?))', [list(channels)]).fetchall())
        gap, result = self.gap_hours(t), {}
        for channel, *counts in self.connection.execute(MANY_SQL, {'channels': list(channels), 't': t}).fetchall():
            values = dict(zip(COUNT_NAMES, counts))
            alarm_age = values['last_alarm_age_hours_30d']
            if t - DAY < first[channel] or (alarm_age is not None and alarm_age <= EPISODE_GAP / timedelta(hours=1)):
                continue
            values['channel_age_days'] = (t - first[channel]) / DAY
            values['log_gap_hours_30d'] = gap
            result[channel] = values
        return t, result

    def class_history(self, channels, moment):
        t = to_utc_naive(moment)
        starts = self.connection.execute(CLASS_STARTS_SQL, {'channels': list(channels), 'states': self.class_states, 't': t}).fetchall()
        seen = dict(self.connection.execute('SELECT channel_id, first_at <= ? FROM class_first WHERE channel_id IN (SELECT unnest(?))', [t, list(channels)]).fetchall())
        return starts, {channel for channel, before in seen.items() if before}

    def gap_hours(self, t):
        start = t - 30 * DAY
        return sum(max(timedelta(0), min(b, t) - max(a, start)) / timedelta(hours=1) for a, b in self.gaps)

    def features(self, channel, moment):
        t = to_utc_naive(moment)
        if t - 30 * DAY < self.global_first:
            raise InsufficientData('global_history', 'journal starts at ' + self.global_first.isoformat())
        if t > self.global_last + timedelta(hours=1):
            raise InsufficientData('stale_journal', 'last journal record at ' + self.global_last.isoformat())
        first = self.connection.execute('SELECT first_at FROM channels WHERE channel_id = ?', [channel]).fetchone()
        if first is None or first[0] > t:
            raise InsufficientData('unknown_channel', 'no records for channel ' + channel)
        values = dict(zip(COUNT_NAMES, self.connection.execute(FEATURE_SQL, {'channel': channel, 't': t}).fetchone()))
        if values['event_count_30d'] == 0:
            raise InsufficientData('no_recent_records', 'no records in the previous 30 days')
        if t - DAY < first[0]:
            raise InsufficientData('channel_history', 'channel observed for less than 24 hours')
        if any(a <= t and b > t - DAY for a, b in self.gaps):
            raise InsufficientData('known_log_gap', 'journal-wide gap in the previous 24 hours')
        if values['last_alarm_age_hours_30d'] is not None and values['last_alarm_age_hours_30d'] <= EPISODE_GAP / timedelta(hours=1):
            raise InsufficientData('active_alarm', 'alarm episode is ongoing; the model forecasts new episodes only')
        values['channel_age_days'] = (t - first[0]) / DAY
        values['log_gap_hours_30d'] = self.gap_hours(t)
        return t, values
