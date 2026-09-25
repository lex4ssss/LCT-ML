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
    def __init__(self, prepared_dir, config_path):
        config = json.loads(Path(config_path).read_text())
        self.gaps = [(datetime.fromisoformat(g['start_at']), datetime.fromisoformat(g['end_at'])) for g in config['gap_intervals_utc']]
        files = sorted(str(p) for p in Path(prepared_dir).glob('20??.parquet'))
        if not files:
            raise FileNotFoundError(str(prepared_dir))
        self.connection = duckdb.connect(config={'threads': 4, 'memory_limit': '2GB'})
        self.connection.execute('CREATE VIEW observations AS SELECT channel_id, occurred_at, is_alarm FROM read_parquet(' + repr(files) + ')')
        self.connection.execute('CREATE TABLE channels AS SELECT channel_id, min(occurred_at) AS first_at FROM observations GROUP BY channel_id')
        self.global_first, self.global_last = self.connection.execute('SELECT min(first_at), (SELECT max(occurred_at) FROM observations) FROM channels').fetchone()

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
