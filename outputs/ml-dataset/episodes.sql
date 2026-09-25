WITH channel_bounds AS (
    SELECT channel_id, min(occurred_at) AS first_observation, max(occurred_at) AS last_observation
    FROM observations GROUP BY channel_id
), alarm_gaps AS (
    SELECT channel_id, occurred_at,
           lag(occurred_at) OVER (PARTITION BY channel_id ORDER BY occurred_at) AS previous_alarm
    FROM observations WHERE is_alarm
), starts AS (
    SELECT *, CASE WHEN previous_alarm IS NULL OR occurred_at-previous_alarm > getvariable('gap_seconds') * INTERVAL '1 second'
                   THEN 1 ELSE 0 END AS starts_episode
    FROM alarm_gaps
), numbered AS (
    SELECT *, sum(starts_episode) OVER (PARTITION BY channel_id ORDER BY occurred_at RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS episode_number
    FROM starts
), grouped AS (
    SELECT channel_id, episode_number, min(occurred_at) AS first_alarm_at,
           max(occurred_at) AS last_alarm_at, count(*) AS alarm_row_count
    FROM numbered GROUP BY channel_id, episode_number
)
SELECT channel_id, first_alarm_at, last_alarm_at, alarm_row_count,
       CASE WHEN episode_number>1 OR first_alarm_at-first_observation > getvariable('gap_seconds') * INTERVAL '1 second'
            THEN 'observed_quiet_gap' ELSE 'unknown' END AS left_boundary,
       CASE WHEN last_observation-last_alarm_at > getvariable('gap_seconds') * INTERVAL '1 second'
            THEN 'observed_quiet_gap' ELSE 'unknown' END AS right_boundary
FROM grouped JOIN channel_bounds USING(channel_id)
ORDER BY channel_id, first_alarm_at
