WITH daily AS (
    SELECT channel_id, date_trunc('day',occurred_at-INTERVAL '1 microsecond')+INTERVAL '1 day' AS as_of,
           count(*) AS event_count_24h, count(*) FILTER(WHERE is_alarm) AS alarm_count_24h,
           max(occurred_at) AS last_event, max(occurred_at) FILTER(WHERE is_alarm) AS last_alarm
    FROM observations GROUP BY channel_id,as_of
), channel_first AS (
    SELECT channel_id,min(occurred_at) AS first_observation FROM observations GROUP BY channel_id
), bounds AS (
    SELECT min(occurred_at) AS global_first,max(occurred_at) AS global_last FROM observations
), future AS (
    SELECT channel_id,date_trunc('day',first_alarm_at-INTERVAL '1 microsecond') AS as_of,
           count(*) FILTER(WHERE left_boundary='observed_quiet_gap') AS known_future,
           count(*) FILTER(WHERE left_boundary='unknown') AS unknown_future
    FROM episodes GROUP BY channel_id,as_of
), joined AS (
    SELECT daily.*,first_observation,global_first,global_last,
           coalesce(known_future,0) AS known_future,coalesce(unknown_future,0) AS unknown_future,
           CASE WHEN as_of<getvariable('ml_validation_start') THEN 'train'
                WHEN as_of<getvariable('ml_test_start') THEN 'validation' ELSE 'test' END AS split
    FROM daily JOIN channel_first USING(channel_id) CROSS JOIN bounds LEFT JOIN future USING(channel_id,as_of)
), diagnosed AS (
    SELECT *,CASE
        WHEN as_of-INTERVAL '24 hours'<global_first THEN 'global_history'
        WHEN as_of-INTERVAL '24 hours'<first_observation THEN 'channel_history'
        WHEN as_of+INTERVAL '24 hours'>global_last THEN 'future_window'
        WHEN (split='train' AND as_of+INTERVAL '24 hours'>=getvariable('ml_validation_start'))
          OR (split='validation' AND as_of+INTERVAL '24 hours'>=getvariable('ml_test_start')) THEN 'split_boundary'
        WHEN EXISTS(SELECT 1 FROM gaps g WHERE g.start_at<=as_of+INTERVAL '24 hours' AND g.end_at>as_of-INTERVAL '24 hours') THEN 'known_log_gap'
        WHEN last_alarm>=as_of-getvariable('ml_gap_seconds')*INTERVAL '1 second' THEN 'active_alarm'
        WHEN unknown_future>0 THEN 'uncertain_first_episode'
        ELSE NULL END AS exclusion_reason
    FROM joined
)
SELECT channel_id,as_of,event_count_24h,alarm_count_24h,
       date_diff('microseconds',last_event,as_of)/3600000000.0 AS last_event_age_hours,
       date_diff('microseconds',last_alarm,as_of)/3600000000.0 AS last_alarm_age_hours_24h,
       split,CASE WHEN exclusion_reason IS NOT NULL THEN NULL WHEN known_future>0 THEN 1 ELSE 0 END AS label,
       exclusion_reason,'recorded_alarm_episode' AS label_scope
FROM diagnosed
