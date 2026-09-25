WITH daily AS (
    SELECT channel_id, date_trunc('day',occurred_at-INTERVAL '1 microsecond')+INTERVAL '1 day' AS as_of,
           count(*) AS n, count(*) FILTER(WHERE is_alarm) AS a,
           max(occurred_at) AS last_event, max(occurred_at) FILTER(WHERE is_alarm) AS last_alarm
    FROM observations GROUP BY channel_id,as_of
), island_marks AS (
    SELECT channel_id,as_of,CASE WHEN lag(as_of) OVER w IS NULL OR as_of-lag(as_of) OVER w>INTERVAL '30 days' THEN 1 ELSE 0 END AS new_island
    FROM daily WINDOW w AS (PARTITION BY channel_id ORDER BY as_of)
), islands AS (
    SELECT channel_id,min(as_of) AS first_day,max(as_of)+INTERVAL '29 days' AS last_day
    FROM (SELECT *,sum(new_island) OVER (PARTITION BY channel_id ORDER BY as_of) AS island FROM island_marks)
    GROUP BY channel_id,island
), grid AS (
    SELECT channel_id,unnest(generate_series(first_day,last_day,INTERVAL '1 day')) AS as_of FROM islands
), episode_daily AS (
    SELECT channel_id,date_trunc('day',first_alarm_at-INTERVAL '1 microsecond')+INTERVAL '1 day' AS as_of,count(*) AS s
    FROM episodes GROUP BY channel_id,as_of
), dense AS (
    SELECT grid.channel_id,grid.as_of,coalesce(n,0) AS n,coalesce(a,0) AS a,coalesce(s,0) AS s,last_event,last_alarm
    FROM grid LEFT JOIN daily USING(channel_id,as_of) LEFT JOIN episode_daily USING(channel_id,as_of)
), windowed AS (
    SELECT channel_id,as_of,n AS event_count_24h,a AS alarm_count_24h,
           sum(n) OVER w7 AS event_count_7d,sum(a) OVER w7 AS alarm_count_7d,
           sum(n) OVER w30 AS event_count_30d,sum(a) OVER w30 AS alarm_count_30d,
           count(*) FILTER(WHERE n>0) OVER w30 AS active_days_30d,
           sum(s) OVER w7 AS episode_starts_7d,sum(s) OVER w30 AS episode_starts_30d,
           max(last_event) OVER w30 AS last_event,max(last_alarm) OVER w30 AS last_alarm
    FROM dense WINDOW w7 AS (PARTITION BY channel_id ORDER BY as_of RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW),
                      w30 AS (PARTITION BY channel_id ORDER BY as_of RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW)
), channel_first AS (
    SELECT channel_id,min(occurred_at) AS first_observation FROM observations GROUP BY channel_id
), bounds AS (
    SELECT min(occurred_at) AS global_first,max(occurred_at) AS global_last FROM observations
), gap_hours AS (
    SELECT d.as_of,coalesce(sum(date_diff('microseconds',greatest(g.start_at,d.as_of-INTERVAL '30 days'),least(g.end_at,d.as_of))) FILTER(WHERE g.start_at IS NOT NULL)/3600000000.0,0) AS log_gap_hours_30d
    FROM (SELECT DISTINCT as_of FROM grid) d LEFT JOIN gaps g ON g.start_at<d.as_of AND g.end_at>d.as_of-INTERVAL '30 days'
    GROUP BY d.as_of
), future AS (
    SELECT channel_id,date_trunc('day',first_alarm_at-INTERVAL '1 microsecond') AS as_of,
           count(*) FILTER(WHERE left_boundary='observed_quiet_gap') AS known_future,
           count(*) FILTER(WHERE left_boundary='unknown') AS unknown_future
    FROM episodes GROUP BY channel_id,as_of
), joined AS (
    SELECT windowed.*,first_observation,global_first,global_last,log_gap_hours_30d,
           coalesce(known_future,0) AS known_future,coalesce(unknown_future,0) AS unknown_future,
           CASE WHEN as_of<getvariable('ml_validation_start') THEN 'train'
                WHEN as_of<getvariable('ml_test_start') THEN 'validation' ELSE 'test' END AS split
    FROM windowed JOIN channel_first USING(channel_id) CROSS JOIN bounds JOIN gap_hours USING(as_of) LEFT JOIN future USING(channel_id,as_of)
), diagnosed AS (
    SELECT *,CASE
        WHEN as_of-INTERVAL '30 days'<global_first THEN 'global_history'
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
SELECT channel_id,as_of,event_count_24h,alarm_count_24h,event_count_7d,alarm_count_7d,event_count_30d,alarm_count_30d,
       active_days_30d,episode_starts_7d,episode_starts_30d,
       date_diff('microseconds',last_event,as_of)/3600000000.0 AS last_event_age_hours,
       date_diff('microseconds',last_alarm,as_of)/3600000000.0 AS last_alarm_age_hours_30d,
       date_diff('microseconds',first_observation,as_of)/86400000000.0 AS channel_age_days,
       log_gap_hours_30d,split,
       CASE WHEN exclusion_reason IS NOT NULL THEN NULL WHEN known_future>0 THEN 1 ELSE 0 END AS label,
       exclusion_reason,'recorded_alarm_episode' AS label_scope
FROM diagnosed
