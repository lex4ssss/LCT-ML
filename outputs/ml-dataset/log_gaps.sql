WITH hours AS (
    SELECT date_trunc('hour',occurred_at) AS hour,
           min(occurred_at) AS first_at,max(occurred_at) AS last_at
    FROM observations GROUP BY hour
), previous AS (
    SELECT first_at,lag(last_at) OVER(ORDER BY hour) AS previous_last
    FROM hours
)
SELECT previous_last+INTERVAL '1 microsecond' AS start_at,first_at AS end_at,
       date_diff('microseconds',previous_last,first_at)/1000000.0 AS seconds_between_records
FROM previous
WHERE first_at-previous_last>=getvariable('gap_threshold_seconds')*INTERVAL '1 second'
ORDER BY start_at
