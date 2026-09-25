CREATE OR REPLACE VIEW observations AS
SELECT source_row, trim(event_id) AS event_id, trim(channel_id) AS channel_id,
       date, time, alarm, value, lower(trim(alarm)) AS alarm_token,
       try_strptime(date || ' ' || time, '%Y-%m-%d %H:%M:%S') AS occurred_local
FROM events;
SELECT count(*) AS rows,
       count(DISTINCT event_id) AS distinct_event_ids,
       count(*) - count(DISTINCT event_id) AS rows_beyond_distinct_event_ids,
       count(DISTINCT channel_id) AS channels,
       min(occurred_local) AS first_local, max(occurred_local) AS last_local,
       count(*) FILTER (WHERE alarm_token IN ('true','t')) AS alarm_rows,
       count(*) FILTER (WHERE alarm_token IN ('false','f')) AS normal_rows,
       count(*) FILTER (WHERE alarm_token IS NULL OR alarm_token NOT IN ('true','t','false','f')) AS invalid_alarm_rows,
       count(*) FILTER (WHERE occurred_local IS NULL OR strftime(occurred_local, '%Y-%m-%d %H:%M:%S') != date || ' ' || time) AS invalid_timestamp_rows,
       count(*) FILTER (WHERE event_id IS NULL OR event_id = '' OR channel_id IS NULL OR channel_id = '') AS missing_id_rows,
       count(*) FILTER (WHERE value IS NULL OR value = '') AS empty_value_rows,
       count(*) FILTER (WHERE try_cast(value AS DOUBLE) IS NOT NULL AND isfinite(try_cast(value AS DOUBLE))) AS finite_numeric_value_rows,
       count(*) FILTER (WHERE channel_id NOT IN (SELECT trim("ид_канала_данных") FROM catalogue)) AS unmapped_rows,
       count(DISTINCT channel_id) FILTER (WHERE channel_id NOT IN (SELECT trim("ид_канала_данных") FROM catalogue)) AS unmapped_channels
FROM observations;
