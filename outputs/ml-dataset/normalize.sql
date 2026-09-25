SELECT
    CASE WHEN event_id IS NULL OR trim(event_id) = '' THEN error('missing_event_id') ELSE trim(event_id) END AS event_id,
    CASE WHEN channel_id IS NULL OR trim(channel_id) = '' THEN error('missing_channel_id') ELSE trim(channel_id) END AS channel_id,
    CASE WHEN date IS NULL OR time IS NULL OR
              strftime(strptime(date || ' ' || time, '%Y-%m-%d %H:%M:%S'), '%Y-%m-%d %H:%M:%S') != date || ' ' || time
         THEN error('invalid_timestamp')
         ELSE timezone('UTC', timezone('Europe/Moscow', strptime(date || ' ' || time, '%Y-%m-%d %H:%M:%S'))) END AS occurred_at,
    CASE WHEN lower(trim(alarm)) IN ('true','t') THEN true
         WHEN lower(trim(alarm)) IN ('false','f') THEN false
         ELSE error('invalid_alarm') END AS is_alarm,
    value AS raw_value
FROM records
