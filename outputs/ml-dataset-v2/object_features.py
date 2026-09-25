from pathlib import Path
import sys
import tempfile
import duckdb

OBJECT_FEATURES = ['obj_channels_30d', 'obj_other_alarm_channels_24h', 'obj_other_alarm_24h', 'obj_other_alarm_7d',
                   'obj_other_episode_starts_7d', 'obj_other_episode_starts_30d', 'type_other_alarm_7d']
CATEGORIES = ['sensor_type', 'system_type']
SQL = """
WITH d AS (SELECT ид_канала_данных AS channel_id, ид_объект AS object_id, тип_датчика AS sensor_type, тип_инж_системы AS system_type
           FROM read_csv(?, all_varchar=true)),
e AS (SELECT e.channel_id, e.as_of, e.split, e.label, e.alarm_count_24h, e.alarm_count_7d, e.episode_starts_7d, e.episode_starts_30d,
             d.object_id, d.sensor_type, d.system_type
      FROM read_parquet(?) e LEFT JOIN d USING (channel_id)),
o AS (SELECT *,
             count(*) OVER w - 1 AS obj_channels_30d,
             count(*) FILTER (WHERE alarm_count_24h > 0) OVER w - (alarm_count_24h > 0)::INTEGER AS obj_other_alarm_channels_24h,
             sum(alarm_count_24h) OVER w - alarm_count_24h AS obj_other_alarm_24h,
             sum(alarm_count_7d) OVER w - alarm_count_7d AS obj_other_alarm_7d,
             sum(episode_starts_7d) OVER w - episode_starts_7d AS obj_other_episode_starts_7d,
             sum(episode_starts_30d) OVER w - episode_starts_30d AS obj_other_episode_starts_30d,
             sum(alarm_count_7d) OVER t - alarm_count_7d AS type_other_alarm_7d
      FROM e WINDOW w AS (PARTITION BY object_id, as_of), t AS (PARTITION BY object_id, sensor_type, as_of))
SELECT channel_id, as_of, split, label, {features}, sensor_type, system_type FROM o
"""


def build(connection, examples, directory, output):
    features = ', '.join(f'CASE WHEN object_id IS NULL THEN NULL ELSE {name} END::DOUBLE AS {name}' for name in OBJECT_FEATURES)
    query = SQL.format(features=features)
    connection.execute(f"COPY ({query}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)", [str(directory), str(examples)])


if __name__ == '__main__':
    examples, directory, output = map(Path, sys.argv[1:4])
    if output.exists():
        raise FileExistsError(str(output))
    with tempfile.TemporaryDirectory(dir=output.parent) as scratch:
        with duckdb.connect(config=dict(threads=4, memory_limit='4GB', temp_directory=scratch)) as connection:
            build(connection, examples, directory, output)
