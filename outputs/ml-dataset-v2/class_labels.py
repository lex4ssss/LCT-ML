from pathlib import Path
import sys
import tempfile
import duckdb

FAILURE_STATES = ['Неисправен', 'Обесточен', 'Отключено устройство', 'Питание от батарей', 'Батарея разряжена', 'Не определено', 'Много неисправных устройств']
INCIDENT_STATES = ['Не замкнут', 'Обнаружено движение', 'Обнаружен дым', 'Обнаружен газ', 'Затоплен', 'Работают все насосы в АНС', 'Рычаг сдернут',
                   'Температура ниже 3ºC', 'Температура выше 40ºC']
CLASSES = dict(failure=FAILURE_STATES, incident=INCIDENT_STATES)
EPISODES_SQL = Path(__file__).resolve().parent.parent / 'ml-dataset' / 'episodes.sql'
LABELS_SQL = """
WITH f AS (SELECT channel_id, date_trunc('day', first_alarm_at - INTERVAL 1 MICROSECOND) AS as_of,
                  count(*) FILTER (WHERE left_boundary = 'observed_quiet_gap') AS known, count(*) FILTER (WHERE left_boundary = 'unknown') AS unknown
           FROM read_parquet(?) GROUP BY ALL),
     i AS (SELECT channel_id, date_trunc('day', first_alarm_at - INTERVAL 1 MICROSECOND) AS as_of,
                  count(*) FILTER (WHERE left_boundary = 'observed_quiet_gap') AS known, count(*) FILTER (WHERE left_boundary = 'unknown') AS unknown
           FROM read_parquet(?) GROUP BY ALL)
SELECT e.channel_id, e.as_of, e.split, e.label,
       CASE WHEN e.label IS NULL OR coalesce(f.unknown, 0) > 0 THEN NULL WHEN coalesce(f.known, 0) > 0 THEN 1 ELSE 0 END AS label_failure,
       CASE WHEN e.label IS NULL OR coalesce(i.unknown, 0) > 0 THEN NULL WHEN coalesce(i.known, 0) > 0 THEN 1 ELSE 0 END AS label_incident
FROM read_parquet(?) e LEFT JOIN f USING (channel_id, as_of) LEFT JOIN i USING (channel_id, as_of)
"""


def build_episodes(connection, observations, states, gap_seconds, output):
    listed = ', '.join("'" + state.replace("'", "''") + "'" for state in states)
    connection.execute(f'CREATE OR REPLACE TEMP VIEW observations AS SELECT channel_id, occurred_at, is_alarm AND raw_value IN ({listed}) AS is_alarm '
                       f'FROM read_parquet({observations!r})')
    connection.execute(f'SET VARIABLE gap_seconds = {int(gap_seconds)}')
    connection.execute(f"COPY ({EPISODES_SQL.read_text()}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def build_labels(connection, examples, failure_episodes, incident_episodes, output):
    connection.execute(f"COPY ({LABELS_SQL}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)", [str(failure_episodes), str(incident_episodes), str(examples)])


if __name__ == '__main__':
    prepared, examples, out_dir = map(Path, sys.argv[1:4])
    targets = {name: out_dir / f'episodes-900-{name}.parquet' for name in CLASSES}
    labels = out_dir / 'class-labels-v3.parquet'
    if labels.exists() or any(path.exists() for path in targets.values()):
        raise FileExistsError(str(out_dir))
    files = sorted(str(path) for path in prepared.glob('20??.parquet'))
    with tempfile.TemporaryDirectory(dir=out_dir) as scratch:
        with duckdb.connect(config=dict(threads=4, memory_limit='4GB', temp_directory=scratch)) as connection:
            for name, states in CLASSES.items():
                build_episodes(connection, files, states, 900, targets[name])
            build_labels(connection, examples, targets['failure'], targets['incident'], labels)
