import json
import os
from pathlib import Path
import tempfile
import duckdb


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def prepare(raw_path, output_path, deduplicate):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(str(output_path))
    if type(deduplicate) is not bool:
        raise ValueError('deduplicate')
    query = Path(__file__).with_name('normalize.sql').read_text()
    header = "event_id='ид_события' AND channel_id='ид_канала_данных' AND date='дата' AND time='время' AND alarm='тревожное' AND value='значение_датчика'"
    with tempfile.TemporaryDirectory(dir=output_path.parent) as directory:
        config = dict(memory_limit='1GB', threads=1, preserve_insertion_order=False, temp_directory=directory, max_temp_directory_size='8GB', autoinstall_known_extensions=False)
        with duckdb.connect(config=config) as connection:
            connection.execute('CREATE VIEW raw_events AS SELECT * FROM read_parquet(' + literal(raw_path) + ')')
            original = connection.execute('SELECT count(*), count(*) FILTER(WHERE ' + header + ') FROM raw_events').fetchone()
            prefix = 'DISTINCT ' if deduplicate else ''
            connection.execute('CREATE VIEW records AS SELECT ' + prefix + 'event_id,channel_id,date,time,alarm,value FROM raw_events WHERE NOT coalesce((' + header + '),false)')
            temporary = Path(directory) / 'prepared.parquet'
            connection.execute('COPY (' + query + ') TO ' + literal(temporary) + ' (FORMAT PARQUET, COMPRESSION ZSTD)')
            result = connection.execute('SELECT count(*), count(*) FILTER(WHERE is_alarm) FROM read_parquet(' + literal(temporary) + ')').fetchone()
            os.link(temporary, output_path)
    return dict(input_rows=original[0], repeated_headers_removed=original[1], output_rows=result[0], alarm_rows=result[1], exact_duplicates_removed=original[0]-original[1]-result[0], flags=dict(exact_tuple_deduplication=deduplicate, header_removal=True, utc_normalization=True, source_timezone='Europe/Moscow_assumed', other_invalid_rows_skipped=False, numeric_value_coercion=False, training=False, inference=False))
