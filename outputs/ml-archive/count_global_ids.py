import json
from pathlib import Path
import sys
import tempfile
import time
import duckdb


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def count_partitioned(connection, directory):
    connection.execute("COPY (SELECT trim(event_id) AS event_id, hash(trim(event_id)) % 64 AS bucket FROM ids_source WHERE event_id IS NOT NULL) TO " + literal(directory) + " (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY(bucket), ROW_GROUP_SIZE 16384)")
    rows = distinct = partitions = 0
    for folder in sorted(Path(directory).glob('bucket=*')):
        source = literal(folder / '*.parquet')
        counts = connection.execute('SELECT count(*), count(DISTINCT event_id) FROM read_parquet(' + source + ')').fetchone()
        rows += counts[0]
        distinct += counts[1]
        partitions += 1
    return dict(nonnull_id_rows=rows, distinct_event_ids=distinct, partitions=partitions)


if __name__ == '__main__':
    cache, output = map(Path, sys.argv[1:3])
    if output.exists():
        raise FileExistsError(str(output))
    files = sorted(cache.glob('20??.parquet'))
    if len(files) != 8:
        raise ValueError('Expected eight yearly Parquet files')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        with duckdb.connect(config=dict(memory_limit='1GB', threads=1, preserve_insertion_order=False, partitioned_write_max_open_files=8)) as connection:
            connection.execute('CREATE VIEW ids_source AS SELECT event_id FROM read_parquet([' + ','.join(map(literal, files)) + '])')
            result = count_partitioned(connection, directory)
    result.update(method='64_hash_partitions_exact_string_distinct', elapsed_seconds=round(time.monotonic()-started, 3), memory_limit='1GB', threads=1, flags=dict(full_scan=True, exact_distinct=True, deduplication=False, training=False, inference=False))
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
