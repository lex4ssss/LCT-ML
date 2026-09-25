import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import duckdb

source, cache, reports = map(Path, sys.argv[1:4])
cache.mkdir(parents=True, exist_ok=True)
reports.mkdir(parents=True, exist_ok=True)
literal = lambda value: "'" + str(value).replace("'", "''") + "'"
options = "all_varchar=true, header=true, delim=',', parallel=false, strict_mode=true, ignore_errors=false"
columns = ['ид_события', 'ид_канала_данных', 'дата', 'время', 'тревожное', 'значение_датчика']
fields = "row_number() OVER () AS source_row, " + ", ".join('"' + a + '" AS ' + b for a, b in zip(columns, ['event_id', 'channel_id', 'date', 'time', 'alarm', 'value']))
force = ', force_not_null=[' + ','.join(map(literal, columns)) + ']'
for archive in sorted(source.glob('ext-journal-*.7z'), reverse=True):
    year = archive.stem.rsplit('-', 1)[1]
    report_path = reports / (year + '.json')
    if report_path.exists():
        raise FileExistsError(str(report_path))
    if shutil.disk_usage(cache).free < 8 * 1024 ** 3:
        raise RuntimeError('Less than 8 GiB of free disk space')
    started = time.monotonic()
    digest = hashlib.sha256()
    with archive.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    csv_path = cache / (archive.stem + '.csv')
    with csv_path.open('wb') as destination:
        subprocess.run(['bsdtar', '-xOf', str(archive), archive.stem + '.csv'], stdout=destination, check=True)
    parquet = cache / (year + '.parquet')
    temporary = cache / (year + '.partial.parquet')
    if parquet.exists():
        raise FileExistsError(str(parquet))
    config = dict(memory_limit='1GB', threads=2, temp_directory=str(cache / 'spill'), max_temp_directory_size='6GB')
    with duckdb.connect(config=config) as connection:
        connection.execute('COPY (SELECT ' + fields + ' FROM read_csv(' + literal(csv_path) + ', ' + options + force + ')) TO ' + literal(temporary) + ' (FORMAT PARQUET, COMPRESSION ZSTD)')
        temporary.rename(parquet)
        connection.execute('CREATE VIEW events AS SELECT * FROM read_parquet(' + literal(parquet) + ')')
        connection.execute('CREATE VIEW catalogue AS SELECT * FROM read_csv(' + literal(source / 'справочник_каналов_датчиков.csv') + ', ' + options + ')')
        result = connection.execute(Path(__file__).with_name('profile.sql').read_text())
        summary = dict(zip([d[0] for d in result.description], result.fetchone()))
        connection.execute('COPY (SELECT channel_id, cast(occurred_local AS DATE) AS day, count(*) AS rows, count(*) FILTER(WHERE alarm_token IN (\'true\',\'t\')) AS alarm_rows, min(occurred_local) AS first_local, max(occurred_local) AS last_local FROM observations GROUP BY channel_id, day) TO ' + literal(cache / (year + '-channel-day.parquet')) + ' (FORMAT PARQUET, COMPRESSION ZSTD)')
        summary['daily'] = [dict(zip(['day','rows','alarm_rows','channels'], row)) for row in connection.execute('SELECT day, sum(rows), sum(alarm_rows), count(*) FROM read_parquet(' + literal(cache / (year + '-channel-day.parquet')) + ') GROUP BY day ORDER BY day').fetchall()]
    summary.update(archive=archive.name, archive_sha256=digest.hexdigest(), csv_bytes=csv_path.stat().st_size, parquet_bytes=parquet.stat().st_size, elapsed_seconds=round(time.monotonic()-started, 3), duckdb_version=duckdb.__version__, flags=dict(full_scan=True, strict_csv=True, numeric_coercion=False, invalid_record_filter=False, deduplication=False, source_timezone='Europe/Moscow_assumed', training=False, inference=False), memory_limit='1GB', threads=2)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, default=str, indent=2) + '\n')
    csv_path.unlink()
    print(json.dumps({k:v for k,v in summary.items() if k != 'daily'}, ensure_ascii=False, default=str), flush=True)
