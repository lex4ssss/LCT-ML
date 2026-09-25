from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tempfile
import duckdb
from examples import build_examples
from prepare import literal


source, output, config_path = map(Path, sys.argv[1:4])
config = json.loads(config_path.read_text())
files = sorted(source.glob('20??.parquet'))
if len(files) != 8 or config['lookback_hours'] != 24 or config['horizon_hours'] != 24:
    raise ValueError('Expected all eight years and 24-hour windows')
if output.exists():
    raise FileExistsError(str(output))
episode_run = json.loads(config_path.with_name('episodes-900-run.json').read_text())
if config['gap_seconds'] != episode_run['gap_seconds']:
    raise ValueError('Episode grouping parameter mismatch')
gap_rows = [(datetime.fromisoformat(g['start_at']),datetime.fromisoformat(g['end_at'])) for g in config['gap_intervals_utc']]
if any(a.tzinfo is not None or b.tzinfo is not None or a>=b for a,b in gap_rows):
    raise ValueError('Invalid UTC gap interval')
with tempfile.TemporaryDirectory(dir=output.parent) as directory:
    with duckdb.connect(config=dict(memory_limit='1GB',threads=1,temp_directory=directory,max_temp_directory_size='8GB')) as c:
        c.execute('CREATE VIEW observations AS SELECT channel_id,occurred_at,is_alarm FROM read_parquet(['+','.join(map(literal,files))+'])')
        c.execute('CREATE VIEW episodes AS SELECT channel_id,first_alarm_at,left_boundary FROM read_parquet('+literal(source/'episodes-900.parquet')+')')
        c.execute('CREATE TABLE gaps(start_at TIMESTAMP,end_at TIMESTAMP)')
        if gap_rows:
            c.executemany('INSERT INTO gaps VALUES (?,?)',gap_rows)
        build_examples(c,config['gap_seconds'],datetime.fromisoformat(config['validation_start_utc']),datetime.fromisoformat(config['test_start_utc']))
        temporary = Path(directory)/'examples.parquet'
        c.execute('COPY training_examples TO '+literal(temporary)+' (FORMAT PARQUET, COMPRESSION ZSTD)')
        c.execute('CREATE VIEW saved AS SELECT * FROM read_parquet('+literal(temporary)+')')
        invalid = c.execute('SELECT count(*) FROM saved WHERE (label IS NULL)!=(exclusion_reason IS NOT NULL) OR event_count_24h<1 OR alarm_count_24h>event_count_24h OR last_event_age_hours<0 OR last_event_age_hours>=24 OR label NOT IN (0,1)').fetchone()[0]
        if invalid:
            raise ValueError('Dataset invariant failed')
        cursor = c.execute('SELECT split,exclusion_reason,count(*) AS row_count,count(*) FILTER(WHERE label=1) AS positives,min(as_of) AS first_as_of,max(as_of) AS last_as_of FROM saved GROUP BY split,exclusion_reason ORDER BY split,exclusion_reason')
        summary = [dict(zip([d[0] for d in cursor.description],row)) for row in cursor.fetchall()]
        os.link(temporary,output)
print(json.dumps(dict(config=config,output_bytes=output.stat().st_size,groups=summary,training=False,inference=False),default=str,indent=2))
