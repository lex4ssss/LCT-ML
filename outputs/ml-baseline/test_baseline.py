import contextlib
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
import duckdb
import joblib
import numpy as np


SCRIPT = Path(__file__).with_name('fit_baseline.py')


class BaselineTests(unittest.TestCase):
    def test_holdout_changes_do_not_affect_fit_or_selection(self):
        reference = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with duckdb.connect() as connection:
                connection.execute('CREATE TABLE examples(channel_id VARCHAR,as_of TIMESTAMP,event_count_24h BIGINT,alarm_count_24h BIGINT,last_event_age_hours DOUBLE,last_alarm_age_hours_24h DOUBLE,split VARCHAR,label INTEGER)')
                rows = []
                for split in ['train','validation']:
                    for index in range(60):
                        alarms = index%4
                        rows.append((str(index),datetime(2024,1,1)+timedelta(days=index),index%8+4,alarms,float(index%24),float(index%12) if alarms else None,split,int(index%5==0)))
                connection.executemany('INSERT INTO examples VALUES (?,?,?,?,?,?,?,?)',rows)
                for case in range(20):
                    connection.execute("DELETE FROM examples WHERE split='test'")
                    connection.execute("INSERT INTO examples VALUES ('holdout',TIMESTAMP '2026-01-01',?,?,?,?, 'test',?)",[-case-1,case,1000+case,None,case+7])
                    source = root/f'input-{case}.parquet'
                    connection.execute('COPY examples TO ? (FORMAT PARQUET)',[str(source)])
                    output = root/f'output-{case}'
                    with patch.object(sys,'argv',[str(SCRIPT),str(source),str(output)]),contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(SCRIPT),run_name='__main__')
                    model = joblib.load(output/'model.joblib')
                    report = json.loads((output/'selection.json').read_text())
                    current = (model[-1].coef_.tolist(),model[-1].intercept_.tolist(),report)
                    with self.subTest(case=case):
                        if reference is not None:
                            self.assertEqual(current,reference)
                        self.assertEqual(report['train_rows'],60)
                        self.assertEqual(report['validation_rows'],60)
                        self.assertFalse(report['test_evaluation'])
                    reference = current

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory)/'marker'
            marker.write_text('preserve')
            with patch.object(sys,'argv',[str(SCRIPT),'missing.parquet',directory]),self.assertRaises(FileExistsError):
                runpy.run_path(str(SCRIPT),run_name='__main__')
            self.assertEqual(marker.read_text(),'preserve')


if __name__ == '__main__':
    unittest.main()
