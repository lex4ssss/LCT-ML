from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from zoneinfo import ZoneInfo
import duckdb

from prepare import prepare


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = self.root / 'raw.parquet'
        self.output = self.root / 'prepared.parquet'

    def write(self, rows):
        with duckdb.connect() as c:
            c.execute('CREATE TABLE source(event_id VARCHAR,channel_id VARCHAR,date VARCHAR,time VARCHAR,alarm VARCHAR,value VARCHAR)')
            if rows:
                c.executemany('INSERT INTO source VALUES (?,?,?,?,?,?)', rows)
            c.execute("COPY source TO '" + str(self.raw) + "' (FORMAT PARQUET)")

    def test_exact_duplicates_only(self):
        self.write([['001','01','2026-01-01','03:00:00','true','x']] * 2 +
                   [['001','02','2026-01-01','04:00:00','false','y'],
                    ['ид_события','ид_канала_данных','дата','время','тревожное','значение_датчика']])
        report = prepare(self.raw, self.output, True)
        self.assertEqual((report['output_rows'], report['exact_duplicates_removed'], report['repeated_headers_removed']), (2,1,1))
        with duckdb.connect() as c:
            result = c.execute('SELECT event_id,channel_id,occurred_at,is_alarm,raw_value FROM read_parquet(?) ORDER BY channel_id', [str(self.output)]).fetchall()
        self.assertEqual(result[0], ('001','01',datetime(2026,1,1),True,'x'))
        self.assertEqual(result[1][1], '02')

    def test_deduplication_can_be_disabled(self):
        self.write([['1','01','2026-01-01','00:00:00','true','x']] * 2)
        report = prepare(self.raw, self.output, False)
        self.assertEqual(report['output_rows'], 2)
        self.assertEqual(report['exact_duplicates_removed'], 0)

    def test_invalid_late_record_is_atomic(self):
        base = ['1','01','2026-01-01','00:00:00','true','x']
        for field, value in ((0,''),(1,None),(2,'2026-1-1'),(2,'invalid'),(3,'25:00:00'),(4,'unknown')):
            with self.subTest(field=field, value=value):
                bad = list(base)
                bad[field] = value
                self.write([base,bad])
                with self.assertRaises(duckdb.Error):
                    prepare(self.raw, self.output, True)
                self.assertFalse(self.output.exists())
                self.assertEqual([p.name for p in self.root.iterdir()], ['raw.parquet'])

    def test_timezone_matches_python(self):
        rows = []
        expected = []
        for i in range(20):
            local = datetime(2019+i%8, 1+i%12, 1+i%27, i%24, i%60, i%60)
            rows.append([str(i),'01',local.strftime('%Y-%m-%d'),local.strftime('%H:%M:%S'),'f','0'])
            expected.append(local.replace(tzinfo=ZoneInfo('Europe/Moscow')).astimezone(timezone.utc).replace(tzinfo=None))
        self.write(rows)
        prepare(self.raw,self.output,False)
        with duckdb.connect() as c:
            actual = [r[0] for r in c.execute('SELECT occurred_at FROM read_parquet(?) ORDER BY cast(event_id AS INT)', [str(self.output)]).fetchall()]
        self.assertEqual(actual, expected)

    def test_existing_output_and_invalid_flag(self):
        self.output.write_text('preserve')
        with self.assertRaises(FileExistsError):
            prepare(self.raw,self.output,True)
        self.assertEqual(self.output.read_text(), 'preserve')
        self.output.unlink()
        with self.assertRaises(ValueError):
            prepare(self.raw,self.output,1)

    def test_empty_input_has_schema(self):
        self.write([])
        report = prepare(self.raw,self.output,True)
        self.assertEqual(report['output_rows'],0)
        with duckdb.connect() as c:
            relation = c.read_parquet(str(self.output))
            self.assertEqual(relation.columns,['event_id','channel_id','occurred_at','is_alarm','raw_value'])


if __name__ == '__main__':
    unittest.main()
