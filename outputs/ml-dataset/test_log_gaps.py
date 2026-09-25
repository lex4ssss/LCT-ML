from datetime import datetime, timedelta
from pathlib import Path
import unittest
import duckdb


class LogGapTests(unittest.TestCase):
    def test_twenty_hour_boundary_cases(self):
        query=Path(__file__).with_name('log_gaps.sql').read_text()
        with duckdb.connect() as c:
            c.execute('CREATE TABLE observations(occurred_at TIMESTAMP)')
            c.execute('SET VARIABLE gap_threshold_seconds=3600')
            for case in range(20):
                start=datetime(2024,4,5,22,1,30)+timedelta(minutes=case*7)
                seconds=(3599,3600,3601,86400,275299)[case%5]
                times=[start-timedelta(seconds=1),start,start+timedelta(seconds=seconds),start+timedelta(seconds=seconds+1)]
                c.execute('DELETE FROM observations')
                c.executemany('INSERT INTO observations VALUES (?)',[(t,) for t in reversed(times)])
                actual=c.execute(query).fetchall()
                expected=[(a+timedelta(microseconds=1),b,(b-a).total_seconds()) for a,b in zip(times,times[1:]) if (b-a).total_seconds()>=3600]
                with self.subTest(case=case):
                    self.assertEqual(actual,expected)


if __name__ == '__main__':
    unittest.main()
