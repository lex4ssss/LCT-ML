from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import sys
import unittest
import duckdb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml-stage2'))
from episodes import iter_episodes


class EpisodeSQLTests(unittest.TestCase):
    def compare(self,rows,gap):
        query=Path(__file__).with_name('episodes.sql').read_text()
        with duckdb.connect() as c:
            c.execute('CREATE TABLE observations(channel_id VARCHAR,occurred_at TIMESTAMP,is_alarm BOOLEAN)')
            if rows:
                c.executemany('INSERT INTO observations VALUES (?,?,?)',rows)
            c.execute('SET VARIABLE gap_seconds = '+str(gap))
            cursor=c.execute(query)
            columns=[d[0] for d in cursor.description]
            actual=[dict(zip(columns,row)) for row in cursor.fetchall()]
        normalized=[dict(channel_id=ch,event_id=str(i),occurred_at=t.replace(tzinfo=timezone.utc),is_alarm=alarm)
                    for i,(ch,t,alarm) in enumerate(sorted(rows,key=lambda r:(r[0],r[1])))]
        expected=list(iter_episodes(normalized,gap))
        for row in expected:
            for field in ('first_alarm_at','last_alarm_at'):
                row[field]=row[field].replace(tzinfo=None)
        self.assertEqual(actual,expected)

    def test_equal_timestamps_and_strict_gap(self):
        t=datetime(2026,1,1)
        rows=[('01',t,True)]*4+[('01',t+timedelta(seconds=60),True)]*3
        rows += [('01',t+timedelta(seconds=120,microseconds=1),True),('01',t+timedelta(seconds=181),False)]
        self.compare(rows,60)

    def test_empty_and_normal_only(self):
        self.compare([],60)
        self.compare([('01',datetime(2026,1,1),False)],60)

    def test_seeded_reference(self):
        rng=random.Random(823)
        for trial in range(200):
            rows=[]
            for channel in ('01','02','03'):
                timestamp=datetime(2024,12,31,23,55)
                for i in range(rng.randrange(1,70)):
                    timestamp += timedelta(seconds=rng.randrange(0,100))
                    rows.append((channel,timestamp,rng.choice([True,False])))
            rng.shuffle(rows)
            with self.subTest(trial=trial):
                self.compare(rows,rng.choice([1,60,900]))


if __name__ == '__main__':
    unittest.main()
