from datetime import datetime, timedelta, timezone
import random
import unittest
import duckdb

from examples import build_examples


DAY = timedelta(days=1)
VALIDATION = datetime(2025,1,1)
TEST = datetime(2026,1,1)
T = datetime(2024,6,3)
COLUMNS = ['channel_id','as_of','event_count_24h','alarm_count_24h','last_event_age_hours','last_alarm_age_hours_24h','split','label','exclusion_reason','label_scope']


def reference(observations, episodes, gaps, gap, validation=VALIDATION, test=TEST):
    if not observations:
        return []
    first = min(r[1] for r in observations)
    last = max(r[1] for r in observations)
    snapshots = set()
    for channel, timestamp, alarm in observations:
        midnight = timestamp.replace(hour=0,minute=0,second=0,microsecond=0)
        snapshots.add((channel,midnight if timestamp == midnight else midnight+DAY))
    result = []
    for channel, t in sorted(snapshots):
        past = [r for r in observations if r[0]==channel and t-DAY<r[1]<=t]
        alarm_times = [r[1] for r in past if r[2]]
        latest_alarm = max(alarm_times) if alarm_times else None
        channel_first = min(r[1] for r in observations if r[0]==channel)
        future = [r for r in episodes if r[0]==channel and t<r[1]<=t+DAY]
        split = 'train' if t<validation else 'validation' if t<test else 'test'
        reasons = [(t-DAY<first,'global_history'),(t-DAY<channel_first,'channel_history'),
                   (t+DAY>last,'future_window'),
                   ((split=='train' and t+DAY>=validation) or (split=='validation' and t+DAY>=test),'split_boundary'),
                   (any(a<=t+DAY and b>t-DAY for a,b in gaps),'known_log_gap'),
                   (latest_alarm is not None and latest_alarm>=t-timedelta(seconds=gap),'active_alarm'),
                   (any(r[2]=='unknown' for r in future),'uncertain_first_episode')]
        reason = next((name for condition,name in reasons if condition),None)
        label = None if reason else int(any(r[2]=='observed_quiet_gap' for r in future))
        result.append((channel,t,len(past),len(alarm_times),(t-max(r[1] for r in past)).total_seconds()/3600,
                       (t-latest_alarm).total_seconds()/3600 if latest_alarm else None,split,label,reason,'recorded_alarm_episode'))
    return result


class ExampleTests(unittest.TestCase):
    def run_case(self, observations, episodes=(), gaps=(), gap=900, validation=VALIDATION, test=TEST):
        with duckdb.connect() as c:
            c.execute('CREATE TABLE observations(channel_id VARCHAR,occurred_at TIMESTAMP,is_alarm BOOLEAN)')
            c.execute('CREATE TABLE episodes(channel_id VARCHAR,first_alarm_at TIMESTAMP,left_boundary VARCHAR)')
            c.execute('CREATE TABLE gaps(start_at TIMESTAMP,end_at TIMESTAMP)')
            for table,rows in [('observations',observations),('episodes',episodes),('gaps',gaps)]:
                if rows:
                    c.executemany('INSERT INTO '+table+' VALUES ('+','.join(['?']*len(rows[0]))+')',rows)
            self.assertIsNone(build_examples(c,gap,validation,test))
            actual = c.execute('SELECT * FROM training_examples ORDER BY channel_id,as_of')
            self.assertEqual([d[0] for d in actual.description],COLUMNS)
            self.assertEqual([str(d[1]) for d in actual.description],['VARCHAR','TIMESTAMP','BIGINT','BIGINT','DOUBLE','DOUBLE','VARCHAR','INTEGER','VARCHAR','VARCHAR'])
            result = actual.fetchall()
        self.assertEqual(result,reference(observations,episodes,gaps,gap,validation,test))
        return result

    def base(self,t=T):
        return [('01',t-2*DAY,False),('01',t-timedelta(hours=12),False),('01',t,False),('01',t+2*DAY,False)]

    def test_future_endpoints_and_multiple_starts(self):
        for offset in (timedelta(0),timedelta(microseconds=1),DAY,DAY+timedelta(microseconds=1)):
            with self.subTest(offset=offset):
                rows = self.run_case(self.base(),[('01',T+offset,'observed_quiet_gap')])
                row = next(r for r in rows if r[1]==T)
                self.assertEqual(row[7],int(timedelta(0)<offset<=DAY))
        self.run_case(self.base(),[('01',T+timedelta(hours=h),'observed_quiet_gap') for h in (1,3,8)])

    def test_active_alarm_boundary(self):
        for seconds in (899,900,901):
            with self.subTest(seconds=seconds):
                rows = self.run_case(self.base()+[('01',T-timedelta(seconds=seconds),True)])
                row = next(r for r in rows if r[1]==T)
                self.assertEqual(row[8],'active_alarm' if seconds<=900 else None)

    def test_gap_endpoints(self):
        cases = [(T-2*DAY,T-DAY),(T-DAY,T-DAY+timedelta(microseconds=1)),
                 (T+DAY,T+2*DAY),(T+DAY+timedelta(microseconds=1),T+2*DAY)]
        for gap in cases:
            with self.subTest(gap=gap):
                self.run_case(self.base(),gaps=[gap])

    def test_split_purge_and_cutoff_assignment(self):
        for t in (VALIDATION-2*DAY,VALIDATION-DAY,VALIDATION,TEST-2*DAY,TEST-DAY,TEST):
            with self.subTest(t=t):
                self.run_case(self.base(t))

    def test_unknown_starts_and_channel_history(self):
        self.run_case(self.base(),[('01',T+timedelta(hours=1),'unknown')])
        self.run_case(self.base()+[('02',T-timedelta(hours=1),False)])

    def test_empty_inputs_and_duplicates(self):
        self.run_case([])
        self.run_case(self.base()+self.base()[1:3])

    def test_microsecond_age_precision(self):
        observations=[('01',T-2*DAY,False),('01',T-timedelta(microseconds=1),False),('01',T+2*DAY,False)]
        self.run_case(observations)

    def test_view_can_be_replaced(self):
        with duckdb.connect() as c:
            c.execute('CREATE TABLE observations(channel_id VARCHAR,occurred_at TIMESTAMP,is_alarm BOOLEAN)')
            c.execute('CREATE TABLE episodes(channel_id VARCHAR,first_alarm_at TIMESTAMP,left_boundary VARCHAR)')
            c.execute('CREATE TABLE gaps(start_at TIMESTAMP,end_at TIMESTAMP)')
            c.executemany('INSERT INTO observations VALUES (?,?,?)',self.base())
            build_examples(c,900,VALIDATION,TEST)
            build_examples(c,900,datetime(2024,1,1),VALIDATION)
            self.assertEqual(c.execute('SELECT DISTINCT split FROM training_examples').fetchall(),[('validation',)])
            self.assertEqual(c.execute('SELECT count(*) FROM observations').fetchone()[0],4)

    def test_features_unchanged_when_future_changes(self):
        before = self.run_case(self.base())
        after = self.run_case(self.base()+[('01',T+timedelta(seconds=1),True),('02',T+DAY,False)],
                              [('01',T+timedelta(seconds=1),'observed_quiet_gap')])
        self.assertEqual(next(r[:6] for r in before if r[1]==T),next(r[:6] for r in after if r[0]=='01' and r[1]==T))
        self.assertFalse(any(r[0]=='02' and r[1]==T for r in after))

    def test_seeded_reference(self):
        rng = random.Random(606)
        for trial in range(20):
            observations = []
            episodes = []
            for channel in ('01','02'):
                for i in range(80):
                    timestamp = T+timedelta(seconds=rng.randrange(-4*86400,4*86400))
                    alarm = rng.random()<0.2
                    observations.append((channel,timestamp,alarm))
                    if alarm: episodes.append((channel,timestamp,rng.choice(['unknown','observed_quiet_gap'])))
            rng.shuffle(observations)
            with self.subTest(trial=trial):
                self.run_case(observations,episodes,[(T+DAY,T+DAY+timedelta(hours=1))])

    def test_invalid_arguments_fail_before_sql(self):
        class NoSQL:
            def execute(self,*args,**kwargs):
                raise AssertionError('SQL executed')
        for gap,validation,test in [(True,VALIDATION,TEST),(0,VALIDATION,TEST),(-1,VALIDATION,TEST),(86400,VALIDATION,TEST),
                                    (1.0,VALIDATION,TEST),(900,TEST,VALIDATION),
                                    (900,VALIDATION,VALIDATION),(900,'2025-01-01',TEST),
                                    (900,VALIDATION.replace(tzinfo=timezone.utc),TEST)]:
            with self.subTest(gap=gap,validation=validation,test=test),self.assertRaises(ValueError):
                build_examples(NoSQL(),gap,validation,test)


if __name__ == '__main__':
    unittest.main()
