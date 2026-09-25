from pathlib import Path
import tempfile
import unittest
import duckdb
import numpy as np
import eval_units as eu
import experiment_v2 as ex
import explore_v8 as e8
from exp_metrics_core import confusion

DAY = np.timedelta64(86400 * 10**6, 'us')
START = np.datetime64('2025-01-01', 'us')


def synthetic(rng, n):
    columns = {name: rng.integers(0, 4, n).astype(np.float64) * (rng.random(n) < 0.4) for name in ex.V2_FEATURES}
    columns['last_alarm_age_hours_30d'] = np.where(rng.random(n) < 0.5, np.nan, rng.integers(1, 700, n).astype(np.float64))
    columns['last_event_age_hours'] = np.where(rng.random(n) < 0.1, np.nan, rng.integers(1, 30, n).astype(np.float64))
    columns['as_of'] = START + rng.integers(0, 6, n) * DAY
    return columns


class AggregateTests(unittest.TestCase):
    def test_matches_brute_force(self):
        rng = np.random.default_rng(3)
        n, type_count = 2000, 4
        columns, codes, sensor = synthetic(rng, n), rng.integers(0, 5, n), rng.integers(-1, type_count, n)
        keys, matrix = e8.aggregate(columns, codes, sensor, type_count)
        row_keys = eu.day_keys(codes, columns['as_of'])
        self.assertEqual(len(keys), len(np.unique(row_keys)))
        for position, key in enumerate(keys):
            rows = np.flatnonzero(row_keys == key)
            value = lambda name: columns[name][rows]
            expected = [len(rows)] + [np.nansum(value(name)) for name in e8.SUMS] + [value(name).max() for name in e8.MAXIMA]
            expected += [np.nanmin(value(name)) if np.isfinite(value(name)).any() else np.nan for name in e8.MINIMA]
            active = [float((value(name) > 0).sum()) for name in e8.ACTIVE]
            expected += active + [count / len(rows) for count in active]
            expected += [float((sensor[rows] == t).sum()) for t in range(type_count)] + [value('episode_starts_7d')[sensor[rows] == t].sum() for t in range(type_count)]
            best = sorted(rows, key=lambda r: (-columns['alarm_count_7d'][r], -columns['episode_starts_30d'][r],
                                               np.inf if np.isnan(columns['last_alarm_age_hours_30d'][r]) else columns['last_alarm_age_hours_30d'][r], r))[0]
            expected += list(ex.matrix_v2({name: columns[name][[best]] for name in ex.V2_FEATURES}, False)[0])
            day = columns['as_of'][rows[0]].astype('datetime64[D]')
            expected += [day.astype('datetime64[M]').astype(np.int64) % 12 + 1, (day.astype(np.int64) + 3) % 7]
            np.testing.assert_allclose(matrix[position], np.array(expected, dtype=np.float64), equal_nan=True)


class HistoryAndLabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.directory, self.episodes = root / 'directory.csv', root / 'episodes.parquet'
        self.directory.write_text('"ид_канала_данных","тип_датчика","ид_объект"\n' + 'a,КД Дверь,o1\nb,Датчик дыма,o1\nc,КД Дверь,o2\n')
        self.starts = [('a', '2025-01-10 00:00:00'), ('b', '2025-01-09 23:00:00'), ('a', '2025-01-09 00:00:00'), ('b', '2025-01-03 12:00:00'),
                       ('c', '2025-01-09 12:00:00'), ('x', '2025-01-09 12:00:00'), ('a', '2024-11-01 00:00:00'), ('b', '2025-01-10 00:00:01')]
        with duckdb.connect() as c:
            c.execute('CREATE TABLE e(channel_id VARCHAR, first_alarm_at TIMESTAMP, left_boundary VARCHAR)')
            c.executemany('INSERT INTO e VALUES (?, ?::TIMESTAMP, ?)', [(ch, at, 'observed_quiet_gap') for ch, at in self.starts])
            c.execute(f"COPY e TO '{self.episodes}' (FORMAT PARQUET)")

    def tearDown(self):
        self.tmp.cleanup()

    def test_history_matches_brute_force(self):
        with duckdb.connect() as c:
            codes, code, sensor, type_count = e8.channel_info(c, self.directory, np.array(['a', 'b', 'c', 'x']))
            self.assertEqual(type_count, 2)
            self.assertEqual(list(sensor), [1, 0, 1, -1])
            keys = eu.day_keys(np.array([code['o1'], code['o2'], code['unknown'], code['o1']]),
                               np.array(['2025-01-10', '2025-01-10', '2025-01-10', '2024-10-01'], dtype='datetime64[us]'))
            history = e8.target_history(c, self.episodes, self.directory, keys)
        owner = {'a': 'o1', 'b': 'o1', 'c': 'o2', 'x': 'unknown'}
        for position, (obj, as_of) in enumerate([('o1', '2025-01-10'), ('o2', '2025-01-10'), ('unknown', '2025-01-10'), ('o1', '2024-10-01')]):
            moment = np.datetime64(as_of, 'us')
            mine = [np.datetime64(at.replace(' ', 'T'), 'us') for ch, at in self.starts if owner[ch] == obj]
            expected = [sum(moment - hours * 3600 * 10**6 * np.timedelta64(1, 'us') < at <= moment for at in mine) for hours in e8.WINDOWS_HOURS]
            past = [at for at in mine if at <= moment]
            expected.append(min((moment - max(past)) / np.timedelta64(3600, 's'), 720) if past else np.nan)
            np.testing.assert_allclose(history[position], expected, equal_nan=True)

    def test_labels_use_known_rows_only(self):
        keys = np.array([10, 20, 30])
        row_keys = np.array([10, 10, 20, 20, 30])
        labels = np.array([0.0, np.nan, 1.0, 0.0, np.nan])
        alarm = np.array([0, 5, 0, 2, 3])
        present, label, rule = e8.object_labels(keys, row_keys, labels, alarm)
        self.assertEqual(list(present), [True, True, False])
        self.assertEqual(list(label[present]), [0, 1])
        self.assertEqual(list(rule[present]), [0.0, 1.0])


class MaskTests(unittest.TestCase):
    def test_label_windows_stay_inside_parts(self):
        day = np.arange('2024-12-20', '2026-01-03', dtype='datetime64[D]').astype('datetime64[us]')
        present = np.ones(len(day), bool)
        present[3] = False
        for hours in (24, 72, 168):
            train, evaluation = e8.masks(day, present, hours)
            window = hours * np.timedelta64(3600, 's')
            self.assertEqual(day[train].max() + window + np.timedelta64(1, 'D'), np.datetime64('2025-01-01', 'us'))
            self.assertEqual(day[evaluation].min(), np.datetime64('2025-01-01', 'us'))
            self.assertEqual(day[evaluation].max() + window, np.datetime64('2026-01-01', 'us'))
            self.assertFalse(train[3] or evaluation[3])


class SelectionTests(unittest.TestCase):
    def test_margin_threshold_matches_brute_force(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            labels = (rng.random(400) < 0.3).astype(np.int64)
            scores = np.round(labels * rng.random(400) * 0.8 + rng.random(400), 2)
            threshold, margin = e8.margin_threshold(labels, scores)
            best = max(min(confusion(labels, scores, t)['precision'] - 0.7, confusion(labels, scores, t)['recall'] - 0.5) for t in np.unique(scores))
            self.assertAlmostEqual(margin, best)
            at = confusion(labels, scores, threshold)
            self.assertAlmostEqual(min(at['precision'] - 0.7, at['recall'] - 0.5), best)

    def test_qualification_and_choice(self):
        labels = np.array([1] * 30 + [0] * 70)
        strong, rule = np.r_[np.linspace(1, 0.6, 30), np.linspace(0.5, 0, 70)], np.r_[np.ones(15), np.zeros(15), np.ones(20), np.zeros(50)]
        row = e8.selection(labels, strong, rule)
        self.assertTrue(row['qualifies'])
        self.assertFalse(e8.selection(labels, strong, labels.astype(float))['qualifies'])
        self.assertFalse(e8.selection(1 - labels, 1 - strong, rule)['qualifies'])
        weak = e8.selection(labels, np.r_[np.linspace(1, 0, 30), np.linspace(1, 0, 70)], rule)
        self.assertFalse(weak['qualifies'])
        results = [dict(target='a', object_day=dict(row, margin=0.1)), dict(target='b', object_day=dict(row, margin=0.2)), dict(target='c', object_day=weak)]
        self.assertEqual(e8.choose(results)['target'], 'b')
        self.assertIsNone(e8.choose(results[2:]))


if __name__ == '__main__':
    unittest.main()
