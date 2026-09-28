from datetime import datetime, timedelta
import unittest
import numpy as np
import object_predictor as op
import explore_v8 as e8
import experiment_v2 as ex


class ReadableTests(unittest.TestCase):
    def test_values_read_like_numbers(self):
        self.assertEqual(op.readable('событий за 7 сут: {}', 758123.0), '758 123')
        self.assertEqual(op.readable('доля каналов: {}', 0.256), '26%')
        self.assertEqual(op.readable('часов с последней тревоги: {}', 10.84), '10.8')
        self.assertEqual(op.readable('каналов: {}', 3.0), '3')
        self.assertEqual(op.readable('каналов: {}', np.nan), 'нет данных')


class DispatcherTextTests(unittest.TestCase):
    def test_blind_spots_name_missing_families_and_silent_channels(self):
        self.assertEqual(op.blind_spots(['Датчик дыма', 'КД Люк', 'Газовый датчик', None], 3, 5),
                         ['нет датчиков тепловых и температурных', 'нет датчиков затопления', '2 из 5 каналов без записей за 30 суток'])
        full = ['Датчик дыма', 'Датчик температуры', 'Датчик затопления', 'Газовый датчик', 'КД Дверь']
        self.assertEqual(op.blind_spots(full, 5, 5), [])

    def test_verdict_names_target_factor_and_suspects_only_on_alert(self):
        factors = [dict(text='тревог за 7 сут, сумма по каналам объекта: 12')]
        suspects = [dict(target_id='sensor_11', sensor_type='КД Дверь'), dict(target_id='sensor_12', sensor_type=None),
                    dict(target_id='sensor_13', sensor_type='Датчик дыма'), dict(target_id='sensor_14', sensor_type='Датчик дыма')]
        self.assertEqual(op.verdict('incident', '7', 0.834, 72, True, factors, suspects),
                         'Инцидент на объекте 7: 83 % за 72 ч, выше порога тревоги. Главный признак: тревог за 7 сут, сумма по каналам объекта: 12. '
                         'Проверить: КД Дверь 11, канал 12, Датчик дыма 13')
        self.assertEqual(op.verdict('failure', '7', 0.1, 168, False, [], suspects), 'Отказ оборудования на объекте 7: 10 % за 168 ч, ниже порога тревоги')


class HistoryRowTests(unittest.TestCase):
    def test_windows_age_and_fallbacks(self):
        t = datetime(2025, 3, 10)
        starts = [t, t - timedelta(hours=23, minutes=59), t - timedelta(hours=24), t - timedelta(days=6), t - timedelta(days=29, hours=23)]
        np.testing.assert_allclose(op.class_history_row(t, starts, True), [2, 4, 5, 0.0])
        np.testing.assert_allclose(op.class_history_row(t, starts[3:], True), [0, 1, 2, 144.0])
        np.testing.assert_allclose(op.class_history_row(t, [], True), [0, 0, 0, 720.0])
        np.testing.assert_allclose(op.class_history_row(t, [], False), [0, 0, 0, np.nan])

    def test_texts_follow_aggregate_layout(self):
        rng = np.random.default_rng(0)
        n, types = 12, ['a', 'b', 'c']
        columns = {name: rng.integers(0, 3, n).astype(np.float64) for name in ex.V2_FEATURES}
        columns['as_of'] = np.full(n, np.datetime64('2025-01-01', 'us'))
        _, matrix = e8.aggregate(columns, np.zeros(n, dtype=np.int64), rng.integers(0, 3, n), 3)
        texts = op.feature_texts(types)
        self.assertEqual(len(texts), matrix.shape[1] + 4)
        self.assertIn('«c»', texts[1 + len(e8.SUMS) + len(e8.MAXIMA) + len(e8.MINIMA) + 2 * len(e8.ACTIVE) + 2])
        self.assertIn('ведущий канал', texts[-4 - 2 - len(ex.V2_FEATURES)])
        self.assertIn('месяц', texts[-6])
        self.assertIn('инцидентов', texts[-1])
        failure = op.feature_texts(types, 'neispraven')
        self.assertTrue(all('«Неисправен»' in text for text in failure[-4:]))
        self.assertEqual(failure[:-4], texts[:-4])


if __name__ == '__main__':
    unittest.main()
