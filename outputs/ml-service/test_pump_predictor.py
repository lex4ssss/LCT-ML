from http.server import ThreadingHTTPServer
import json
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

import pump_predictor
from pump_predictor import BadReadings, PumpPredictor, pumps_data
import service


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict_proba(self, matrix):
        return np.column_stack([1 - np.full(len(matrix), self.value), np.full(len(matrix), self.value)])


def make_predictor(b=0.6, c=0.4):
    predictor = PumpPredictor.__new__(PumpPredictor)
    predictor.model_name = 'pumps-test'
    predictor.profile = {pump: np.linspace(1, 2, 12) for pump in pumps_data.PUMPS}
    fitted = dict(A=('a3', ConstantModel(0.1)), B=('v3', ConstantModel(b)), C=('v3', ConstantModel(c)))
    predictor.models = {'30min': fitted, '1h': fitted}
    predictor.decision = dict(horizons={'30min': dict(horizon_hours=0.5, variant='D', threshold=0.5),
                                        '1h': dict(horizon_hours=1.0, variant='D', threshold=0.51)})
    return predictor


def readings(count, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(zip(pumps_data.SENSORS, map(float, row))) for row in rng.uniform(1, 3, (count, 12))]


class FeatureParityTest(unittest.TestCase):
    def test_last_row_matches_experiment_features(self):
        predictor = make_predictor()
        for count in (1, 5, 13, 30, 97):
            rows = readings(count, seed=count)
            frame = pd.DataFrame(rows)
            frame['pump_id'], frame['trajectory'] = 'NPV_2_2', 0
            profile = np.tile(predictor.profile['NPV_2_2'], (count, 1))
            expected = pumps_data.feature_frame_v3(frame, np.ones(count, dtype=bool), profile).iloc[[-1]]
            got = predictor.features('NPV_2_2', rows)['v3']
            pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))

    def test_reading_96_steps_back_sets_the_96_step_change(self):
        predictor = make_predictor()
        rows = readings(97)
        changed = [dict(rows[0], motor_current=500.0)] + rows[1:]
        got = predictor.features('NPV_2_1', changed)['v3']
        self.assertAlmostEqual(got['motor_current_change_96'].iloc[0], rows[-1]['motor_current'] - 500.0)
        pd.testing.assert_series_equal(got['motor_current_change_36'], predictor.features('NPV_2_1', rows)['v3']['motor_current_change_36'])

    def test_short_history_is_padded_with_normal_profile(self):
        predictor = make_predictor()
        rows = readings(3)
        got = predictor.features('NPV_2_3', rows)['v3']
        self.assertAlmostEqual(got['motor_current_change_12'].iloc[0], rows[-1]['motor_current'] - predictor.profile['NPV_2_3'][4])


class ValidationTest(unittest.TestCase):
    def test_bad_inputs(self):
        predictor = make_predictor()
        broken = readings(2)
        del broken[1]['motor_current']
        nan = readings(2)
        nan[0]['motor_current'] = float('nan')
        for pump, rows in (('NPV_9', readings(2)), ('NPV_2_1', []), ('NPV_2_1', readings(98)), ('NPV_2_1', broken),
                           ('NPV_2_1', nan), ('NPV_2_1', 'x'), ('NPV_2_1', [dict(readings(1)[0], motor_current='a')])):
            with self.assertRaises(BadReadings):
                predictor.predict(pump, rows)


class PredictTest(unittest.TestCase):
    def test_ensemble_mean_and_thresholds(self):
        result = make_predictor(0.6, 0.41).predict('NPV_2_4', readings(20))
        first, second = result['forecasts']
        self.assertAlmostEqual(first['probability'], 0.505)
        self.assertEqual((first['alert'], second['alert']), (True, False))
        self.assertEqual((first['horizon_hours'], second['horizon_hours'], second['model_threshold']), (0.5, 1.0, 0.51))
        self.assertTrue(result['synthetic'])


class RouteTest(unittest.TestCase):
    def serve(self, pumps):
        server = ThreadingHTTPServer(('127.0.0.1', 0), service.make_handler(None, None, pumps))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_port}/predict_pump'

    def post(self, url, body):
        request = urllib.request.Request(url, data=json.dumps(body).encode(), method='POST')
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_routes(self):
        url = self.serve(make_predictor())
        code, body = self.post(url, dict(pump_id='NPV_2_1', readings=readings(10)))
        self.assertEqual((code, body['status'], len(body['forecasts'])), (200, 'ok', 2))
        self.assertEqual(self.post(url, dict(pump_id='NPV_2_1', readings=[]))[0], 400)
        self.assertEqual(self.post(url, dict(readings=readings(2)))[0], 400)

    def test_absent_without_pump_model(self):
        self.assertEqual(self.post(self.serve(None), dict(pump_id='NPV_2_1', readings=readings(2)))[0], 404)


if __name__ == '__main__':
    unittest.main()
