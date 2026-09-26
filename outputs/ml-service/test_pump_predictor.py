from http.server import ThreadingHTTPServer
import json
import threading
import unittest
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from pump_predictor import BadReadings, PumpPredictor, pumps_data, v5
import service


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict_proba(self, matrix):
        return np.column_stack([1 - np.full(len(matrix), self.value), np.full(len(matrix), self.value)])


class Router:
    classes_ = np.array(['normal', 'Несоосность'])

    def predict(self, matrix):
        return np.array(['Несоосность'] * len(matrix))


def make_predictor(values=(0.6, 0.4, 0.2)):
    predictor = PumpPredictor.__new__(PumpPredictor)
    predictor.model_name = 'pumps-test'
    predictor.max_readings = 289
    predictor.stats = dict(median={p: np.linspace(1, 2, 6) for p in pumps_data.PUMPS}, std={p: np.full(6, 0.1) for p in pumps_data.PUMPS})
    predictor.router = Router()
    fitted = dict(F=ConstantModel(values[0]), T=ConstantModel(values[1]), M=ConstantModel(values[2]))
    predictor.models = {'30min': fitted, '1h': fitted}
    predictor.decision = dict(horizons={'30min': dict(horizon_hours=0.5, variant='T', threshold=0.3),
                                        '1h': dict(horizon_hours=1.0, variant='F', threshold=0.7)})
    return predictor


def readings(count, seed=0):
    rng = np.random.default_rng(seed)
    return [dict(zip(pumps_data.SENSORS, map(float, row))) for row in rng.uniform(1, 3, (count, 12))]


class FeatureParityTest(unittest.TestCase):
    def test_last_row_matches_experiment_features(self):
        predictor = make_predictor()
        for count in (1, 5, 13, 100, 289):
            rows = readings(count, seed=count)
            frame = pd.DataFrame(rows)
            frame['pump_id'], frame['trajectory'] = 'NPV_2_2', 0
            expected = v5.feature_frame(frame, predictor.stats).iloc[[-1]]
            pd.testing.assert_frame_equal(predictor.features('NPV_2_2', rows).reset_index(drop=True), expected.reset_index(drop=True))

    def test_reading_288_rows_back_is_inside_the_288_mean(self):
        predictor = make_predictor()
        rows = readings(289)
        changed = [dict(rows[0], motor_current=500.0)] + rows[1:]
        self.assertEqual(predictor.features('NPV_2_1', changed)['motor_current_mean_288'].iloc[0],
                         predictor.features('NPV_2_1', rows)['motor_current_mean_288'].iloc[0])
        changed = [rows[0], dict(rows[1], motor_current=500.0)] + rows[2:]
        self.assertNotEqual(predictor.features('NPV_2_1', changed)['motor_current_mean_288'].iloc[0],
                            predictor.features('NPV_2_1', rows)['motor_current_mean_288'].iloc[0])


class ValidationTest(unittest.TestCase):
    def test_bad_inputs(self):
        predictor = make_predictor()
        broken = readings(2)
        del broken[1]['motor_current']
        nan = readings(2)
        nan[0]['motor_current'] = float('nan')
        for pump, rows in (('NPV_9', readings(2)), ('NPV_2_1', []), ('NPV_2_1', readings(290)), ('NPV_2_1', broken),
                           ('NPV_2_1', nan), ('NPV_2_1', 'x'), ('NPV_2_1', [dict(readings(1)[0], motor_current='a')])):
            with self.assertRaises(BadReadings):
                predictor.predict(pump, rows)


class PredictTest(unittest.TestCase):
    def test_variants_thresholds_and_diagnosis(self):
        result = make_predictor().predict('NPV_2_4', readings(20))
        first, second = result['forecasts']
        self.assertAlmostEqual(first['probability'], 0.4)
        self.assertAlmostEqual(second['probability'], 0.6)
        self.assertEqual((first['alert'], second['alert']), (True, False))
        self.assertEqual((first['horizon_hours'], second['horizon_hours'], second['model_threshold']), (0.5, 1.0, 0.7))
        self.assertEqual(result['diagnosis'], 'Несоосность')
        self.assertTrue(result['synthetic'])


class RouteTest(unittest.TestCase):
    def serve(self, pumps):
        server = ThreadingHTTPServer(('127.0.0.1', 0), service.make_handler(None, None, pumps))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
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
        code, body = self.post(url, dict(pump_id='NPV_2_1', readings=readings(289)))
        self.assertEqual((code, body['status'], len(body['forecasts'])), (200, 'ok', 2))
        self.assertEqual(self.post(url, dict(pump_id='NPV_2_1', readings=[]))[0], 400)
        self.assertEqual(self.post(url, dict(readings=readings(2)))[0], 400)

    def test_absent_without_pump_model(self):
        self.assertEqual(self.post(self.serve(None), dict(pump_id='NPV_2_1', readings=readings(2)))[0], 404)


if __name__ == '__main__':
    unittest.main()
