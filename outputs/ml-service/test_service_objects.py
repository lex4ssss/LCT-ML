from http.server import ThreadingHTTPServer
import json
import threading
import unittest
import urllib.error
import urllib.request
from feature_store import InsufficientData
import service


class FakeChannels:
    demo_offset, threshold, model_name = None, 0.5, 'fake'
    lock = threading.Lock()

    def predict(self, target_id, as_of):
        return dict(target_id=target_id)


class FakeObjects:
    hours, model_name, api_target = 72, 'fake-object', 'incident'
    objects = {'a': ['1'], 'b': ['2'], 'c': ['3'], 'd': ['4']}

    def predict_objects(self, object_ids, moment):
        if moment.year == 2000:
            raise InsufficientData('stale_journal', 'x')
        table = {'a': (0.5, 0.6), 'b': (0.9, 0.95), 'c': (0.5, 0.7)}
        results = {}
        for obj in object_ids:
            if obj in table:
                probability, score = table[obj]
                results[obj] = dict(status='ok', object_id=obj, as_of_utc=moment.isoformat(), probability=probability, score=score, alert=score >= 0.65)
            else:
                results[obj] = dict(status='insufficient_data', reason='no_eligible_channels', object_id=obj)
        return results


class FakeFailures(FakeObjects):
    hours, model_name, api_target = 168, 'fake-failure', 'failure'

    def predict_objects(self, object_ids, moment):
        return {obj: dict(status='ok', object_id=obj, as_of_utc=moment.isoformat(), probability=0.1, score=0.1, alert=False) for obj in object_ids}


class ServiceTests(unittest.TestCase):
    def start(self, objects):
        server = ThreadingHTTPServer(('127.0.0.1', 0), service.make_handler(FakeChannels(), objects))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f'http://127.0.0.1:{server.server_address[1]}'

    def call(self, base, path, body):
        request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(request) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def test_object_routes(self):
        base = self.start(dict(incident=FakeObjects(), failure=FakeFailures()))
        code, ranked = self.call(base, '/risk_map', {'as_of': '2025-01-01T03:00:00+03:00'})
        self.assertEqual(code, 200)
        self.assertEqual([row['object_id'] for row in ranked['objects']], ['b', 'c', 'a', 'd'])
        self.assertEqual((ranked['alerts'], ranked['horizon_hours'], ranked['as_of_utc'], ranked['target']), (2, 72, '2025-01-01T00:00:00', 'incident'))
        code, failures = self.call(base, '/risk_map', {'as_of': '2025-01-01', 'target': 'failure'})
        self.assertEqual((code, failures['target'], failures['horizon_hours'], failures['alerts']), (200, 'failure', 168, 0))
        self.assertEqual(self.call(base, '/predict_object', {'object_id': 'd', 'as_of': '2025-01-01', 'target': 'failure'})[0], 200)
        self.assertEqual(self.call(base, '/risk_map', {'as_of': '2025-01-01', 'target': 'weather'})[0], 400)
        self.assertEqual(self.call(base, '/risk_map', {'as_of': '2025-01-01', 'target': ['incident']})[0], 400)
        self.assertEqual(self.call(base, '/predict_object', {'object_id': 'c', 'as_of': '2025-01-01'})[1]['score'], 0.7)
        self.assertEqual(self.call(base, '/predict_object', {'object_id': 'd', 'as_of': '2025-01-01'})[0], 422)
        self.assertEqual(self.call(base, '/predict_object', {'object_id': 'zz', 'as_of': '2025-01-01'})[0], 404)
        self.assertEqual(self.call(base, '/predict_object', {'object_id': '', 'as_of': '2025-01-01'})[0], 400)
        self.assertEqual(self.call(base, '/predict_object', {'object_id': 'a'})[0], 400)
        self.assertEqual(self.call(base, '/risk_map', {'as_of': '2000-01-01'}), (422, dict(status='insufficient_data', reason='stale_journal', detail='x')))
        self.assertEqual(self.call(base, '/predict', {'target_id': 'sensor_1', 'as_of': '2025-01-01'}), (200, dict(target_id='sensor_1')))

    def test_object_routes_absent_without_object_model(self):
        base = self.start(None)
        self.assertEqual(self.call(base, '/risk_map', {'as_of': '2025-01-01'})[0], 404)
        self.assertEqual(self.call(base, '/predict_object', {'object_id': 'a', 'as_of': '2025-01-01'})[0], 404)
        self.assertEqual(self.call(base, '/predict', {'target_id': 'sensor_1', 'as_of': '2025-01-01'})[0], 200)


if __name__ == '__main__':
    unittest.main()
