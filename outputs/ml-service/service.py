import argparse
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import joblib
import numpy as np
from feature_store import FeatureStore, InsufficientData, to_utc_naive

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-baseline-v2'))
import experiment_v2 as ex
from object_predictor import ObjectPredictor

MAX_BODY = 64 * 1024


class Predictor:
    def __init__(self, store, decision_path, demo_anchor=None):
        decision_path = Path(decision_path)
        self.decision = json.loads(decision_path.read_text())
        model_path = decision_path.parent / self.decision['model_path']
        if hashlib.file_digest(model_path.open('rb'), 'sha256').hexdigest() != self.decision['model_sha256']:
            raise ValueError('model file does not match decision.json')
        self.model = joblib.load(model_path)
        self.store, self.lock = store, threading.Lock()
        self.threshold = float(self.decision['threshold'])
        self.model_name = 'hgb-v2-run002-' + self.decision['model_sha256'][:8]
        self.demo_anchor = None if demo_anchor is None else to_utc_naive(demo_anchor)
        self.demo_offset = None if demo_anchor is None else self.demo_anchor - datetime.now(timezone.utc).replace(tzinfo=None)

    def predict(self, target_id, as_of):
        channel = target_id.removeprefix('sensor_')
        requested = to_utc_naive(as_of)
        effective = requested if self.demo_offset is None else requested + self.demo_offset
        with self.lock:
            t, values = self.store.features(channel, effective)
        columns = {name: np.array([np.nan if values[name] is None else values[name]], dtype=np.float64) for name in ex.V2_FEATURES}
        columns['as_of'] = np.array([np.datetime64(t, 'us')])
        probability = float(self.model.predict_proba(ex.matrix_v2(columns, True))[0, 1])
        alert = probability >= self.threshold
        alarm_age = values['last_alarm_age_hours_30d']
        factors = [f"тревог за 24 ч: {values['alarm_count_24h']}, за 7 сут: {values['alarm_count_7d']}, за 30 сут: {values['alarm_count_30d']}",
                   f"начал тревожных эпизодов за 30 сут: {values['episode_starts_30d']}",
                   'тревог за 30 сут не было' if alarm_age is None else f'последняя тревога {alarm_age:.1f} ч назад',
                   f"активных суток за 30: {values['active_days_30d']}"]
        recommendation = ('Риск нового тревожного эпизода в ближайшие 24 ч выше рабочего порога модели: проверить канал и объект' if alert
                          else 'Риск ниже рабочего порога модели: плановое наблюдение')
        return dict(probability=probability, lead_min_hours=0.0, prediction_window_hours=24.0, top_factors=factors, recommendation=recommendation,
                    model_name=self.model_name, status='ok', alert=alert, threshold=self.threshold, model_threshold=self.threshold,
                    horizon_hours=24.0, as_of_utc=t.isoformat(),
                    target_scope='recorded_alarm_episode_start', features=values,
                    demo_clock=None if self.demo_offset is None else dict(requested_as_of_utc=requested.isoformat(), anchor_utc=self.demo_anchor.isoformat()))


def make_handler(predictor, objects=None):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != '/health':
                return self.reply(404, dict(error='not_found'))
            demo = None if predictor.demo_offset is None else (datetime.now(timezone.utc).replace(tzinfo=None) + predictor.demo_offset).isoformat()
            self.reply(200, dict(status='ok', model_name=predictor.model_name, threshold=predictor.threshold, journal_last_record=predictor.store.global_last.isoformat(), demo_now_utc=demo,
                                 object_models={name: model.model_name for name, model in (objects or {}).items()}))

        def do_POST(self):
            if self.path not in ('/predict', '/predict_object', '/risk_map') or (self.path != '/predict' and not objects):
                return self.reply(404, dict(error='not_found'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    raise ValueError('body size')
                payload = json.loads(self.rfile.read(length))
                as_of = datetime.fromisoformat(payload['as_of'])
                key = {'/predict': 'target_id', '/predict_object': 'object_id'}.get(self.path)
                if key is not None and (not isinstance(payload[key], str) or not payload[key]):
                    raise ValueError(key)
                if self.path != '/predict':
                    model = objects[payload.get('target', 'incident')]
            except (ValueError, KeyError, TypeError) as error:
                return self.reply(400, dict(error='bad_request', detail=str(error)))
            try:
                if self.path == '/predict':
                    return self.reply(200, predictor.predict(payload['target_id'], as_of))
                if self.path == '/predict_object' and payload['object_id'] not in model.objects:
                    return self.reply(404, dict(error='unknown_object', object_id=payload['object_id']))
                wanted = [payload['object_id']] if self.path == '/predict_object' else sorted(model.objects)
                requested = to_utc_naive(as_of)
                with predictor.lock:
                    results = model.predict_objects(wanted, requested if predictor.demo_offset is None else requested + predictor.demo_offset)
                if self.path == '/predict_object':
                    result = results[payload['object_id']]
                    return self.reply(200 if result['status'] == 'ok' else 422, result)
                ranked = sorted(results.values(), key=lambda row: (-row.get('probability', -1.0), -row.get('score', -1.0)))
                self.reply(200, dict(as_of_utc=next((row['as_of_utc'] for row in ranked if 'as_of_utc' in row), None), target=model.api_target, horizon_hours=model.hours,
                                     alerts=sum(row.get('alert', False) for row in ranked), objects=ranked))
            except InsufficientData as error:
                self.reply(422, dict(status='insufficient_data', reason=error.reason, detail=error.detail))

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--decision', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8090)
    parser.add_argument('--demo-anchor', type=datetime.fromisoformat)
    parser.add_argument('--object-decision', action='append', default=[])
    parser.add_argument('--directory')
    args = parser.parse_args()
    if bool(args.object_decision) != (args.directory is not None):
        parser.error('--object-decision and --directory go together')
    decisions = [json.loads(Path(path).read_text()) for path in args.object_decision]
    states = {decision['target']: decision['class_states'] for decision in decisions}
    predictor = Predictor(FeatureStore(args.prepared, args.config, states), args.decision, args.demo_anchor)
    objects = {}
    for path in args.object_decision:
        model = ObjectPredictor(predictor.store, args.directory, path, predictor.model)
        if model.api_target in objects:
            parser.error('two object models for ' + model.api_target)
        objects[model.api_target] = model
    ThreadingHTTPServer((args.host, args.port), make_handler(predictor, objects)).serve_forever()


if __name__ == '__main__':
    main()
