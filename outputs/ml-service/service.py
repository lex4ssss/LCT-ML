import argparse
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import joblib
import numpy as np
from feature_store import FeatureStore, InsufficientData

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-baseline-v2'))
import experiment_v2 as ex

MAX_BODY = 64 * 1024


class Predictor:
    def __init__(self, store, decision_path):
        decision_path = Path(decision_path)
        self.decision = json.loads(decision_path.read_text())
        model_path = decision_path.parent / self.decision['model_path']
        if hashlib.file_digest(model_path.open('rb'), 'sha256').hexdigest() != self.decision['model_sha256']:
            raise ValueError('model file does not match decision.json')
        self.model = joblib.load(model_path)
        self.store, self.lock = store, threading.Lock()
        self.threshold = float(self.decision['threshold'])
        self.model_name = 'hgb-v2-run002-' + self.decision['model_sha256'][:8]

    def predict(self, target_id, as_of):
        channel = target_id.removeprefix('sensor_')
        with self.lock:
            t, values = self.store.features(channel, as_of)
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
                    model_name=self.model_name, status='ok', alert=alert, threshold=self.threshold, as_of_utc=t.isoformat(),
                    target_scope='recorded_alarm_episode_start', features=values)


def make_handler(predictor):
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
            self.reply(200, dict(status='ok', model_name=predictor.model_name, threshold=predictor.threshold, journal_last_record=predictor.store.global_last.isoformat()))

        def do_POST(self):
            if self.path != '/predict':
                return self.reply(404, dict(error='not_found'))
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_BODY:
                    raise ValueError('body size')
                payload = json.loads(self.rfile.read(length))
                target_id, as_of = payload['target_id'], datetime.fromisoformat(payload['as_of'])
                if not isinstance(target_id, str) or not target_id:
                    raise ValueError('target_id')
            except (ValueError, KeyError, TypeError) as error:
                return self.reply(400, dict(error='bad_request', detail=str(error)))
            try:
                self.reply(200, predictor.predict(target_id, as_of))
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
    args = parser.parse_args()
    predictor = Predictor(FeatureStore(args.prepared, args.config), args.decision)
    ThreadingHTTPServer((args.host, args.port), make_handler(predictor)).serve_forever()


if __name__ == '__main__':
    main()
