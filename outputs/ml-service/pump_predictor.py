import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-pumps'))
import pumps_data
import pumps_v5
import pumps_v5_features as v5

FAULT_ALERT_TEXT = {
    'Несоосность': 'несоосность: срок до отказа оценивается надёжно',
    'Засорение фильтра': 'засорение фильтра: срок до отказа оценивается надёжно',
    'Неизвестная неисправность': 'неизвестная неисправность: насос вне нормы, срок до отказа не определить, проверить вручную',
    'normal': 'насос в норме',
}


class BadReadings(ValueError):
    pass


class PumpPredictor:
    def __init__(self, decision_path):
        decision_path = Path(decision_path)
        self.decision = json.loads(decision_path.read_text())
        model_path = decision_path.parent / self.decision['model_path']
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.decision['model_sha256']:
            raise ValueError('pump model file does not match pump-decision.json')
        bundle = joblib.load(model_path)
        self.stats, self.router, self.models = bundle['stats'], bundle['router'], bundle['models']
        self.max_readings = int(self.decision['max_readings'])
        self.model_name = 'pumps-v5-' + self.decision['model_sha256'][:8]

    def features(self, pump_id, readings):
        if pump_id not in self.stats['median']:
            raise BadReadings(f'unknown pump_id {pump_id!r}')
        if not isinstance(readings, list) or not 0 < len(readings) <= self.max_readings:
            raise BadReadings(f'readings: 1 to {self.max_readings} rows, oldest first, 5 minutes apart')
        try:
            values = np.array([[float(row[name]) for name in pumps_data.SENSORS] for row in readings])
        except (KeyError, TypeError, ValueError) as error:
            raise BadReadings(f'every reading needs the 12 sensors: {error}') from error
        if not np.isfinite(values).all():
            raise BadReadings('sensor values must be finite numbers')
        frame = pd.DataFrame(values, columns=pumps_data.SENSORS)
        frame['pump_id'], frame['trajectory'] = pump_id, 0
        return v5.feature_frame(frame, self.stats).iloc[[-1]]

    def predict(self, pump_id, readings):
        features = self.features(pump_id, readings)
        diagnosis = str(self.router.predict(features)[0])
        forecasts = []
        for name, horizon in self.decision['horizons'].items():
            probability = float(pumps_v5.score_all(self.models[name], features)[horizon['variant']][0])
            forecasts.append(dict(horizon_hours=horizon['horizon_hours'], probability=probability, alert=probability >= horizon['threshold'],
                                  model_threshold=horizon['threshold']))
        return dict(status='ok', pump_id=pump_id, readings_used=len(readings), model_name=self.model_name, synthetic=True,
                    target_scope='pump_fault_within_horizon', diagnosis=diagnosis, diagnosis_text=FAULT_ALERT_TEXT[diagnosis],
                    forecasts=forecasts)
