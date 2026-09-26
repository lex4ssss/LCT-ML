import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-pumps'))
import pumps_data
import pumps_v2

MAX_READINGS = 97


class BadReadings(ValueError):
    pass


class PumpPredictor:
    def __init__(self, decision_path):
        decision_path = Path(decision_path)
        self.decision = json.loads(decision_path.read_text())
        model_path = decision_path.parent / self.decision['model_path']
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != self.decision['model_sha256']:
            raise ValueError('pump model file does not match pump-decision.json')
        self.models = joblib.load(model_path)
        self.model_name = 'pumps-v4-' + self.decision['model_sha256'][:8]
        self.profile = {pump: np.array([values[name] for name in pumps_data.SENSORS]) for pump, values in self.decision['normal_profile'].items()}

    def features(self, pump_id, readings):
        if pump_id not in self.profile:
            raise BadReadings(f'unknown pump_id {pump_id!r}')
        if not isinstance(readings, list) or not 0 < len(readings) <= MAX_READINGS:
            raise BadReadings(f'readings: 1 to {MAX_READINGS} rows, oldest first, 5 minutes apart')
        try:
            values = np.array([[float(row[name]) for name in pumps_data.SENSORS] for row in readings])
        except (KeyError, TypeError, ValueError) as error:
            raise BadReadings(f'every reading needs the 12 sensors: {error}') from error
        if not np.isfinite(values).all():
            raise BadReadings('sensor values must be finite numbers')
        frame = pd.DataFrame(values, columns=pumps_data.SENSORS)
        frame['pump_id'], frame['trajectory'] = pump_id, 0
        in_trajectory = np.ones(len(frame), dtype=bool)
        profile = np.tile(self.profile[pump_id], (len(frame), 1))
        v3 = pumps_data.feature_frame_v3(frame, in_trajectory, profile).iloc[[-1]]
        a3 = pumps_data.feature_frame_a3(frame, in_trajectory, profile=profile)[0].iloc[[-1]]
        return dict(a3=a3, v3=v3)

    def predict(self, pump_id, readings):
        features = self.features(pump_id, readings)
        forecasts = []
        for name, horizon in self.decision['horizons'].items():
            probability = float(pumps_v2.predict(self.models[name], features, horizon['variant'])[0])
            forecasts.append(dict(horizon_hours=horizon['horizon_hours'], probability=probability, alert=probability >= horizon['threshold'],
                                  model_threshold=horizon['threshold']))
        return dict(status='ok', pump_id=pump_id, readings_used=len(readings), model_name=self.model_name, synthetic=True,
                    target_scope='pump_fault_within_horizon', forecasts=forecasts)
