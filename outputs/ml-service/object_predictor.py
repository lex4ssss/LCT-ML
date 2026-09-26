import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-baseline-v2'))
import eval_units as eu
import experiment_v2 as ex
import explore_v8 as e8

RU = {'event_count_24h': 'событий за 24 ч', 'alarm_count_24h': 'тревог за 24 ч', 'event_count_7d': 'событий за 7 сут', 'alarm_count_7d': 'тревог за 7 сут',
      'event_count_30d': 'событий за 30 сут', 'alarm_count_30d': 'тревог за 30 сут', 'active_days_30d': 'активных суток из 30',
      'episode_starts_7d': 'начал тревожных эпизодов за 7 сут', 'episode_starts_30d': 'начал тревожных эпизодов за 30 сут',
      'last_event_age_hours': 'часов с последнего события', 'last_alarm_age_hours_30d': 'часов с последней тревоги',
      'channel_age_days': 'возраст канала, сут', 'log_gap_hours_30d': 'часов пауз журнала за 30 сут'}
TOP_FACTORS = 3
TOP_CHANNELS = 5


def feature_texts(type_names):
    texts = ['каналов с данными в объекте: {}']
    texts += [f'{RU[name]}, сумма по каналам объекта: {{}}' for name in e8.SUMS]
    texts += [f'{RU[name]}, максимум на одном канале: {{}}' for name in e8.MAXIMA]
    texts += [f'{RU[name]}, минимум по каналам: {{}}' for name in e8.MINIMA]
    texts += [f'каналов, где {RU[name]} больше нуля: {{}}' for name in e8.ACTIVE]
    texts += [f'доля каналов, где {RU[name]} больше нуля: {{}}' for name in e8.ACTIVE]
    texts += [f'каналов типа «{name}»: {{}}' for name in type_names]
    texts += [f'начал тревожных эпизодов за 7 сут на датчиках «{name}»: {{}}' for name in type_names]
    texts += [f'ведущий канал объекта, {RU[name]}: {{}}' for name in ex.V2_FEATURES]
    texts += ['месяц: {}', 'день недели: {}']
    return texts + ['начал инцидентов в объекте за 24 ч: {}', 'начал инцидентов в объекте за 7 сут: {}', 'начал инцидентов в объекте за 30 сут: {}',
                    'часов с последнего начала инцидента в объекте (не больше 720): {}']


def class_history_row(t, starts, seen):
    moments = np.array([start for start in starts], dtype='datetime64[us]')
    now = np.datetime64(t, 'us')
    counts = [float(((moments > now - hours * np.timedelta64(3600, 's')) & (moments <= now)).sum()) for hours in e8.WINDOWS_HOURS]
    if len(moments):
        age = min(float((now - moments.max()) / np.timedelta64(3600, 's')), e8.AGE_CAP_HOURS)
    else:
        age = float(e8.AGE_CAP_HOURS) if seen else np.nan
    return np.array(counts + [age])


def verified(path, digest):
    if hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest() != digest:
        raise ValueError('file does not match object decision: ' + str(path))
    return joblib.load(path)


class ObjectPredictor:
    def __init__(self, store, directory, decision_path, channel_model):
        decision_path = Path(decision_path)
        self.decision = json.loads(decision_path.read_text())
        self.model = verified(decision_path.parent / self.decision['model_path'], self.decision['model_sha256'])
        self.calibrator = verified(decision_path.parent / self.decision['calibration_path'], self.decision['calibration_sha256'])
        self.threshold, self.hours = float(self.decision['threshold']), int(self.decision['horizon_hours'])
        self.store, self.channel_model, self.explainer = store, channel_model, shap.TreeExplainer(self.model)
        with duckdb.connect() as connection:
            rows = connection.execute('SELECT ид_канала_данных, ид_объект, тип_датчика FROM read_csv(?, all_varchar=true)', [str(directory)]).fetchall()
        type_names = sorted({row[2] for row in rows})
        self.type_index, self.type_count = {name: index for index, name in enumerate(type_names)}, len(type_names)
        self.sensor_type = {row[0]: row[2] for row in rows}
        self.objects = {}
        for channel, obj, _ in rows:
            self.objects.setdefault(obj, []).append(channel)
        self.texts = feature_texts(type_names)
        if len(self.texts) != self.model.n_features_in_:
            raise ValueError('feature names do not match the object model')
        self.model_name = 'hgb-object-run008-' + self.decision['model_sha256'][:8]

    def object_row(self, t, features, starts, seen):
        channels = sorted(features)
        columns = {name: np.array([np.nan if features[c][name] is None else features[c][name] for c in channels], dtype=np.float64) for name in ex.V2_FEATURES}
        columns['as_of'] = np.full(len(channels), np.datetime64(t, 'us'))
        sensor = np.array([self.type_index.get(self.sensor_type.get(c), -1) for c in channels], dtype=np.int64)
        _, matrix = e8.aggregate(columns, np.zeros(len(channels), dtype=np.int64), sensor, self.type_count)
        return np.r_[matrix[0], class_history_row(t, starts, seen)], columns, channels

    def factors(self, row):
        contributions = self.explainer.shap_values(row[None, :])[0]
        order = [index for index in np.argsort(-contributions) if contributions[index] > 0][:TOP_FACTORS]
        return [dict(text=self.texts[index].format('нет данных' if np.isnan(row[index]) else f'{row[index]:.3g}'), contribution_log_odds=float(contributions[index]))
                for index in order]

    def suspects(self, columns, channels):
        scores = self.channel_model.predict_proba(ex.matrix_v2(columns, True))[:, 1]
        order = np.argsort(-scores)[:TOP_CHANNELS]
        return [dict(target_id='sensor_' + channels[i], sensor_type=self.sensor_type.get(channels[i]), channel_score_24h=float(scores[i])) for i in order]

    def predict_objects(self, object_ids, moment):
        channels = [channel for obj in object_ids for channel in self.objects[obj]]
        t, features = self.store.features_many(channels, moment)
        starts, seen = self.store.class_history(channels, t)
        results = {}
        for obj in object_ids:
            mine = set(self.objects[obj])
            own = {channel: values for channel, values in features.items() if channel in mine}
            if not own:
                results[obj] = dict(status='insufficient_data', reason='no_eligible_channels', object_id=obj)
                continue
            row, columns, used = self.object_row(t, own, [at for channel, at in starts if channel in mine], bool(seen & mine))
            score = float(self.model.predict_proba(row[None, :])[0, 1])
            results[obj] = dict(status='ok', object_id=obj, as_of_utc=t.isoformat(), horizon_hours=self.hours, target_scope='incident_episode_start_in_object',
                                probability=float(self.calibrator.predict([score])[0]), score=score, threshold=self.threshold, alert=score >= self.threshold,
                                channels_used=len(used), channels_total=len(mine), top_factors=self.factors(row), suspect_channels=self.suspects(columns, used),
                                model_name=self.model_name)
        return results
