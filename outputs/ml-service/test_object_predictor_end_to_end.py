from datetime import timedelta
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import object_predictor as op
from feature_store import FeatureStore
from test_feature_store_many import START, STATES, build_journal

TYPES = ['КД Дверь', 'Датчик дыма', 'ИБП']


class ChannelModel:
    def predict_proba(self, matrix):
        score = np.tanh(np.nan_to_num(matrix[:, 1]) / 5)
        return np.column_stack([1 - score, score])


def write_models(root, target, states, width):
    rng = np.random.default_rng(1)
    features, labels = rng.random((400, width)) * 10, rng.integers(0, 2, 400)
    joblib.dump(HistGradientBoostingClassifier(max_iter=20).fit(features, labels), root / f'{target}.joblib')
    joblib.dump(IsotonicRegression(out_of_bounds='clip').fit([0.0, 1.0], [0.0, 0.5]), root / f'{target}-iso.joblib')
    sha = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    decision = dict(run='test', api_target='failure' if target == 'neispraven' else target, target=target, class_states=states, horizon_hours=72, threshold=0.5,
                    model_path=f'{target}.joblib', model_sha256=sha(f'{target}.joblib'), calibration_path=f'{target}-iso.joblib', calibration_sha256=sha(f'{target}-iso.joblib'))
    (root / f'{target}.json').write_text(json.dumps(decision, ensure_ascii=False))
    return root / f'{target}.json'


class EndToEndTests(unittest.TestCase):
    def test_predict_objects_on_synthetic_journal(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, channels, config = build_journal(root, random.Random(9))
            directory = root / 'directory.csv'
            directory.write_text('"ид_канала_данных","тип_датчика","ид_объект"\n' + ''.join(f'{c},{TYPES[i % 3]},{"o1" if i % 2 else "o2"}\n' for i, c in enumerate(channels)))
            width = len(op.feature_texts(TYPES))
            classes = dict(incident=STATES, neispraven=['Неисправен'])
            store = FeatureStore(root, config, classes)
            t = START + timedelta(days=60)
            for target, states in classes.items():
                predictor = op.ObjectPredictor(store, directory, write_models(root, target, states, width), ChannelModel())
                self.assertEqual(predictor.texts, op.feature_texts(sorted(TYPES), target))
                results = predictor.predict_objects(['o1', 'o2'], t)
                _, eligible = store.features_many(channels, t)
                for obj in ('o1', 'o2'):
                    row = results[obj]
                    mine = [c for c in predictor.objects[obj] if c in eligible]
                    self.assertEqual((row['status'], row['channels_used'], row['channels_total']), ('ok', len(mine), len(predictor.objects[obj])))
                    self.assertAlmostEqual(row['probability'], row['score'] / 2, places=9)
                    self.assertEqual(row['alert'], row['score'] >= 0.5)
                    self.assertEqual(row['target_scope'], f'{target}_episode_start_in_object')
                    self.assertTrue(set(s['target_id'].removeprefix('sensor_') for s in row['suspect_channels']) <= set(mine))
            store.connection.close()


if __name__ == '__main__':
    unittest.main()
