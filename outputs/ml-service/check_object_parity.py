import json
from pathlib import Path
import random
import sys
import duckdb
import numpy as np
from feature_store import FeatureStore, InsufficientData
from object_predictor import ObjectPredictor
import eval_units as eu
import explore_v8 as e8
import run_v2 as rv

class NoChannelModel:
    pass


def main(prepared, config, examples, directory, episodes, decision, seed, days):
    rng = random.Random(int(seed))
    decided = json.loads(Path(decision).read_text())
    predictor = ObjectPredictor(FeatureStore(prepared, config, {decided['target']: decided['class_states']}), directory, decision, NoChannelModel())
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        codes, code, sensor, type_count = e8.channel_info(connection, directory, columns['channel_id'])
        keys, matrix = e8.aggregate(columns, codes, sensor, type_count)
        matrix = np.column_stack([matrix, e8.target_history(connection, episodes, directory, keys)])
    names = {index: name for name, index in code.items()}
    row_keys = eu.day_keys(codes, columns['as_of'])
    candidates = np.flatnonzero((keys // 1_000_000) != code[eu.UNKNOWN_OBJECT])
    same = differ = row_set = skipped = 0
    for position in rng.sample(list(candidates), int(days)):
        key = keys[position]
        obj, t = names[key // 1_000_000], e8.key_moments(np.array([key]))[0].astype('datetime64[us]').item()
        trained = set(columns['channel_id'][row_keys == key])
        try:
            _, features = predictor.store.features_many(predictor.objects[obj], t)
        except InsufficientData:
            skipped += 1
            continue
        if set(features) != trained:
            row_set += 1
            continue
        starts, seen = predictor.store.class_history(predictor.objects[obj], t, predictor.target)
        served, _, _ = predictor.object_row(t, features, [at for _, at in starts], bool(seen))
        if np.allclose(served, matrix[position], equal_nan=True, rtol=1e-9, atol=1e-9):
            same += 1
        else:
            differ += 1
            print(obj, t, np.flatnonzero(~np.isclose(served, matrix[position], equal_nan=True))[:10])
    print(dict(object_days=int(days), identical=same, different=differ, channel_set_differs=row_set, insufficient=skipped))
    return 1 if differ else 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:9]))
