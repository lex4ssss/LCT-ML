import argparse
import json
from pathlib import Path

import joblib
import numpy as np

import pumps_data
import pumps_v5
import pumps_v5_features as v5
from pumps_v1 import recall_by_fault_type, scores_at

STUCK_SENSORS = ['vibration_pump_bearing_drive', 'motor_current', 'pressure_diff_filter', 'pressure_discharge', 'temperature_motor_bearing_drive']


def stuck(reconstructed, sensor):
    frame = reconstructed.copy()
    normal = frame[frame['trajectory'] < 0].groupby('pump_id')[sensor].median()
    frame[sensor] = frame['pump_id'].map(normal).to_numpy()
    return frame


def evaluate(bundle, decision, reconstructed):
    features, labels, meta = v5.prediction_rows(reconstructed, bundle['stats'], pumps_v5.HORIZONS.values())
    result = {}
    for name, horizon in decision['horizons'].items():
        y = labels[pumps_v5.HORIZONS[name]].astype(np.int64)
        alarms = pumps_v5.score_all(bundle['models'][name], features)[horizon['variant']] >= horizon['threshold']
        result[name] = dict(**scores_at(y, alarms), recall_by_fault_type=recall_by_fault_type(y, alarms, meta['fault_type'].to_numpy()))
    return result


def run(run_dir, seeds_dir, fresh_seeds, dropout_seed):
    decision = json.loads((run_dir / 'pump-decision.json').read_text())
    bundle = joblib.load(run_dir / decision['model_path'])
    report = dict(fresh={}, stuck_sensor={})
    for seed in fresh_seeds:
        reconstructed = pumps_data.reconstruct(pumps_data.load(seeds_dir / f'seed-{seed}.csv'), allow_extra=True)
        report['fresh'][seed] = evaluate(bundle, decision, reconstructed)
        if seed == dropout_seed:
            for sensor in STUCK_SENSORS:
                report['stuck_sensor'][sensor] = evaluate(bundle, decision, stuck(reconstructed, sensor))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--seeds-dir', type=Path, required=True)
    parser.add_argument('--fresh', type=int, nargs='+', required=True)
    parser.add_argument('--dropout-seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(args.run, args.seeds_dir, args.fresh, args.dropout_seed), ensure_ascii=False, indent=2) + '\n')
