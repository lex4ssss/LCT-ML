import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import pumps_data


def build(primary, run_dir):
    report = json.loads((run_dir / 'report.json').read_text())
    reconstructed = pumps_data.reconstruct(pumps_data.load(primary), allow_extra=True)
    profile = pumps_data.normal_profile(reconstructed)
    pumps = reconstructed['pump_id'].to_numpy()
    profiles = {pump: dict(zip(pumps_data.SENSORS, map(float, profile[np.flatnonzero(pumps == pump)[0]]))) for pump in pumps_data.PUMPS}
    horizons = {}
    for name, minutes in (('30min', 30), ('1h', 60)):
        entry = report['horizons'][name]
        variant = entry['chosen']
        horizons[name] = dict(horizon_hours=minutes / 60, variant=variant, threshold=entry['validation']['variants'][variant]['threshold'],
                              test=entry['test']['model'])
    models = run_dir / 'pumps_v2_models.joblib'
    decision = dict(spec=report['spec'], threshold_rule=report['threshold_rule'], data=primary.name, synthetic=True,
                    model_path=models.name, model_sha256=hashlib.sha256(models.read_bytes()).hexdigest(),
                    normal_profile=profiles, horizons=horizons)
    (run_dir / 'pump-decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    build(args.primary, args.run)
