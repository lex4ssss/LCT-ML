import argparse
import hashlib
import json
from pathlib import Path


def build(run_dir):
    report = json.loads((run_dir / 'report.json').read_text())
    models = run_dir / 'pumps_v5_models.joblib'
    horizons = {}
    for name, minutes in (('30min', 30), ('1h', 60)):
        entry = report['horizons'][name]
        variant = entry['chosen']
        horizons[name] = dict(horizon_hours=minutes / 60, variant=variant, threshold=entry['validation']['variants'][variant]['threshold'],
                              test=entry['test']['model'], test_recall_by_fault_type=entry['test']['recall_by_fault_type'])
    decision = dict(spec=report['spec'], version='v5', synthetic=True, test=report['test'], model_path=models.name,
                    model_sha256=hashlib.sha256(models.read_bytes()).hexdigest(), max_readings=289, horizons=horizons,
                    router_test=report['test_router'])
    (run_dir / 'pump-decision.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    build(parser.parse_args().run)
