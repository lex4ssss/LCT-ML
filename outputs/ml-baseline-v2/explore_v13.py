import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
import adaptive_v13 as adaptive
import explore_v8 as e8
import models_v12 as metrics


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def search(cache, manifest, output):
    source = Path(__file__).parent
    previous = json.loads(manifest.read_text())
    if not previous['complete'] or any(entry['winner'] != 'R' for entry in previous['targets'].values()):
        raise ValueError('unexpected source experiment')
    for name, expected in previous['code'].items():
        if adaptive.file_digest(source / name) != expected:
            raise ValueError('feature implementation changed')
    output.mkdir(parents=True, exist_ok=True)
    code_files = ['adaptive_v13.py', 'explore_v13.py', 'spec_explore_v13.txt', 'exp_data.py', 'explore_v8.py', 'models_v12.py']
    fingerprint = dict(cache=adaptive.file_digest(cache), manifest=adaptive.file_digest(manifest),
                       code={name: adaptive.file_digest(source / name) for name in code_files})
    identity = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    report_path = output / 'search.json'
    report = dict(fingerprint=fingerprint, packages=previous['packages'], protocol='2025 chronological seven-day refits with horizon purge',
                  test_evaluation=False, service_changed=False, complete=False, targets={})
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['fingerprint'] != fingerprint:
            raise ValueError('resume fingerprint changed')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    shared = joblib.load(cache)
    if np.any(shared['days'] >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('test rows in cache')
    for target, hours in [('incident', 72), ('neispraven', 168)]:
        data = shared['targets'][target]
        matrix = np.column_stack([shared['base'], data['history']])
        _, evaluation = e8.masks(shared['days'], data['present'], hours)
        labels = data['labels'][evaluation]
        reference_scores = np.load(manifest.parent / f'{target}-R-scores.npy')
        own = report['targets'].setdefault(target, dict(horizon_hours=hours, results={}))
        for variant in adaptive.VARIANTS:
            if variant in own['results']:
                continue
            started = time.monotonic()
            if variant == 'R':
                scores = reference_scores
                records = []
                fixed_threshold = previous['targets'][target]['results']['R']['validation']['threshold']
            else:
                def callback(index, record):
                    if variant == 'W' and index == 0:
                        saved = joblib.load(output / f'{target}-{variant}' / '000.joblib')
                        positions = np.searchsorted(np.flatnonzero(evaluation), saved['indices'])
                        np.testing.assert_allclose(saved['scores'], reference_scores[positions], rtol=1e-9, atol=1e-9)
                        emit('initial_refit_matches_reference', target=target)
                    emit('block_complete', target=target, variant=variant, index=index, **record)
                scores, records = adaptive.score_blocks(matrix, data['labels'], shared['days'], data['present'], hours,
                    variant, output / f'{target}-{variant}', identity, on_block=callback)
                fixed_threshold = None
            validation = metrics.metrics(labels, scores, data['rule'][evaluation], data['strong_rule'][evaluation], fixed_threshold)
            if variant == 'R':
                old = previous['targets'][target]['results']['R']['validation']
                for name in ['rows', 'base_rate', 'precision', 'recall', 'f1', 'ap', 'threshold']:
                    if not np.isclose(validation[name], old[name], rtol=1e-12, atol=1e-12):
                        raise ValueError('reference result changed')
            known = shared['object_ids'][evaluation] != 'unknown'
            own['results'][variant] = dict(validation=validation, toggles=adaptive.toggles(variant),
                diagnostics=adaptive.diagnostics(labels, scores, shared['days'][evaluation], validation['threshold']),
                known_objects=metrics.metrics(labels[known], scores[known], data['rule'][evaluation][known],
                                               data['strong_rule'][evaluation][known], validation['threshold']),
                blocks=records, model_count=len(records) if variant != 'R' else 1,
                features=matrix.shape[1], elapsed_seconds=time.monotonic() - started)
            np.save(output / f'{target}-{variant}-scores.npy', scores)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            emit('variant_complete', target=target, variant=variant, toggles=adaptive.toggles(variant), validation=validation,
                 elapsed_seconds=own['results'][variant]['elapsed_seconds'])
        own['winner'] = adaptive.choose(own['results'])
        emit('target_selected', target=target, winner=own['winner'])
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cache', type=Path)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    search(args.cache, args.manifest, args.output)
