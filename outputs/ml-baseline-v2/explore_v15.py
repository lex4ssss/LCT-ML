import argparse
import json
from pathlib import Path
import time
import duckdb
import joblib
import numpy as np
import adaptive_v13 as adaptive
import explore_v8 as e8
import models_v12 as models
import states_v15 as states
import telemetry_v12 as telemetry


VARIANTS = ('R', 'H_S', 'H_SQ', 'C_SQ')


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def toggles(variant):
    return dict(state_history=variant != 'R', state_context=variant.endswith('SQ'), object_identity=variant == 'C_SQ',
                retraining=False, recency_weights=False, class_balancing=False, numeric=False, domains=False,
                weather=False, synthetic=False, neural_network=False, calibration=False, per_state_thresholds=False)


def search(cache, manifest, directory, output):
    source = Path(__file__).parent
    previous = json.loads(manifest.read_text())
    if not previous['complete']:
        raise ValueError('source experiment incomplete')
    for name, expected in previous['code'].items():
        if adaptive.file_digest(source / name) != expected:
            raise ValueError('source implementation changed')
    observations = [Path(name) for name in previous['inputs'] if Path(name).stem.isdigit() and int(Path(name).stem) < 2026]
    for path in observations + [directory]:
        if adaptive.file_digest(path) != previous['inputs'][str(path)]:
            raise ValueError('observation or registry input changed')
    fingerprint = dict(cache=adaptive.file_digest(cache), manifest=adaptive.file_digest(manifest),
                       code={name: adaptive.file_digest(source / name) for name in
                             ['states_v15.py', 'explore_v15.py', 'spec_explore_v15.txt', 'exp_data.py', 'models_v12.py', 'explore_v8.py', 'adaptive_v13.py']},
                       state_source=adaptive.file_digest(source.parent / 'ml-dataset-v2' / 'class_labels.py'))
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'search.json'
    report = dict(fingerprint=fingerprint, packages=previous['packages'], complete=False, test_evaluation=False,
                  service_changed=False, states=list(states.STATES), targets={})
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['fingerprint'] != fingerprint:
            raise ValueError('resume fingerprint changed')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    shared = joblib.load(cache)
    if np.any(shared['days'] >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('test rows in cache')
    feature_path = output / 'state_features.joblib'
    if feature_path.exists():
        built = joblib.load(feature_path)
    else:
        with duckdb.connect(config={'threads': 4, 'memory_limit': '2GB'}) as connection:
            daily = states.daily_states(connection, observations, telemetry.registry(directory), shared['days'].max())
            primary, context, names = states.state_matrix(daily, shared['object_ids'], shared['days'])
        built = dict(primary=primary, context=context, names=names)
        joblib.dump(built, feature_path)
        emit('state_features_ready', daily_rows=len(daily), primary_features=primary.shape[1], context_features=context.shape[1])
    for target, hours in [('incident', 72), ('neispraven', 168)]:
        data = shared['targets'][target]
        base = np.column_stack([shared['base'], data['history']])
        train, evaluation = e8.masks(shared['days'], data['present'], hours)
        own = report['targets'].setdefault(target, dict(horizon_hours=hours, results={}))
        for variant in VARIANTS:
            if variant in own['results']:
                continue
            started = time.monotonic()
            flags = toggles(variant)
            parts = [base]
            if flags['state_history']:
                parts.append(built['primary'])
            if flags['state_context']:
                parts.append(built['context'])
            matrix = np.column_stack(parts) if len(parts) > 1 else base
            fixed_threshold = None
            if variant == 'R':
                scores = np.load(manifest.parent / f'{target}-R-scores.npy')
                fixed_threshold = previous['targets'][target]['results']['R']['validation']['threshold']
            else:
                model = models.fit_model(variant, matrix[train], data['labels'][train], shared['object_ids'][train], shared['categories'], shared['days'][train])
                scores = models.predict_model(variant, model, matrix[evaluation], shared['object_ids'][evaluation], shared['categories'])
                models.save_verified(model, output / f'{target}-{variant}.joblib', variant, matrix[evaluation][:37], shared['object_ids'][evaluation][:37], shared['categories'])
            validation = models.metrics(data['labels'][evaluation], scores, data['rule'][evaluation], data['strong_rule'][evaluation], fixed_threshold)
            if variant == 'R':
                old = previous['targets'][target]['results']['R']['validation']
                for name in ['rows', 'base_rate', 'precision', 'recall', 'f1', 'ap', 'threshold']:
                    if not np.isclose(validation[name], old[name], rtol=1e-12, atol=1e-12):
                        raise ValueError('reference result changed')
            known = shared['object_ids'][evaluation] != 'unknown'
            own['results'][variant] = dict(validation=validation, toggles=flags, features=matrix.shape[1] + int(flags['object_identity']),
                known_objects=models.metrics(data['labels'][evaluation][known], scores[known], data['rule'][evaluation][known], data['strong_rule'][evaluation][known], validation['threshold']),
                diagnostics=adaptive.diagnostics(data['labels'][evaluation], scores, shared['days'][evaluation], validation['threshold']),
                elapsed_seconds=time.monotonic() - started)
            np.save(output / f'{target}-{variant}-scores.npy', scores)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            emit('variant_complete', target=target, variant=variant, toggles=flags, validation=validation)
        own['winner'] = models.winner(own['results'])
        emit('target_selected', target=target, winner=own['winner'])
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['cache', 'manifest', 'directory', 'output']:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    search(args.cache, args.manifest, args.directory, args.output)
