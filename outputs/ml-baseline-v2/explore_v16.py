import argparse
import json
from importlib.metadata import version as package_version
import os
from pathlib import Path
import time
import duckdb
import joblib
import numpy as np
import adaptive_v13 as adaptive
import channel_features_v16 as channels
import explore_v8 as e8
import models_v12 as models
import telemetry_v12 as telemetry


VARIANTS = ('R', 'H_T', 'H_N', 'H_TN', 'C_TN', 'E_TN')


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def toggles(variant):
    if variant not in VARIANTS:
        raise ValueError('variant')
    return dict(channel_trajectories=variant in ('H_T', 'H_TN', 'C_TN', 'E_TN'),
                channel_numeric=variant in ('H_N', 'H_TN', 'C_TN', 'E_TN'), object_identity=variant in ('C_TN', 'E_TN'),
                equal_ensemble=variant == 'E_TN', retraining=False, recency_weights=False, class_balancing=False,
                object_numeric=False, domains=False, object_state_histories=False, weather=False, synthetic=False,
                neural_network=False, calibration=False, per_channel_thresholds=False, per_object_thresholds=False)


def fingerprint(source, cache, manifest, previous):
    if not previous['complete']:
        raise ValueError('source experiment incomplete')
    for name, expected in previous['code'].items():
        if adaptive.file_digest(source / name) != expected:
            raise ValueError('source implementation changed')
    for name, expected in previous['inputs'].items():
        if adaptive.file_digest(Path(name)) != expected:
            raise ValueError('source input changed')
    return dict(cache=adaptive.file_digest(cache), manifest=adaptive.file_digest(manifest),
                code={name: adaptive.file_digest(source / name) for name in
                      ('channel_features_v16.py', 'explore_v16.py', 'spec_explore_v16.txt', 'states_v15.py',
                       'exp_data.py', 'experiment_v2.py', 'exp_metrics_core.py', 'explore_v7.py', 'models_v12.py',
                       'explore_v8.py', 'adaptive_v13.py', 'telemetry_v12.py')},
                state_source=adaptive.file_digest(source.parent / 'ml-dataset-v2' / 'class_labels.py'),
                reference_scores={target: adaptive.file_digest(manifest.parent / f'{target}-R-scores.npy')
                                  for target in ('incident', 'neispraven')})


def restore_report(path, identity, packages):
    report = dict(fingerprint=identity, packages=packages, complete=False, test_evaluation=False,
                  service_changed=False, observation_resolution='daily', feature_names=channels.feature_names(), targets={})
    if path.exists():
        report = json.loads(path.read_text())
        if report['fingerprint'] != identity:
            raise ValueError('resume fingerprint changed')
    return report


def select(results):
    reference = results['R']['validation']
    eligible = [(name, row) for name, row in results.items() if name != 'R' and models.accepted(row['validation'], reference)]
    if not eligible:
        return 'R'
    return max(eligible, key=lambda item: (item[1]['validation']['ap'], item[1]['validation']['margin'],
                                        -item[1]['features'], -item[1]['model_count']))[0]


def verify_scores(results, output, target):
    for variant, row in results.items():
        if adaptive.file_digest(output / f'{target}-{variant}-scores.npy') != row['score_digest']:
            raise ValueError('saved scores changed')


def search(cache, manifest, directory, output):
    source = Path(__file__).parent
    previous = json.loads(manifest.read_text())
    packages = {name: package_version(name) for name in previous['packages']}
    if packages != previous['packages']:
        raise ValueError('runtime package versions changed')
    threads = {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}
    if threads != dict(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='4'):
        raise ValueError('preregistered thread limits required')
    if str(directory) not in previous['inputs']:
        raise ValueError('registry absent from source manifest')
    identity = fingerprint(source, cache, manifest, previous)
    observations = [Path(name) for name in previous['inputs'] if Path(name).stem.isdigit() and int(Path(name).stem) < 2026]
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'search.json'
    report = restore_report(report_path, identity, packages)
    report['threads'] = threads
    parameters = models.ex.make_model('hgb').get_params()
    report['hgb_parameters'] = {name: parameters[name] for name in ('max_iter', 'learning_rate', 'max_leaf_nodes', 'min_samples_leaf', 'l2_regularization', 'early_stopping', 'random_state')}
    report['catboost_parameters'] = dict(iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=10,
                                        loss_function='Logloss', random_seed=0, thread_count=4, has_time=True,
                                        allow_writing_files=False, early_stopping=False)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    shared = joblib.load(cache)
    if np.any(shared['days'] >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('test rows in cache')
    feature_path = output / 'channel_features.joblib'
    if feature_path.exists() and 'feature_digest' in report:
        if adaptive.file_digest(feature_path) != report['feature_digest']:
            raise ValueError('feature cache changed')
        built = joblib.load(feature_path)
    else:
        emit('daily_channels_started', years=[path.stem for path in observations])
        with duckdb.connect(config={'threads': 4, 'memory_limit': '2GB'}) as connection:
            daily = channels.daily_channels(connection, observations, telemetry.registry(directory), shared['days'].max())
        emit('daily_channels_ready', rows=len(daily), channels=int(daily['channel_id'].nunique()))
        trajectories = channels.trajectory_matrix(daily, shared['object_ids'], shared['days'])
        emit('trajectories_ready', features=trajectories.shape[1])
        numeric = channels.numeric_matrix(daily, shared['object_ids'], shared['days'])
        built = dict(trajectories=trajectories, numeric=numeric, names=channels.feature_names())
        joblib.dump(built, feature_path)
        report['feature_digest'] = adaptive.file_digest(feature_path)
        report['daily_rows'] = len(daily)
        report['channel_count'] = int(daily['channel_id'].nunique())
        del daily
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        emit('channel_features_ready', numeric_features=numeric.shape[1])
    for target, hours in (('incident', 72), ('neispraven', 168)):
        data = shared['targets'][target]
        base = np.column_stack([shared['base'], data['history']])
        train, evaluation = e8.masks(shared['days'], data['present'], hours)
        own = report['targets'].setdefault(target, dict(horizon_hours=hours, results={}))
        verify_scores(own['results'], output, target)
        for variant in VARIANTS:
            if variant in own['results']:
                continue
            started = time.monotonic()
            flags = toggles(variant)
            parts = [base]
            if flags['channel_trajectories']:
                parts.append(built['trajectories'])
            if flags['channel_numeric']:
                parts.append(built['numeric'])
            matrix = np.column_stack(parts) if len(parts) > 1 else base
            fixed_threshold = None
            if variant == 'R':
                scores = np.load(manifest.parent / f'{target}-R-scores.npy')
                fixed_threshold = previous['targets'][target]['results']['R']['validation']['threshold']
            elif variant == 'E_TN':
                scores = (np.load(output / f'{target}-H_TN-scores.npy') + np.load(output / f'{target}-C_TN-scores.npy')) / 2
            else:
                emit('training_started', target=target, variant=variant, toggles=flags, train_rows=int(train.sum()), features=matrix.shape[1])
                model = models.fit_model(variant, matrix[train], data['labels'][train], shared['object_ids'][train], shared['categories'], shared['days'][train])
                scores = models.predict_model(variant, model, matrix[evaluation], shared['object_ids'][evaluation], shared['categories'])
                models.save_verified(model, output / f'{target}-{variant}.joblib', variant, matrix[evaluation][:37], shared['object_ids'][evaluation][:37], shared['categories'])
                del model
            validation = models.metrics(data['labels'][evaluation], scores, data['rule'][evaluation], data['strong_rule'][evaluation], fixed_threshold)
            if variant == 'R':
                old = previous['targets'][target]['results']['R']['validation']
                for name in ('rows', 'base_rate', 'precision', 'recall', 'f1', 'ap', 'threshold'):
                    if not np.isclose(validation[name], old[name], rtol=1e-12, atol=1e-12):
                        raise ValueError('reference result changed')
            known = shared['object_ids'][evaluation] != 'unknown'
            own['results'][variant] = dict(validation=validation, toggles=flags,
                features=matrix.shape[1] + int(flags['object_identity']), model_count=2 if variant == 'E_TN' else 1,
                known_objects=models.metrics(data['labels'][evaluation][known], scores[known], data['rule'][evaluation][known], data['strong_rule'][evaluation][known], validation['threshold']),
                diagnostics=adaptive.diagnostics(data['labels'][evaluation], scores, shared['days'][evaluation], validation['threshold']),
                elapsed_seconds=time.monotonic() - started)
            np.save(output / f'{target}-{variant}-scores.npy', scores)
            own['results'][variant]['score_digest'] = adaptive.file_digest(output / f'{target}-{variant}-scores.npy')
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            emit('variant_complete', target=target, variant=variant, toggles=flags, validation=validation)
        own['winner'] = select(own['results'])
        emit('target_selected', target=target, winner=own['winner'])
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'manifest', 'directory', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    search(args.cache, args.manifest, args.directory, args.output)
