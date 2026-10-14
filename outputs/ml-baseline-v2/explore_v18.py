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
import explore_v8 as e8
import explore_v16 as prior
import models_v12 as models
import telemetry_v12 as telemetry
import time_v18 as timing


VARIANTS = ('R', 'H_TIME_B', 'C_TIME_B', 'E_TIME_B', 'H_TIME_TN', 'C_TIME_TN', 'E_TIME_TN')


def toggles(variant):
    if variant not in VARIANTS:
        raise ValueError('variant')
    return dict(conditional_time_heads=variant != 'R', channel_trajectories=variant.endswith('_TN'),
                channel_numeric=variant.endswith('_TN'), object_identity=variant.startswith(('C_', 'E_')),
                equal_ensemble=variant.startswith('E_'), retraining=False, recency_weights=False, class_balancing=False,
                object_numeric=False, domains=False, object_state_histories=False, object_family_histories=False,
                object_rhythm=False, weather=False, synthetic=False, neural_network=False, calibration=False,
                per_channel_thresholds=False, per_object_thresholds=False)


def persist(report, path):
    path.write_text(json.dumps(report, indent=2) + '\n')


def fit_training(variant, matrix, data, shared, onsets, hours):
    train, _ = e8.masks(shared['days'], data['present'], hours)
    return timing.fit(variant, matrix[train], onsets[train], shared['object_ids'][train], shared['categories'], shared['days'][train], hours)


def verify_labels(derived, data, days, hours):
    train, evaluation = e8.masks(days, np.ones(len(days), bool), hours)
    scope = train | evaluation
    np.testing.assert_array_equal(derived['present'][scope], data['present'][scope])
    selected = scope & data['present']
    np.testing.assert_array_equal(derived['label'][selected], data['labels'][selected])
    np.testing.assert_array_equal(np.isfinite(derived['next_hours'][selected]), data['labels'][selected].astype(bool))


def search(cache, manifest, channel_report_path, output):
    source = Path(__file__).parent
    previous = json.loads(manifest.read_text())
    channel_report = json.loads(channel_report_path.read_text())
    if not channel_report['complete'] or channel_report['test_evaluation']:
        raise ValueError('invalid channel source experiment')
    identity = prior.fingerprint(source, cache, manifest, previous)
    if identity != channel_report['fingerprint']:
        raise ValueError('channel source fingerprint changed')
    packages = {name: package_version(name) for name in previous['packages']}
    threads = {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}
    if packages != previous['packages'] or threads != channel_report['threads']:
        raise ValueError('runtime changed')
    feature_path = channel_report_path.parent / 'channel_features.joblib'
    if adaptive.file_digest(feature_path) != channel_report['feature_digest']:
        raise ValueError('channel feature cache changed')
    identity = dict(source=identity, channel_report=adaptive.file_digest(channel_report_path),
                    channel_features=adaptive.file_digest(feature_path),
                    code={name: adaptive.file_digest(source / name) for name in
                          ('time_v18.py', 'explore_v18.py', 'spec_explore_v18.txt', 'families_v14.py')})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'search.json'
    report = prior.restore_report(report_path, identity, packages)
    report['threads'] = threads
    report['hgb_parameters'] = channel_report['hgb_parameters']
    report['catboost_parameters'] = channel_report['catboost_parameters']
    report['intervals'] = {str(hours): list(ends) for hours, ends in timing.INTERVALS.items()}
    persist(report, report_path)
    shared = joblib.load(cache)
    if np.any(shared['days'] >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('test rows in cache')
    built = joblib.load(feature_path)
    label_path = output / 'time_labels.joblib'
    if label_path.exists() and 'label_digest' in report:
        if adaptive.file_digest(label_path) != report['label_digest']:
            raise ValueError('time label cache changed')
        labels = joblib.load(label_path)
    else:
        paths = {Path(name).name: Path(name) for name in previous['inputs']}
        directory = next(Path(name) for name in previous['inputs'] if Path(name).suffix == '.csv')
        mapping = telemetry.registry(directory).drop_duplicates('channel_id', keep='last')
        codes = {name: index for index, name in enumerate(shared['categories'])}
        mapping['object_code'] = [codes[name] for name in mapping['object_id']]
        labels = {}
        with duckdb.connect(config={'threads': 4, 'memory_limit': '2GB'}) as connection:
            for target, hours in (('incident', 72), ('neispraven', 168)):
                prior.emit('time_labels_started', target=target)
                table = timing.label_table(connection, paths['training-examples-v2.parquet'], paths[f'episodes-900-{target}.parquet'], mapping, codes['unknown'], hours)
                labels[target] = timing.align(table, shared)
                verify_labels(labels[target], shared['targets'][target], shared['days'], hours)
                prior.emit('time_labels_verified', target=target, object_days=len(table))
        joblib.dump(labels, label_path)
        report['label_digest'] = adaptive.file_digest(label_path)
        persist(report, report_path)
    report['reference_error_audit'] = {}
    for target, hours in (('incident', 72), ('neispraven', 168)):
        data = shared['targets'][target]
        verify_labels(labels[target], data, shared['days'], hours)
        _, evaluation = e8.masks(shared['days'], data['present'], hours)
        threshold = previous['targets'][target]['results']['R']['validation']['threshold']
        audit = timing.error_audit(data['labels'][evaluation], np.load(manifest.parent / f'{target}-R-scores.npy'), threshold,
                                   labels[target]['next_hours'][evaluation], data['history'][evaluation],
                                   {name: values[evaluation] for name, values in labels[target].items() if name.endswith('_channels')}, hours)
        report['reference_error_audit'][target] = dict(toggles=toggles('R'), audit=audit)
        prior.emit('reference_error_audit_ready', target=target, audit=audit, toggles=toggles('R'))
    persist(report, report_path)
    for target, hours in (('incident', 72), ('neispraven', 168)):
        data = shared['targets'][target]
        base = np.column_stack([shared['base'], data['history']])
        _, evaluation = e8.masks(shared['days'], data['present'], hours)
        own = report['targets'].setdefault(target, dict(horizon_hours=hours, results={}))
        prior.verify_scores(own['results'], output, target)
        for variant in VARIANTS:
            if variant in own['results']:
                continue
            started = time.monotonic()
            flags = toggles(variant)
            matrix = np.column_stack([base, built['trajectories'], built['numeric']]) if flags['channel_trajectories'] else base
            threshold = None
            training = None
            if variant == 'R':
                scores = np.load(manifest.parent / f'{target}-R-scores.npy')
                threshold = previous['targets'][target]['results']['R']['validation']['threshold']
                model_count = 1
            elif variant.startswith('E_'):
                h_variant, c_variant = variant.replace('E_', 'H_', 1), variant.replace('E_', 'C_', 1)
                scores = (np.load(output / f'{target}-{h_variant}-scores.npy') + np.load(output / f'{target}-{c_variant}-scores.npy')) / 2
                model_count = own['results'][h_variant]['model_count'] + own['results'][c_variant]['model_count']
            else:
                prior.emit('training_started', target=target, variant=variant, toggles=flags)
                model = fit_training(variant, matrix, data, shared, labels[target]['next_hours'], hours)
                scores = models.predict_model(variant, model, matrix[evaluation], shared['object_ids'][evaluation], shared['categories'])
                models.save_verified(model, output / f'{target}-{variant}.joblib', variant, matrix[evaluation][:37], shared['object_ids'][evaluation][:37], shared['categories'])
                training = model.training
                model_count = len(model.heads)
                del model
            validation = models.metrics(data['labels'][evaluation], scores, data['rule'][evaluation], data['strong_rule'][evaluation], threshold)
            if variant == 'R':
                old = previous['targets'][target]['results']['R']['validation']
                for name in ('rows', 'base_rate', 'precision', 'recall', 'f1', 'ap', 'threshold'):
                    if not np.isclose(validation[name], old[name], rtol=1e-12, atol=1e-12):
                        raise ValueError('reference result changed')
            known = shared['object_ids'][evaluation] != 'unknown'
            score_path = output / f'{target}-{variant}-scores.npy'
            np.save(score_path, scores)
            own['results'][variant] = dict(validation=validation, toggles=flags, head_training=training,
                features=matrix.shape[1] + int(flags['object_identity']), model_count=model_count, score_digest=adaptive.file_digest(score_path),
                known_objects=models.metrics(data['labels'][evaluation][known], scores[known], data['rule'][evaluation][known], data['strong_rule'][evaluation][known], validation['threshold']),
                diagnostics=adaptive.diagnostics(data['labels'][evaluation], scores, shared['days'][evaluation], validation['threshold']),
                elapsed_seconds=time.monotonic() - started)
            persist(report, report_path)
            prior.emit('variant_complete', target=target, variant=variant, validation=validation, toggles=flags)
        own['winner'] = prior.select(own['results'])
        prior.emit('target_selected', target=target, winner=own['winner'])
    report['complete'] = True
    persist(report, report_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'manifest', 'channel_report', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    search(args.cache, args.manifest, args.channel_report, args.output)
