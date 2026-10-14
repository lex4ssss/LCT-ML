import argparse
import json
from importlib.metadata import version as package_version
import os
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
import adaptive_v13 as adaptive
import explore_v8 as e8
import explore_v16 as prior
import models_v12 as models
import models_v17 as capacity


VARIANTS = ('R', 'H_TUNED', 'C_TUNED', 'E_TUNED')


def toggles(variant):
    return dict(channel_trajectories=variant != 'R', channel_numeric=variant != 'R',
                object_identity=variant in ('C_TUNED', 'E_TUNED'), equal_ensemble=variant == 'E_TUNED',
                parameter_selection_2024=variant != 'R', retraining=False, recency_weights=False, class_balancing=False,
                object_numeric=False, domains=False, object_state_histories=False, object_family_histories=False,
                object_rhythm=False, weather=False, synthetic=False, neural_network=False, calibration=False,
                per_channel_thresholds=False, per_object_thresholds=False)


def persist(report, path):
    path.write_text(json.dumps(report, indent=2) + '\n')


def inner_search(base, matrix, data, shared, hours, target, output, own, report, report_path):
    train, evaluation = capacity.inner_masks(shared['days'], data['present'], hours)
    inner = own.setdefault('inner', {})
    prior.verify_scores(inner, output, target)
    for name, parameters in {'R': capacity.HGB_GRID['H0'], **capacity.grid()}.items():
        key = f'{name}-inner'
        if key in inner:
            continue
        started = time.monotonic()
        selected_matrix = base if name == 'R' else matrix
        estimator_name = 'H0' if name == 'R' else name
        prior.emit('inner_training_started', target=target, variant=name, parameters=parameters,
                   toggles=toggles('R' if name == 'R' else 'C_TUNED' if name.startswith('C') else 'H_TUNED'))
        model = capacity.fit(estimator_name, selected_matrix[train], data['labels'][train], shared['object_ids'][train], shared['categories'], shared['days'][train])
        scores = capacity.predict(estimator_name, model, selected_matrix[evaluation], shared['object_ids'][evaluation], shared['categories'])
        score_path = output / f'{target}-{key}-scores.npy'
        np.save(score_path, scores)
        inner[key] = dict(ap=float(average_precision_score(data['labels'][evaluation], scores)),
                          base_rate=float(data['labels'][evaluation].mean()), train_rows=int(train.sum()), rows=int(evaluation.sum()),
                          parameters=parameters, score_digest=adaptive.file_digest(score_path),
                          toggles=toggles('R' if name == 'R' else 'C_TUNED' if name.startswith('C') else 'H_TUNED'),
                          elapsed_seconds=time.monotonic() - started)
        del model
        persist(report, report_path)
        prior.emit('inner_candidate_complete', target=target, variant=name, ap=inner[key]['ap'], parameters=parameters, toggles=inner[key]['toggles'])
    own['selected'] = {family: capacity.choose(inner, family) for family in ('H', 'C')}
    persist(report, report_path)
    prior.emit('inner_selected', target=target, selected=own['selected'])


def search(cache, manifest, channel_report_path, output):
    source = Path(__file__).parent
    previous = json.loads(manifest.read_text())
    channel_report = json.loads(channel_report_path.read_text())
    if not channel_report['complete'] or channel_report['test_evaluation']:
        raise ValueError('invalid channel source experiment')
    if any(value['winner'] != 'R' for value in channel_report['targets'].values()):
        raise ValueError('reference source changed')
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
                    code={name: adaptive.file_digest(source / name) for name in ('models_v17.py', 'explore_v17.py', 'spec_explore_v17.txt')})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'search.json'
    report = prior.restore_report(report_path, identity, packages)
    report['threads'] = threads
    report['grid'] = capacity.grid()
    persist(report, report_path)
    shared = joblib.load(cache)
    if np.any(shared['days'] >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('test rows in cache')
    built = joblib.load(feature_path)
    for target, hours in (('incident', 72), ('neispraven', 168)):
        data = shared['targets'][target]
        base = np.column_stack([shared['base'], data['history']])
        matrix = np.column_stack([base, built['trajectories'], built['numeric']])
        own = report['targets'].setdefault(target, dict(horizon_hours=hours, results={}))
        inner_search(base, matrix, data, shared, hours, target, output, own, report, report_path)
        train, evaluation = e8.masks(shared['days'], data['present'], hours)
        prior.verify_scores(own['results'], output, target)
        for variant in VARIANTS:
            if variant in own['results']:
                continue
            started = time.monotonic()
            flags = toggles(variant)
            threshold = None
            parameters = None
            if variant == 'R':
                scores = np.load(manifest.parent / f'{target}-R-scores.npy')
                threshold = previous['targets'][target]['results']['R']['validation']['threshold']
            elif variant == 'E_TUNED':
                scores = (np.load(output / f'{target}-H_TUNED-scores.npy') + np.load(output / f'{target}-C_TUNED-scores.npy')) / 2
                parameters = {family: capacity.grid()[name] for family, name in own['selected'].items()}
            else:
                name = own['selected'][variant[0]]
                parameters = capacity.grid()[name]
                prior.emit('training_started', target=target, variant=variant, candidate=name, parameters=parameters, toggles=flags)
                model = capacity.fit(name, matrix[train], data['labels'][train], shared['object_ids'][train], shared['categories'], shared['days'][train])
                scores = capacity.predict(name, model, matrix[evaluation], shared['object_ids'][evaluation], shared['categories'])
                models.save_verified(model, output / f'{target}-{variant}.joblib', variant, matrix[evaluation][:37], shared['object_ids'][evaluation][:37], shared['categories'])
                if name in ('H0', 'C0'):
                    original = 'H_TN' if name == 'H0' else 'C_TN'
                    old_scores_path = channel_report_path.parent / f'{target}-{original}-scores.npy'
                    if adaptive.file_digest(old_scores_path) != channel_report['targets'][target]['results'][original]['score_digest']:
                        raise ValueError('source channel scores changed')
                    np.testing.assert_array_equal(scores, np.load(old_scores_path))
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
            own['results'][variant] = dict(validation=validation, toggles=flags, parameters=parameters,
                features=(base.shape[1] if variant == 'R' else matrix.shape[1]) + int(flags['object_identity']),
                model_count=2 if variant == 'E_TUNED' else 1, score_digest=adaptive.file_digest(score_path),
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
