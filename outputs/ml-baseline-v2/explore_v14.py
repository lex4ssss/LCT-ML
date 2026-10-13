import argparse
import json
from pathlib import Path
import time
import duckdb
import joblib
import numpy as np
import adaptive_v13 as adaptive
import explore_v8 as e8
import families_v14 as families
import models_v12 as metrics


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def toggles(variant):
    return dict(family_histories=variant != 'R', separate_heads=variant.startswith('HEAD_'),
                combination=variant if variant.startswith('HEAD_') else None, recency_weights=False,
                retraining=False, numeric=False, domains=False, object_identity=False, class_balancing=False,
                weather=False, synthetic=False, calibration=False, per_family_thresholds=False)


def search(cache, manifest, examples, directory, episodes, output, target='incident'):
    hours = {'incident': 72, 'neispraven': 168}[target]
    source = Path(__file__).parent
    previous = json.loads(manifest.read_text())
    if not previous['complete']:
        raise ValueError('source experiment incomplete')
    for name, expected in previous['code'].items():
        if adaptive.file_digest(source / name) != expected:
            raise ValueError('source implementation changed')
    inputs = [cache, manifest, examples, directory, episodes]
    fingerprint = dict(target=target, horizon_hours=hours, inputs={str(path): adaptive.file_digest(path) for path in inputs},
                       code={name: adaptive.file_digest(source / name) for name in
                             ['explore_v14.py', 'families_v14.py', 'spec_explore_v14.txt', 'exp_data.py', 'explore_v8.py']})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'search.json'
    report = dict(fingerprint=fingerprint, packages=previous['packages'], complete=False, test_evaluation=False,
                  service_changed=False, horizon_hours=hours, target=target, results={})
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['fingerprint'] != fingerprint:
            raise ValueError('resume fingerprint changed')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    shared = joblib.load(cache)
    if np.any(shared['days'] >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('test rows in cache')
    data = shared['targets'][target]
    object_code = {name: index for index, name in enumerate(shared['categories'])}
    keys = np.array([object_code[name] for name in shared['object_ids']]) * 1_000_000 + shared['days'].astype('datetime64[D]').astype(np.int64)
    train, evaluation = e8.masks(shared['days'], data['present'], hours)
    feature_path = output / 'family_features.joblib'
    if feature_path.exists():
        built = joblib.load(feature_path)
    else:
        with duckdb.connect(config={'threads': 2, 'memory_limit': '2GB'}) as connection:
            mapping = families.channel_mapping(directory, object_code)
            table = families.label_table(connection, examples, episodes, mapping, object_code['unknown'], hours)
            np.testing.assert_array_equal(table['object_key'], keys)
            required = train | evaluation
            np.testing.assert_array_equal(table['present'].to_numpy()[required], data['present'][required])
            np.testing.assert_array_equal(table['label'].to_numpy()[required], data['labels'][required])
            head_labels = table[[f'family_{index}' for index in range(5)]].to_numpy(dtype=int)
            np.testing.assert_array_equal(head_labels.max(axis=1)[required], data['labels'][required])
            history, names = families.family_history(connection, episodes, mapping, keys, object_code['unknown'])
        built = dict(labels=head_labels, history=history, feature_names=names)
        joblib.dump(built, feature_path)
        emit('labels_match_reference', rows=int(required.sum()), eligible_validation_rows=int(evaluation.sum()), features=len(names))
    base = np.column_stack([shared['base'], data['history']])
    matrix = np.column_stack([base, built['history']])
    labels = data['labels'][evaluation]
    model_scores = None
    for variant in families.VARIANTS:
        if variant in report['results']:
            continue
        started = time.monotonic()
        fixed_threshold = None
        if variant == 'R':
            scores = np.load(manifest.parent / f'{target}-R-scores.npy')
            fixed_threshold = previous['targets'][target]['results']['R']['validation']['threshold']
        elif variant == 'H_F':
            model = families.fit_head(matrix[train], data['labels'][train])
            scores = model.predict_proba(matrix[evaluation])[:, 1]
            families.save_verified(model, output / 'H_F.joblib', matrix[evaluation][:37])
        else:
            head_scores_path = output / 'head_scores.npy'
            if head_scores_path.exists():
                model_scores = np.load(head_scores_path)
            else:
                model_scores = np.zeros((evaluation.sum(), 5))
                for head, name in enumerate(families.FAMILIES):
                    path = output / f'{name}.joblib'
                    if path.exists():
                        model = joblib.load(path)
                    else:
                        model = families.fit_head(matrix[train], built['labels'][train, head])
                        families.save_verified(model, path, matrix[evaluation][:37])
                    model_scores[:, head] = model.predict_proba(matrix[evaluation])[:, 1]
                    emit('head_complete', family=name, train_positives=int(built['labels'][train, head].sum()))
                np.save(head_scores_path, model_scores)
            scores = families.combine(model_scores, variant)
        validation = metrics.metrics(labels, scores, data['rule'][evaluation], data['strong_rule'][evaluation], fixed_threshold)
        if variant == 'R':
            old = previous['targets'][target]['results']['R']['validation']
            for name in ['rows', 'base_rate', 'precision', 'recall', 'f1', 'ap', 'threshold']:
                if not np.isclose(validation[name], old[name], rtol=1e-12, atol=1e-12):
                    raise ValueError('reference result changed')
        known = shared['object_ids'][evaluation] != 'unknown'
        report['results'][variant] = dict(validation=validation, toggles=toggles(variant),
            diagnostics=adaptive.diagnostics(labels, scores, shared['days'][evaluation], validation['threshold']),
            known_objects=metrics.metrics(labels[known], scores[known], data['rule'][evaluation][known],
                                          data['strong_rule'][evaluation][known], validation['threshold']),
            features=base.shape[1] if variant == 'R' else matrix.shape[1], model_count=5 if variant.startswith('HEAD_') else 1,
            elapsed_seconds=time.monotonic() - started)
        np.save(output / f'{variant}-scores.npy', scores)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        emit('variant_complete', variant=variant, toggles=toggles(variant), validation=validation)
    report['winner'] = adaptive.choose(report['results'])
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    emit('selected', winner=report['winner'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['cache', 'manifest', 'examples', 'directory', 'episodes', 'output']:
        parser.add_argument(name, type=Path)
    parser.add_argument('--target', choices=['incident', 'neispraven'], default='incident')
    args = parser.parse_args()
    search(args.cache, args.manifest, args.examples, args.directory, args.episodes, args.output, args.target)
