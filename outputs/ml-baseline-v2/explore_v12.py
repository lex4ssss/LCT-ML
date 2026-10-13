import argparse
import gc
import glob
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import duckdb
import joblib
import numpy as np
import explore_v7 as e7
import explore_v8 as e8
import eval_units as eu
import models_v12 as models
import run_v2 as rv
import telemetry_v12 as telemetry


TARGETS = {'incident': (72, 'run-008-incident72', 'hgb_object_incident_72h.joblib'),
           'neispraven': (168, 'run-009-neispraven168', 'hgb_object_neispraven_168h.joblib')}


def digest(path):
    value = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values)), flush=True)


def build_shared(connection, examples, directory, observations):
    columns = rv.load(connection, examples)
    emit('channel_snapshots_loaded', rows=len(columns['as_of']))
    codes, object_code, sensor, type_count = e8.channel_info(connection, directory, columns['channel_id'])
    keys, base = e8.aggregate(columns, codes, sensor, type_count)
    days = e8.key_moments(keys)
    categories = [name for name, code in sorted(object_code.items(), key=lambda pair: pair[1])]
    object_ids = np.asarray(categories)[keys // 1_000_000]
    channel_registry = telemetry.registry(directory)
    domains, domain_names = telemetry.domain_matrix(columns, channel_registry, keys, object_code)
    emit('domains_ready', object_days=len(keys), features=len(domain_names))
    daily = telemetry.numeric_daily(connection, observations, channel_registry, days.max())
    numeric, numeric_names = telemetry.numeric_matrix(daily, object_ids, days)
    emit('numeric_ready', daily_rows=len(daily), features=len(numeric_names))
    targets = {}
    row_keys = eu.day_keys(codes, columns['as_of'])
    for target, (hours, _, _) in TARGETS.items():
        episodes = examples.parent / f'episodes-900-{target}.parquet'
        labels = e7.horizon_labels(connection, episodes, columns['channel_id'], columns['as_of'], hours)
        present, label, rule = e8.object_labels(keys, row_keys, labels, columns['alarm_count_24h'])
        history = e8.target_history(connection, episodes, directory, keys)
        targets[target] = dict(present=present, labels=label, rule=rule, strong_rule=(history[:, 1] > 0).astype(float), history=history)
        emit('target_ready', target=target, eligible_object_days=int(present.sum()))
    del columns, codes, sensor, daily, labels, row_keys
    gc.collect()
    return dict(base=base, numeric=numeric, domains=domains, numeric_names=numeric_names,
                domain_names=domain_names, days=days, object_ids=object_ids, categories=categories, targets=targets)


def search(examples, directory, observations, output):
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parent
    inputs = [examples, directory, examples.parent / 'episodes-900-incident.parquet', examples.parent / 'episodes-900-neispraven.parquet']
    observations_paths = [Path(path) for path in sorted(glob.glob(observations)) if Path(path).stem.isdigit() and int(Path(path).stem) < 2026]
    if not observations_paths:
        raise ValueError('no pre-2026 observation files')
    inputs += observations_paths
    fingerprints = {str(path): digest(path) for path in inputs}
    code_files = ['explore_v12.py', 'models_v12.py', 'telemetry_v12.py', 'spec_explore_v12.txt']
    code_hashes = {name: digest(source / name) for name in code_files}
    report_path, cache_path = output / 'search.json', output / 'features.joblib'
    report = dict(protocol='Selection on 2025 only; test 2026 predictions are not computed.',
                  previous_test_looks=['run-002', 'run-006', 'run-007', 'run-008', 'run-009', 'run-010', 'run-011'],
                  inputs=fingerprints, code=code_hashes, targets={}, complete=False,
                  packages={name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'duckdb', 'catboost', 'lightgbm']})
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['inputs'] != fingerprints or report['code'] != code_hashes:
            raise ValueError('resume inputs or implementation changed')
    elif cache_path.exists():
        raise ValueError('orphan feature cache')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    if cache_path.exists():
        shared = joblib.load(cache_path)
        emit('feature_cache_loaded')
    else:
        with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
            shared = build_shared(connection, examples, directory, observations_paths)
        joblib.dump(shared, cache_path)
    for target, (hours, reference_run, reference_file) in TARGETS.items():
        target_data = shared['targets'][target]
        base = np.column_stack([shared['base'], target_data['history']])
        results = report['targets'].setdefault(target, dict(horizon_hours=hours, results={}))['results']
        reference_report = json.loads((source / reference_run / 'report.json').read_text())['year_2025']
        scores_by_variant = {}
        for variant in models.VARIANTS:
            score_path = output / f'{target}-{variant}-scores.npy'
            if variant in results:
                scores_by_variant[variant] = np.load(score_path)
                continue
            started = time.monotonic()
            matrix = models.feature_matrix(variant, base, shared['numeric'], shared['domains'])
            train, evaluation = models.training_masks(shared['days'], target_data['present'], hours, variant)
            labels = target_data['labels'][evaluation]
            object_ids = shared['object_ids'][evaluation]
            if variant == 'E':
                scores = (scores_by_variant['C_ND_2022'] + scores_by_variant['L_ND_2022']) / 2
            else:
                model = joblib.load(source / reference_run / reference_file) if variant == 'R' else models.fit_model(
                    variant, matrix[train], target_data['labels'][train], shared['object_ids'][train], shared['categories'], shared['days'][train])
                scores = models.predict_model(variant, model, matrix[evaluation], object_ids, shared['categories'])
                models.save_verified(model, output / f'{target}-{variant}.joblib', variant, matrix[evaluation][:37], object_ids[:37], shared['categories'])
            scores_by_variant[variant] = scores
            np.save(score_path, scores)
            validation = models.metrics(labels, scores, target_data['rule'][evaluation], target_data['strong_rule'][evaluation],
                                        reference_report['threshold'] if variant == 'R' else None)
            known = object_ids != 'unknown'
            known_metrics = models.metrics(labels[known], scores[known], target_data['rule'][evaluation][known],
                                          target_data['strong_rule'][evaluation][known], validation['threshold'])
            if variant == 'R':
                for name in ['rows', 'base_rate', 'precision_at_recall_0_5', 'precision', 'recall', 'f1', 'threshold']:
                    if not np.isclose(validation[name], reference_report[name], atol=1e-12, rtol=1e-12):
                        raise ValueError('reference 2025 result changed: ' + name)
            results[variant] = dict(validation=validation, known_objects=known_metrics, toggles=models.toggles(variant),
                                    features=matrix.shape[1] + int(models.toggles(variant)['object_identity']),
                                    train_rows=int(train.sum()), train_first_as_of=str(shared['days'][train].min()),
                                    train_last_as_of=str(shared['days'][train].max()), elapsed_seconds=time.monotonic() - started)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            emit('variant_complete', target=target, variant=variant, toggles=models.toggles(variant), validation=validation,
                 elapsed_seconds=results[variant]['elapsed_seconds'])
            del matrix
            gc.collect()
        report['targets'][target]['winner'] = models.winner(results)
        emit('target_selected', target=target, winner=report['targets'][target]['winner'])
    report['complete'] = True
    report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('examples', type=Path)
    parser.add_argument('directory', type=Path)
    parser.add_argument('observations')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    search(args.examples, args.directory, args.observations, args.output)
