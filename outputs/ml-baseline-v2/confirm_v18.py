import argparse
import json
from importlib.metadata import version as package_version
import os
from pathlib import Path
import duckdb
import joblib
import numpy as np
import adaptive_v13 as adaptive
import channel_features_v16 as channels
import explore_v8 as e8
import explore_v16 as prior
import final_v2
import models_v12 as models
import telemetry_v12 as telemetry


TARGET = 'neispraven'
VARIANT = 'C_TIME_TN'
MODEL_SHA = '219517cede07436d3d99bd0f8c232d8e3eb018fb30cb4a67a7c7970213cd9463'
REFERENCE_SHA = '97d51d0e584f6992b21e66cc7a67580ea9d76e689b59ef846a7d631762fcfe93'
THRESHOLD = 0.5680074649760845


def freeze(selection, model_digest):
    own = selection['targets'][TARGET]
    if not selection['complete'] or selection['test_evaluation'] or own['winner'] != VARIANT or prior.select(own['results']) != VARIANT:
        raise ValueError('selected model changed')
    if own['results'][VARIANT]['validation']['threshold'] != THRESHOLD or model_digest != MODEL_SHA:
        raise ValueError('frozen model or threshold changed')
    return own['results'][VARIANT]


def test_mask(days, present, hours):
    days = np.asarray(days).astype('datetime64[us]')
    present = np.asarray(present, dtype=bool)
    if len(days) != len(present) or np.isnat(days).any() or not present.any() or np.any(days < np.datetime64('2026-01-01')):
        raise ValueError('original test timestamps required')
    return present & (days <= days[present].max() - np.timedelta64(hours - 24, 'h'))


def verify_reference(actual, expected):
    aliases = dict(rows='rows', base_rate='prevalence', precision='precision', recall='recall', f1='f1', ap='average_precision', threshold='threshold')
    for name, old in aliases.items():
        if not np.isclose(actual[name], expected[old], rtol=1e-12, atol=1e-12):
            raise ValueError('reference test result changed: ' + name)


def accepted(candidate, reference):
    return bool(candidate['qualifies'] and all(candidate[name] >= reference[name] for name in ('precision', 'recall', 'f1', 'ap')))


def saved_confirmation(path, identity):
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if report['fingerprint'] != identity:
        raise ValueError('confirmation fingerprint changed')
    if not report['complete']:
        return None
    for name, digest in report['score_digests'].items():
        if adaptive.file_digest(path.parent / name) != digest:
            raise ValueError('confirmation scores changed')
    return report


def verified_feature_cache(path, expected):
    path = Path(path)
    if adaptive.file_digest(path) != expected:
        raise ValueError('source feature cache changed')
    return path


def build_features(source, previous, cache, feature_cache, observations, directory, episodes, examples):
    with duckdb.connect(config={'threads': 4, 'memory_limit': '2GB'}) as connection:
        columns = final_v2.load(connection, examples, 'test')
        base, days, present, labels, rule = e8.frame(connection, columns, directory, episodes, 168)
        codes, code, sensor, count = e8.channel_info(connection, directory, columns['channel_id'])
        keys, _ = e8.aggregate(columns, codes, sensor, count)
        names = {value: name for name, value in code.items()}
        objects = np.array([names[int(key // 1_000_000)] for key in keys])
        selected = test_mask(days, present, 168)
        base, days, objects, labels, rule = base[selected], days[selected], objects[selected], labels[selected], rule[selected]
        if len(days) != 13320 or labels.sum() != 3180 or days.max() != np.datetime64('2026-06-23', 'us'):
            raise ValueError('original test cohort changed')
        prior.emit('test_feature_build_started', rows=len(days), last_as_of=str(days.max()))
        daily = channels.daily_channels(connection, observations, telemetry.registry(directory), days.max())
    old = joblib.load(cache)
    earlier = joblib.load(feature_cache)
    sample = np.linspace(0, len(old['days']) - 1, 37, dtype=int)
    for name, builder in (('trajectories', channels.trajectory_matrix), ('numeric', channels.numeric_matrix)):
        actual = builder(daily, old['object_ids'][sample], old['days'][sample])
        np.testing.assert_allclose(actual, earlier[name][sample], rtol=1e-9, atol=1e-9, equal_nan=True)
    trajectories = channels.trajectory_matrix(daily, objects, days)
    numeric = channels.numeric_matrix(daily, objects, days)
    return dict(base=base, matrix=np.column_stack([base, trajectories, numeric]), days=days, object_ids=objects,
                labels=labels, rule=rule, strong_rule=(base[:, 74] > 0).astype(float), categories=old['categories'], prefix_checks=37)


def confirm(cache, manifest, search_path, feature_cache, output):
    source = Path(__file__).parent
    previous = json.loads(manifest.read_text())
    selection = json.loads(search_path.read_text())
    model_path = search_path.parent / f'{TARGET}-{VARIANT}.joblib'
    chosen = freeze(selection, adaptive.file_digest(model_path))
    source_identity = prior.fingerprint(source, cache, manifest, previous)
    if selection['fingerprint']['source'] != source_identity:
        raise ValueError('search source changed')
    for name, digest in selection['fingerprint']['code'].items():
        if adaptive.file_digest(source / name) != digest:
            raise ValueError('search code changed')
    packages = {name: package_version(name) for name in previous['packages']}
    threads = {name: os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')}
    if packages != selection['packages'] or threads != selection['threads']:
        raise ValueError('runtime changed')
    paths = {Path(name).name: Path(name) for name in previous['inputs']}
    observations = [Path(name) for name in previous['inputs'] if Path(name).stem.isdigit()]
    test_observations = observations[0].parent / '2026.parquet'
    observations.append(test_observations)
    directory = next(Path(name) for name in previous['inputs'] if Path(name).suffix == '.csv')
    reference_directory = source / 'run-009-neispraven168'
    reference_decision = json.loads((reference_directory / 'object-decision.json').read_text())
    reference_path = reference_directory / reference_decision['model_path']
    if adaptive.file_digest(reference_path) != REFERENCE_SHA:
        raise ValueError('reference model changed')
    feature_cache = verified_feature_cache(feature_cache, selection['fingerprint']['channel_features'])
    identity = dict(search_source=selection['fingerprint'], selection=adaptive.file_digest(search_path),
                    model=MODEL_SHA, reference_model=REFERENCE_SHA, observations_2026=adaptive.file_digest(test_observations),
                    reference_decision=adaptive.file_digest(reference_directory / 'object-decision.json'),
                    reference_report=adaptive.file_digest(reference_directory / 'report.json'),
                    code={name: adaptive.file_digest(source / name) for name in ('confirm_v18.py', 'spec_confirm_v18.txt', 'final_v2.py')})
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'confirmation.json'
    completed = saved_confirmation(report_path, identity)
    if completed is not None:
        prior.emit('confirmation_verified_without_prediction', accepted=completed['accepted'])
        return completed
    report = json.loads(report_path.read_text()) if report_path.exists() else dict(fingerprint=identity, complete=False,
        target=TARGET, variant=VARIANT, packages=packages, threads=threads, threshold=THRESHOLD,
        training_unchanged=True, calibration=False, prior_test_inspections=7, current_test_inspection=8,
        service_changed=False, toggles=chosen['toggles'])
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    feature_path = output / 'test_features.joblib'
    if 'feature_digest' in report and feature_path.exists():
        if adaptive.file_digest(feature_path) != report['feature_digest']:
            raise ValueError('test feature cache changed')
        built = joblib.load(feature_path)
    else:
        built = build_features(source, previous, cache, feature_cache, observations, directory,
                               paths['episodes-900-neispraven.parquet'], paths['training-examples-v2.parquet'])
        joblib.dump(built, feature_path)
        report['feature_digest'] = adaptive.file_digest(feature_path)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    reference_scores = joblib.load(reference_path).predict_proba(built['base'])[:, 1]
    reference = models.metrics(built['labels'], reference_scores, built['rule'], built['strong_rule'], reference_decision['threshold'])
    expected = json.loads((reference_directory / 'report.json').read_text())['test_object_day']['model']
    verify_reference(reference, expected)
    prior.emit('reference_test_reproduced', rows=reference['rows'])
    np.save(output / 'reference-scores.npy', reference_scores)
    marker = output / 'candidate-started.json'
    candidate_path = output / 'candidate-scores.npy'
    if marker.exists():
        if json.loads(marker.read_text())['fingerprint'] != identity or not candidate_path.exists():
            raise ValueError('candidate already started without recoverable saved scores')
        scores = np.load(candidate_path)
    else:
        if candidate_path.exists():
            raise ValueError('unexpected candidate scores')
        marker.write_text(json.dumps(dict(fingerprint=identity)) + '\n')
        scores = models.predict_model(VARIANT, joblib.load(model_path), built['matrix'], built['object_ids'], built['categories'])
        np.save(candidate_path, scores)
    if scores.shape != reference_scores.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('invalid saved candidate scores')
    candidate = models.metrics(built['labels'], scores, built['rule'], built['strong_rule'], THRESHOLD)
    known = built['object_ids'] != 'unknown'
    report.update(reference=reference, candidate=candidate, accepted=accepted(candidate, reference), prefix_checks=built['prefix_checks'],
                  score_digests={name: adaptive.file_digest(output / name) for name in ('reference-scores.npy', 'candidate-scores.npy')},
                  known_objects=models.metrics(built['labels'][known], scores[known], built['rule'][known], built['strong_rule'][known], THRESHOLD),
                  diagnostics=adaptive.diagnostics(built['labels'], scores, built['days'], THRESHOLD), complete=True)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    prior.emit('confirmation_complete', accepted=report['accepted'], candidate=candidate, reference=reference, toggles=chosen['toggles'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('cache', 'manifest', 'search_report', 'channel_features', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    confirm(args.cache, args.manifest, args.search_report, args.channel_features, args.output)
