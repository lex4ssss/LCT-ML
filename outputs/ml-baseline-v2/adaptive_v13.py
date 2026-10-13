from pathlib import Path
import hashlib
import joblib
import numpy as np
import experiment_v2 as ex
import explore_v7 as e7
import explore_v8 as e8
import models_v12 as metrics


VARIANTS = ('R', 'F365', 'W', 'W365')


def toggles(variant):
    if variant not in VARIANTS:
        raise ValueError('variant')
    return dict(retraining=variant.startswith('W'), cadence_days=7 if variant.startswith('W') else None,
                recency_weights=variant.endswith('365'), half_life_days=365 if variant.endswith('365') else None,
                class_balancing=False, weather=False, numeric=False, domains=False, object_identity=False,
                synthetic=False, neural_network=False, calibration=False, per_object_thresholds=False,
                per_month_thresholds=False, validation_early_stopping=False)


def blocks(days, present, hours, variant):
    flags = toggles(variant)
    days = np.asarray(days).astype('datetime64[us]')
    if np.any(days >= e7.VALIDATION[1]):
        raise ValueError('test rows in search')
    _, evaluation = e8.masks(days, present, hours)
    starts = np.arange(e7.VALIDATION[0], e7.VALIDATION[1], np.timedelta64(7, 'D')) if flags['retraining'] else [e7.VALIDATION[0]]
    for start in starts:
        end = min(start + np.timedelta64(7, 'D'), e7.VALIDATION[1]) if flags['retraining'] else e7.VALIDATION[1]
        prediction = np.flatnonzero(evaluation & (days >= start) & (days < end))
        if len(prediction):
            train = np.flatnonzero(present & (days < start - hours * e7.HOUR))
            if not len(train):
                raise ValueError('no training history')
            yield np.datetime64(start, 'us'), np.datetime64(end, 'us'), train, prediction


def recency_weights(days, cutoff, half_life=365):
    if half_life <= 0:
        raise ValueError('half life')
    ages = (np.datetime64(cutoff, 'us') - np.asarray(days).astype('datetime64[us]')) / np.timedelta64(1, 'D')
    if not len(ages) or np.any(ages < 0) or not np.isfinite(ages).all():
        raise ValueError('training ages')
    weights = np.exp2(-ages / half_life)
    return weights / weights.mean()


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def score_blocks(matrix, labels, days, present, hours, variant, checkpoint_dir=None,
                 input_digest=None, model_factory=None, on_block=None):
    if variant == 'R':
        raise ValueError('reference uses submitted artifact')
    flags = toggles(variant)
    if checkpoint_dir is not None:
        if not input_digest:
            raise ValueError('checkpoint fingerprint required')
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _, evaluation = e8.masks(days, present, hours)
    evaluation_indices = np.flatnonzero(evaluation)
    if not len(evaluation_indices):
        raise ValueError('no evaluation rows')
    scores = np.full(len(days), np.nan)
    records = []
    for index, (start, end, train, prediction) in enumerate(blocks(days, present, hours, variant)):
        path = checkpoint_dir / f'{index:03d}.joblib' if checkpoint_dir is not None else None
        if path is not None and path.exists():
            saved = joblib.load(path)
            if saved['input_digest'] != input_digest or saved['variant'] != variant or saved['hours'] != hours:
                raise ValueError('checkpoint fingerprint changed')
            if saved['start'] != str(start) or not np.array_equal(saved['indices'], prediction):
                raise ValueError('checkpoint schedule changed')
            if file_digest(path.with_suffix('.model.joblib')) != saved['model_sha256']:
                raise ValueError('checkpoint model changed')
            probabilities = saved['scores']
            record = saved['record']
        else:
            cutoff = start - hours * e7.HOUR
            weights = recency_weights(days[train], cutoff) if flags['recency_weights'] else None
            model = (model_factory or (lambda: ex.make_model('hgb')))()
            model.fit(matrix[train], labels[train], sample_weight=weights)
            probabilities = model.predict_proba(matrix[prediction])[:, 1]
            record = dict(block_start=str(start), block_end=str(end), train_rows=len(train), predict_rows=len(prediction),
                          train_first_as_of=str(days[train].min()), train_last_as_of=str(days[train].max()),
                          label_window_last_end=str(days[train].max() + hours * e7.HOUR))
            if path is not None:
                model_path = path.with_suffix('.model.joblib')
                joblib.dump(model, model_path)
                loaded = joblib.load(model_path)
                np.testing.assert_array_equal(loaded.predict_proba(matrix[prediction][:37])[:, 1], probabilities[:37])
                saved = dict(input_digest=input_digest, variant=variant, hours=hours, start=str(start), indices=prediction,
                             scores=probabilities, record=record, model_sha256=file_digest(model_path))
                joblib.dump(saved, path)
        if len(probabilities) != len(prediction) or not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError('invalid probabilities')
        if np.isfinite(scores[prediction]).any():
            raise ValueError('duplicate predictions')
        scores[prediction] = probabilities
        records.append(record)
        if on_block:
            on_block(index, record)
    if not np.isfinite(scores[evaluation]).all():
        raise ValueError('missing predictions')
    return scores[evaluation], records


def diagnostics(labels, scores, days, threshold):
    result = {}
    months = np.asarray(days).astype('datetime64[M]')
    for month in np.unique(months):
        mask = months == month
        row = e8.confusion(labels[mask], scores[mask], threshold)
        row.update(rows=int(mask.sum()), base_rate=float(labels[mask].mean()))
        result[str(month)] = row
    worst = {name: min(row[name] for row in result.values()) for name in ['precision', 'recall', 'f1']}
    return dict(monthly=result, worst_month=worst)


def choose(results):
    reference = results['R']['validation']
    accepted = [(name, row) for name, row in results.items() if name != 'R' and metrics.accepted(row['validation'], reference)]
    if not accepted:
        return 'R'
    return max(accepted, key=lambda pair: (pair[1]['validation']['ap'], pair[1]['validation']['margin'], -pair[1]['model_count']))[0]
