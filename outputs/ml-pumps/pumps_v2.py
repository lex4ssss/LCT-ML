import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pumps_data
from pumps_v1 import HORIZONS, curve, recall_by_fault_type, scores_at

TARGET_PRECISION = 0.95
TARGET_RECALL = 0.5
MAX_BASE_RATE = 0.35


def choose_threshold(labels, scores):
    thresholds, precision, recall = curve(labels, scores)
    eligible = (precision >= TARGET_PRECISION) & (recall >= TARGET_RECALL)
    if not eligible.any():
        return None, None
    margin = np.minimum(precision - TARGET_PRECISION, recall - TARGET_RECALL)
    best = np.flatnonzero(eligible)[np.argmax(margin[eligible])]
    return float(thresholds[best]), float(margin[best])


def validation_summary(labels, scores):
    _, precision, recall = curve(labels, scores)
    at_recall = recall >= TARGET_RECALL
    at_precision = precision >= TARGET_PRECISION
    return dict(best_precision_at_recall_0_5=round(float(precision[at_recall].max()), 4) if at_recall.any() else 0.0,
                recall_at_precision_0_95=round(float(recall[at_precision].max()), 4) if at_precision.any() else 0.0)


def reconstruction_quality(reconstructed, seed=0):
    rows = reconstructed[reconstructed['trajectory'] >= 0].sort_values(['trajectory', 'step'], kind='stable')
    score = rows['anomaly_score_1'].to_numpy()
    same = rows['trajectory'].to_numpy()[1:] == rows['trajectory'].to_numpy()[:-1]
    matched = float(np.abs(np.diff(score))[same].mean())
    rng = np.random.default_rng(seed)
    by_step = rows.groupby(['pump_id', 'step'])['anomaly_score_1'].transform(lambda s: rng.permutation(s.to_numpy()))
    shuffled = rows.assign(anomaly_score_1=by_step.to_numpy())
    random_score = shuffled['anomaly_score_1'].to_numpy()
    return dict(matched_mean_abs_step=round(matched, 5),
                random_mean_abs_step=round(float(np.abs(np.diff(random_score))[same].mean()), 5))


def prepare(path):
    reconstructed = pumps_data.reconstruct(pumps_data.load(path), allow_extra=True)
    v1, labels, meta = pumps_data.prediction_rows(reconstructed, HORIZONS.values(), feature_set='v1')
    v2, _, _ = pumps_data.prediction_rows(reconstructed, HORIZONS.values(), feature_set='v2')
    shape = dict(rows=int(len(reconstructed)), trajectories=int(reconstructed['trajectory'].max() + 1),
                 normal_rows=int((reconstructed['trajectory'] < 0).sum()), prediction_rows=int(len(v1)),
                 reconstruction=reconstruction_quality(reconstructed))
    return dict(v1=v1, v2=v2), labels, meta, shape


def make_models():
    return dict(
        A=('v1', HistGradientBoostingClassifier(random_state=0)),
        B=('v2', HistGradientBoostingClassifier(random_state=0)),
        C=('v2', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                               random_state=0))),
    )


def predict(fitted, features, variant):
    if variant == 'D':
        return (predict(fitted, features, 'B') + predict(fitted, features, 'C')) / 2
    feature_set, model = fitted[variant]
    return model.predict_proba(features[feature_set])[:, 1]


def run(primary, replications, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    features, labels, meta, shape = prepare(primary)
    part = pumps_data.split(meta)
    rule = pumps_data.rule_alarm(features['v1'])
    fault_type = meta['fault_type'].fillna('').to_numpy()
    report = dict(spec='spec_pumps_v2.txt', primary=dict(file=primary.name, **shape,
                  split={p: int((part == p).sum()) for p in ('train', 'validation', 'test')}), horizons={})
    fitted_by_horizon, chosen = {}, {}
    train, valid, test = part == 'train', part == 'validation', part == 'test'
    for name, minutes in HORIZONS.items():
        y = labels[minutes].astype(np.int64)
        fitted = make_models()
        for feature_set, model in fitted.values():
            model.fit(features[feature_set][train], y[train])
        fitted_by_horizon[name] = fitted
        rule_valid = scores_at(y[valid], rule[valid])
        variants = {}
        for variant in ('A', 'B', 'C', 'D'):
            scores = predict(fitted, {k: v[valid] for k, v in features.items()}, variant)
            threshold, margin = choose_threshold(y[valid], scores)
            model_valid = scores_at(y[valid], scores >= threshold) if threshold is not None else None
            qualified = (threshold is not None and rule_valid['base_rate'] <= MAX_BASE_RATE
                         and model_valid['f1'] > rule_valid['f1'])
            variants[variant] = dict(**validation_summary(y[valid], scores), threshold=threshold, margin=margin,
                                     model=model_valid, qualified=bool(qualified))
        qualifiers = [v for v in variants if variants[v]['qualified']]
        entry = dict(validation=dict(rule=rule_valid, variants=variants))
        if qualifiers:
            best = max(qualifiers, key=lambda v: variants[v]['margin'])
            chosen[name] = best
            alarms = predict(fitted, {k: v[test] for k, v in features.items()}, best) >= variants[best]['threshold']
            entry['chosen'] = best
            entry['test'] = dict(model=scores_at(y[test], alarms), rule=scores_at(y[test], rule[test]),
                                 recall_by_fault_type=recall_by_fault_type(y[test], alarms, fault_type[test]))
        report['horizons'][name] = entry
    report['replication'] = {}
    for path in replications:
        try:
            rep_features, rep_labels, rep_meta, rep_shape = prepare(path)
        except pumps_data.NotReconstructable as error:
            report['replication'][path.name] = dict(not_reconstructable=str(error))
            continue
        rep_rule = pumps_data.rule_alarm(rep_features['v1'])
        rep_fault_type = rep_meta['fault_type'].fillna('').to_numpy()
        entry = dict(**rep_shape)
        for name, variant in chosen.items():
            y = rep_labels[HORIZONS[name]].astype(np.int64)
            threshold = report['horizons'][name]['validation']['variants'][variant]['threshold']
            alarms = predict(fitted_by_horizon[name], rep_features, variant) >= threshold
            entry[name] = dict(variant=variant, model=scores_at(y, alarms), rule=scores_at(y, rep_rule),
                               recall_by_fault_type=recall_by_fault_type(y, alarms, rep_fault_type))
        report['replication'][path.name] = entry
    joblib.dump({name: fitted_by_horizon[name] for name in chosen}, out_dir / 'pumps_v2_models.joblib')
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--replication', type=Path, nargs='*', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.primary, args.replication, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
