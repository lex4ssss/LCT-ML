import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import pumps_data

HORIZONS = {'30min': 30, '1h': 60, '8h': 480, '24h': 1440}
TARGET_PRECISION = 0.7
TARGET_RECALL = 0.5
MAX_BASE_RATE = 0.35


def curve(labels, scores):
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    last_of_group = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp = np.cumsum(sorted_labels)[last_of_group]
    fp = np.cumsum(1 - sorted_labels)[last_of_group]
    positives = max(int(labels.sum()), 1)
    return sorted_scores[last_of_group], tp / (tp + fp), tp / positives


def best_precision_at_recall(labels, scores):
    _, precision, recall = curve(labels, scores)
    eligible = recall >= TARGET_RECALL
    return float(precision[eligible].max()) if eligible.any() else 0.0


def choose_threshold(labels, scores):
    thresholds, precision, recall = curve(labels, scores)
    margin = np.minimum(precision - TARGET_PRECISION, recall - TARGET_RECALL)
    eligible = (precision >= TARGET_PRECISION) & (recall >= TARGET_RECALL)
    if not eligible.any():
        return None
    best = np.flatnonzero(eligible)[np.argmax(margin[eligible])]
    return float(thresholds[best])


def scores_at(labels, alarms):
    tp = int((alarms & (labels == 1)).sum())
    fp = int((alarms & (labels == 0)).sum())
    fn = int((~alarms & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(precision=round(precision, 4), recall=round(recall, 4), f1=round(f1, 4), tp=tp, fp=fp, fn=fn,
                rows=int(len(labels)), base_rate=round(float(labels.mean()), 4))


def recall_by_fault_type(labels, alarms, fault_type):
    result = {}
    for name in sorted(set(fault_type[labels == 1])):
        mask = (labels == 1) & (fault_type == name)
        result[name] = round(float(alarms[mask].mean()), 4)
    return result


def prepare(path):
    reconstructed = pumps_data.reconstruct(pumps_data.load(path))
    features, labels, meta = pumps_data.prediction_rows(reconstructed, HORIZONS.values())
    shape = dict(rows=int(len(reconstructed)), trajectories=int(reconstructed['trajectory'].max() + 1),
                 normal_rows=int((reconstructed['trajectory'] < 0).sum()), prediction_rows=int(len(features)))
    return features, labels, meta, shape


def run(primary, replications, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    features, labels, meta, shape = prepare(primary)
    part = pumps_data.split(meta)
    rule = pumps_data.rule_alarm(features)
    fault_type = meta['fault_type'].fillna('').to_numpy()
    report = dict(spec='spec_pumps_v1.txt', primary=dict(file=primary.name, **shape,
                  split={p: int((part == p).sum()) for p in ('train', 'validation', 'test')}), horizons={})
    models = {}
    for name, minutes in HORIZONS.items():
        y = labels[minutes].astype(np.int64)
        model = HistGradientBoostingClassifier(random_state=0)
        model.fit(features[part == 'train'], y[part == 'train'])
        models[name] = model
        valid = part == 'validation'
        valid_scores = model.predict_proba(features[valid])[:, 1]
        threshold = choose_threshold(y[valid], valid_scores)
        model_valid = scores_at(y[valid], valid_scores >= threshold) if threshold is not None else None
        rule_valid = scores_at(y[valid], rule[valid])
        qualified = (threshold is not None and rule_valid['base_rate'] <= MAX_BASE_RATE
                     and model_valid['f1'] > rule_valid['f1'])
        entry = dict(validation=dict(best_precision_at_recall_0_5=round(best_precision_at_recall(y[valid], valid_scores), 4),
                                     threshold=threshold, model=model_valid, rule=rule_valid), qualified=bool(qualified))
        if qualified:
            test = part == 'test'
            alarms = model.predict_proba(features[test])[:, 1] >= threshold
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
        rep_rule = pumps_data.rule_alarm(rep_features)
        rep_fault_type = rep_meta['fault_type'].fillna('').to_numpy()
        entry = dict(**rep_shape)
        for name, minutes in HORIZONS.items():
            if not report['horizons'][name]['qualified']:
                continue
            y = rep_labels[minutes].astype(np.int64)
            alarms = models[name].predict_proba(rep_features)[:, 1] >= report['horizons'][name]['validation']['threshold']
            entry[name] = dict(model=scores_at(y, alarms), rule=scores_at(y, rep_rule),
                               recall_by_fault_type=recall_by_fault_type(y, alarms, rep_fault_type))
        report['replication'][path.name] = entry
    joblib.dump(models, out_dir / 'hgb_pumps.joblib')
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
