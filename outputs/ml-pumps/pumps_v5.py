import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import pumps_data
import pumps_v2
import pumps_v5_features as v5
from pumps_v1 import recall_by_fault_type, scores_at

HORIZONS = {'30min': 30, '1h': 60}
FAULTS = ['Несоосность', 'Засорение фильтра', 'Неизвестная неисправность']
CLASSES = ['normal', *FAULTS]
MLP_ROWS = 1_000_000


class Tree:
    def __init__(self, router, experts):
        self.router, self.experts = router, experts

    def expert_score(self, fault, features):
        expert = self.experts[fault]
        if isinstance(expert, float):
            return np.full(len(features), expert)
        return expert.predict_proba(features)[:, 1]

    def predict_proba(self, features):
        routed = self.router.predict_proba(features)
        index = {name: i for i, name in enumerate(self.router.classes_)}
        probability = np.zeros(len(features))
        for fault in self.experts:
            if fault in index:
                probability += routed[:, index[fault]] * self.expert_score(fault, features)
        return np.column_stack([1 - probability, probability])


def fit_expert(features, labels):
    if labels.min() == labels.max():
        return float(labels[0])
    return HistGradientBoostingClassifier(random_state=0).fit(features, labels)


def fit_tree(router, features, labels, fault_type):
    experts = {fault: fit_expert(features[fault_type == fault], labels[fault_type == fault])
               for fault in FAULTS if (fault_type == fault).any()}
    return Tree(router, experts)


def load_rows(path, stats=None):
    reconstructed = pumps_data.reconstruct(pumps_data.load(path), allow_extra=True)
    return reconstructed if stats is None else v5.prediction_rows(reconstructed, stats, HORIZONS.values())


def stack(parts):
    features = pd.concat([p[0] for p in parts], ignore_index=True)
    labels = {h: np.concatenate([p[1][h] for p in parts]) for h in HORIZONS.values()}
    meta = pd.concat([p[2] for p in parts], ignore_index=True)
    return features, labels, meta


def router_report(router, features, fault_type):
    predicted = router.predict(features)
    return dict(accuracy=round(float(accuracy_score(fault_type, predicted)), 4),
                f1=dict(zip(CLASSES, np.round(f1_score(fault_type, predicted, labels=CLASSES, average=None, zero_division=0), 4).tolist())))


def run(train_paths, validation_path, test_path, v4_run, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    reconstructed = [load_rows(path) for path in train_paths]
    stats = v5.normal_stats(reconstructed)
    train = stack([v5.prediction_rows(frame, stats, HORIZONS.values()) for frame in reconstructed])
    del reconstructed
    valid = load_rows(validation_path, stats)
    features, labels, meta = train
    fault_type = meta['fault_type'].to_numpy()
    router = HistGradientBoostingClassifier(random_state=0).fit(features, fault_type)
    subsample = np.random.default_rng(0).choice(len(features), min(MLP_ROWS, len(features)), replace=False)
    report = dict(spec='spec_pumps_v5.txt', train=[p.name for p in train_paths], validation=validation_path.name, test=test_path.name,
                  train_rows=int(len(features)), horizons={})
    models, chosen = {}, {}
    valid_rule = pumps_data.rule_alarm(valid[0])
    for name, minutes in HORIZONS.items():
        y = labels[minutes].astype(np.int64)
        flat = HistGradientBoostingClassifier(random_state=0).fit(features, y)
        tree = fit_tree(router, features, y, fault_type)
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=0)).fit(
            features.iloc[subsample], y[subsample])
        models[name] = dict(F=flat, T=tree, M=mlp)
        yv = valid[1][minutes].astype(np.int64)
        scores = score_all(models[name], valid[0])
        rule = scores_at(yv, valid_rule)
        variants = {}
        for variant, values in scores.items():
            threshold, recall = pumps_v2.choose_threshold(yv, values, 'recall')
            at = scores_at(yv, values >= threshold) if threshold is not None else None
            variants[variant] = dict(threshold=threshold, recall_at_precision_0_95=recall, model=at,
                                     qualified=bool(threshold is not None and at['f1'] > rule['f1'] and rule['base_rate'] <= pumps_v2.MAX_BASE_RATE))
        qualifiers = [v for v in variants if variants[v]['qualified']]
        report['horizons'][name] = dict(validation=dict(rule=rule, variants=variants))
        if qualifiers:
            chosen[name] = max(qualifiers, key=lambda v: variants[v]['recall_at_precision_0_95'])
            report['horizons'][name]['chosen'] = chosen[name]
    report['validation_router'] = router_report(router, valid[0], valid[2]['fault_type'].to_numpy())
    del train, features, valid
    test = load_rows(test_path, stats)
    test_rule = pumps_data.rule_alarm(test[0])
    test_type = test[2]['fault_type'].to_numpy()
    v4 = v4_scores(test_path, v4_run)
    for name, minutes in HORIZONS.items():
        if name not in chosen:
            continue
        y = test[1][minutes].astype(np.int64)
        variant = chosen[name]
        threshold = report['horizons'][name]['validation']['variants'][variant]['threshold']
        alarms = score_all(models[name], test[0])[variant] >= threshold
        v4_alarms = v4['scores'][name] >= v4['thresholds'][name]
        report['horizons'][name]['test'] = dict(model=scores_at(y, alarms), rule=scores_at(y, test_rule),
                                                recall_by_fault_type=recall_by_fault_type(y, alarms, test_type),
                                                pumps_v4=dict(model=scores_at(v4['labels'][name], v4_alarms),
                                                              recall_by_fault_type=recall_by_fault_type(v4['labels'][name], v4_alarms, v4['fault_type'])))
    report['test_router'] = router_report(router, test[0], test_type)
    joblib.dump(dict(stats=stats, router=router, models={name: models[name] for name in chosen}, chosen=chosen), out_dir / 'pumps_v5_models.joblib')
    (out_dir / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def score_all(fitted, features):
    scores = {variant: model.predict_proba(features)[:, 1] for variant, model in fitted.items()}
    scores['D'] = (scores['T'] + scores['M']) / 2
    return scores


def v4_scores(test_path, v4_run):
    decision = json.loads((v4_run / 'pump-decision.json').read_text())
    fitted = joblib.load(v4_run / decision['model_path'])
    reconstructed = pumps_data.reconstruct(pumps_data.load(test_path), allow_extra=True)
    built = {name: pumps_data.prediction_rows(reconstructed, HORIZONS.values(), feature_set=name) for name in ('a3', 'v3')}
    _, labels, meta = built['a3']
    features = {name: built[name][0] for name in built}
    return dict(scores={name: pumps_v2.predict(fitted[name], features, h['variant']) for name, h in decision['horizons'].items()},
                thresholds={name: h['threshold'] for name, h in decision['horizons'].items()},
                labels={name: labels[HORIZONS[name]].astype(np.int64) for name in HORIZONS},
                fault_type=meta['fault_type'].fillna('').to_numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, nargs='+', required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--v4-run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.train, args.validation, args.test, args.v4_run, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
