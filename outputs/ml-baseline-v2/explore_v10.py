import itertools
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eval_units as eu
import experiment_v2 as ex
import explore_v7 as e7
import explore_v8 as e8
import final_v2
import run_v2 as rv
from exp_metrics_core import threshold_curve

GRID = dict(learning_rate=(0.05, 0.1), max_leaf_nodes=(31, 63), min_samples_leaf=(50, 200), max_iter=(300, 600))
INNER_END = np.datetime64('2024-01-01', 'us')
MIN_AP_GAIN = 0.01
STRICT_RECALL = 0.6


def hgb(params):
    return HistGradientBoostingClassifier(l2_regularization=1.0, early_stopping=False, random_state=0, **params)


def mlp():
    return make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, random_state=0))


def grid_points():
    names = list(GRID)
    return [dict(zip(names, values)) for values in itertools.product(*GRID.values())]


def inner_masks(day, present, hours):
    return present & (day < INNER_END - hours * e7.HOUR), present & (day >= INNER_END) & (day + hours * e7.HOUR <= e7.VALIDATION[0])


def tune(matrix, day, present, label, hours):
    train, evaluation = inner_masks(day, present, hours)
    scored = []
    for params in grid_points():
        model = hgb(params).fit(matrix[train], label[train])
        scored.append((float(average_precision_score(label[evaluation], model.predict_proba(matrix[evaluation])[:, 1])), params))
    best = max(scored, key=lambda item: item[0])
    return best[1], [dict(ap_2024=ap, **params) for ap, params in scored]


def strict_threshold(labels, scores):
    thresholds, tp, fp, positives = threshold_curve(labels, scores)
    precision, recall = tp / (tp + fp), tp / positives
    eligible = recall >= STRICT_RECALL
    best = np.flatnonzero(eligible)[np.argmax(precision[eligible])]
    return float(thresholds[best])


def fit_all(matrix, day, present, label, hours):
    params, grid = tune(matrix, day, present, label, hours)
    train, _ = e8.masks(day, present, hours)
    models = dict(R=ex.make_model('hgb'), T=hgb(params), M=mlp())
    for model in models.values():
        model.fit(matrix[train], label[train])
    return models, params, grid


def scores_of(models, matrix):
    scores = {name: model.predict_proba(matrix)[:, 1] for name, model in models.items()}
    scores['E'] = (scores['T'] + scores['M']) / 2
    return scores


def evaluate_2025(labels, scores, rule):
    results = {}
    for name, values in scores.items():
        row = e8.selection(labels, values, rule)
        row['ap'] = float(average_precision_score(labels, values))
        row['strict_threshold'] = strict_threshold(labels, values)
        results[name] = row
    reference = results['R']
    for name in ('T', 'M', 'E'):
        row = results[name]
        row['ap_gain'] = row['ap'] - reference['ap']
        row['accepted'] = bool(row['ap_gain'] >= MIN_AP_GAIN and row['precision_at_recall_0_5'] >= reference['precision_at_recall_0_5'] and row['qualifies'])
    accepted = [name for name in ('T', 'M', 'E') if results[name]['accepted']]
    chosen = max(accepted, key=lambda name: results[name]['ap']) if accepted else 'R'
    return results, chosen


def run_target(connection, columns, directory, episodes, hours):
    matrix, day, present, label, rule = e8.frame(connection, columns, directory, episodes, hours)
    models, params, grid = fit_all(matrix, day, present, label, hours)
    _, evaluation = e8.masks(day, present, hours)
    results, chosen = evaluate_2025(label[evaluation], scores_of(models, matrix[evaluation]), rule[evaluation])
    return models, dict(tuned_params=params, grid=grid, variants=results, chosen=chosen)


def parse_targets(targets):
    parsed = []
    for target in targets:
        name, rest = target.split('=')
        episodes, hours = rest.rsplit(':', 1)
        parsed.append((name, Path(episodes), int(hours)))
    return parsed


def search(examples, directory, output, targets):
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        report = {}
        for name, episodes, hours in parse_targets(targets):
            _, result = run_target(connection, columns, directory, episodes, hours)
            report[name] = dict(episodes=str(episodes), horizon_hours=hours, **result)
            summary = {k: (round(v['ap'], 4), round(v['precision_at_recall_0_5'], 4), v.get('accepted')) for k, v in result['variants'].items()}
            print(json.dumps(dict(target=name, chosen=result['chosen'], params=result['tuned_params'], variants=summary)), flush=True)
    output.write_text(json.dumps(dict(targets=report, test_evaluation=False), indent=2) + '\n')


def confirm(examples, directory, search_report, output, targets):
    if output.exists():
        raise FileExistsError(str(output))
    report = json.loads(search_report.read_text())['targets']
    result = {}
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        test_columns = final_v2.load(connection, examples, 'test')
        output.mkdir(parents=True)
        for name, episodes, hours in parse_targets(targets):
            models, again = run_target(connection, columns, directory, episodes, hours)
            chosen = report[name]['chosen']
            saved = report[name]['variants'][chosen]
            if again['chosen'] != chosen or not np.isclose(again['variants'][chosen]['threshold'], saved['threshold']):
                raise ValueError('retrained models do not reproduce the search result')
            matrix, day, present, label, rule = e8.frame(connection, test_columns, directory, episodes, hours)
            rows = present & (day <= day[present].max() - (hours - 24) * e7.HOUR)
            days = int((day[rows].max() - day[rows].min()) / (24 * e7.HOUR)) + 1
            scores = scores_of(models, matrix[rows])
            result[name] = dict(horizon_hours=hours, chosen=chosen, year_2025=saved, test_object_days=int(rows.sum()),
                                test_margin=eu.unit_report(label[rows], scores[chosen], rule[rows], saved['threshold'], days),
                                test_strict=eu.unit_report(label[rows], scores[chosen], rule[rows], saved['strict_threshold'], days),
                                test_reference=eu.unit_report(label[rows], scores['R'], rule[rows], report[name]['variants']['R']['threshold'], days))
            joblib.dump(models, output / f'object_models_{name}_{hours}h.joblib')
    (output / 'report.json').write_text(json.dumps(dict(targets=result, protocol='explore-v10, sixth look at test 2026'), indent=2) + '\n')


if __name__ == '__main__':
    mode, examples, directory = sys.argv[1:4]
    if mode == 'search':
        search(Path(examples), Path(directory), Path(sys.argv[4]), sys.argv[5:])
    elif mode == 'confirm':
        confirm(Path(examples), Path(directory), Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6:])
    else:
        raise ValueError('mode')
