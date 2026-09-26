import json
from pathlib import Path
import sys
import duckdb
import numpy as np
from sklearn.metrics import average_precision_score
import eval_units as eu
import experiment_v2 as ex
import explore_v7 as e7
import explore_v8 as e8
import explore_v11 as e11
import final_v2
import run_v2 as rv

TOP_OBJECTS = 5


def frames(connection, columns, directory, episodes, hours):
    codes, _, sensor, type_count = e8.channel_info(connection, directory, columns['channel_id'])
    keys, matrix = e8.aggregate(columns, codes, sensor, type_count)
    labels = e7.horizon_labels(connection, episodes, columns['channel_id'], columns['as_of'], hours)
    present, label, rule = e8.object_labels(keys, eu.day_keys(codes, columns['as_of']), labels, columns['alarm_count_24h'])
    history = e8.target_history(connection, episodes, directory, keys)
    day = e8.key_moments(keys)
    start_codes, start_times = e11.episode_starts(connection, episodes, directory)
    rhythm = e11.rhythm_features(start_codes, start_times, keys // 1_000_000, day, hours)
    return np.column_stack([matrix, history]), history, rhythm, keys // 1_000_000, day, present, label, rule


def naive_scores(history, rhythm):
    return {'target starts 24 h': history[:, 0], 'target starts 7 d': history[:, 1], 'target starts 30 d': history[:, 2],
            'active days of 7': rhythm[:, 4], 'active days of 30': rhythm[:, 5]}


def fixed_threshold(labels, scores):
    threshold, margin = e8.margin_threshold(labels, scores)
    if margin >= 0:
        return threshold, 'margin'
    thresholds, tp, fp, positives = e8.threshold_curve(labels, scores)
    f1 = 2 * tp / (tp + fp + positives)
    return float(thresholds[int(np.argmax(f1))]), 'max_f1'


def scored(labels, scores, threshold):
    alarms = scores >= threshold
    tp, fp = int((alarms & (labels == 1)).sum()), int((alarms & (labels == 0)).sum())
    positives = int(labels.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positives if positives else 0.0
    return dict(precision=round(precision, 4), recall=round(recall, 4), f1=round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
                alerts=int(tp + fp), rows=int(len(labels)), base_rate=round(float(labels.mean()), 4))


def per_object(labels, alarms, objects):
    table = []
    for obj in np.unique(objects):
        mask = objects == obj
        tp, fp = int((alarms & mask & (labels == 1)).sum()), int((alarms & mask & (labels == 0)).sum())
        table.append(dict(object=int(obj), alerts=tp + fp, tp=tp, positives=int((mask & (labels == 1)).sum())))
    return sorted(table, key=lambda row: -row['alerts'])


def validate_target(connection, columns, test_columns, directory, episodes, hours):
    matrix, history, rhythm, objects, day, present, label, rule = frames(connection, columns, directory, episodes, hours)
    train, evaluation = e8.masks(day, present, hours)
    model = ex.make_model('hgb').fit(matrix[train], label[train])
    scores_2025 = model.predict_proba(matrix[evaluation])[:, 1]
    threshold, _ = e8.margin_threshold(label[evaluation], scores_2025)
    shuffled = np.random.default_rng(0).permutation(label[train])
    shuffled_model = ex.make_model('hgb').fit(matrix[train], shuffled)
    shuffle_check = dict(ap_2025_true_labels=round(float(average_precision_score(label[evaluation], scores_2025)), 4),
                         ap_2025_shuffled_labels=round(float(average_precision_score(label[evaluation], shuffled_model.predict_proba(matrix[evaluation])[:, 1])), 4),
                         base_rate_2025=round(float(label[evaluation].mean()), 4))
    naive_2025 = naive_scores(history[evaluation], rhythm[evaluation])
    naive_thresholds = {name: fixed_threshold(label[evaluation], values) for name, values in naive_2025.items()}
    t_matrix, t_history, t_rhythm, t_objects, t_day, t_present, t_label, t_rule = frames(connection, test_columns, directory, episodes, hours)
    rows = t_present & (t_day <= t_day[t_present].max() - (hours - 24) * e7.HOUR)
    y, obj = t_label[rows], t_objects[rows]
    test_scores = model.predict_proba(t_matrix[rows])[:, 1]
    alarms = test_scores >= threshold
    naive_test = naive_scores(t_history[rows], t_rhythm[rows])
    objects_table = per_object(y, alarms, obj)
    top = np.array([row['object'] for row in objects_table[:TOP_OBJECTS]])
    rest = ~np.isin(obj, top)
    precisions = [row['tp'] / row['alerts'] for row in objects_table if row['alerts'] >= 10]
    return dict(
        model_test=scored(y, test_scores, threshold),
        rule_alarm_24h_test=scored(y, t_rule[rows], 1.0),
        naive_test={name: dict(threshold=naive_thresholds[name][0], chosen_by=naive_thresholds[name][1], **scored(y, values, naive_thresholds[name][0]))
                    for name, values in naive_test.items()},
        objects=dict(with_alerts=int(sum(row['alerts'] > 0 for row in objects_table)), total=int(len(objects_table)),
                     top_alert_share=round(sum(row['alerts'] for row in objects_table[:TOP_OBJECTS]) / max(int(alarms.sum()), 1), 4),
                     top_objects=objects_table[:TOP_OBJECTS], without_top=scored(y[rest], test_scores[rest], threshold),
                     median_precision_objects_with_10_alerts=round(float(np.median(precisions)), 4) if precisions else None,
                     objects_with_10_alerts=len(precisions)),
        shuffle_check=shuffle_check)


if __name__ == '__main__':
    examples, directory, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    report = {}
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        test_columns = final_v2.load(connection, examples, 'test')
        for name, episodes, hours in e11.parse_targets(sys.argv[4:]):
            report[name] = validate_target(connection, columns, test_columns, directory, episodes, hours)
            print(json.dumps({name: dict(model=report[name]['model_test'], shuffle=report[name]['shuffle_check'])}), flush=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
