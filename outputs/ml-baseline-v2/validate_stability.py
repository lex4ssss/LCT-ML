import json
from pathlib import Path
import sys
import duckdb
import numpy as np
import experiment_v2 as ex
import explore_v7 as e7
import explore_v8 as e8
import explore_v11 as e11
import final_v2
import run_v2 as rv
import validate_objects as vo

REPLICATES = 1000


def precision_recall(labels, alarms):
    tp = (alarms & (labels == 1)).sum(axis=-1)
    predicted, positives = alarms.sum(axis=-1), (labels == 1).sum(axis=-1)
    return tp / np.maximum(predicted, 1), tp / np.maximum(positives, 1)


def by_month(labels, alarms, day):
    month = day.astype('datetime64[M]')
    table = {}
    for value in np.unique(month):
        mask = month == value
        table[str(value)] = vo.scored(labels[mask], alarms[mask].astype(np.float64), 0.5)
    return table


def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return [round(float(low), 4), round(float(high), 4)]


def bootstrap(labels, alarms, groups, seed=0):
    rng = np.random.default_rng(seed)
    rows_ci, objects_ci = [], []
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    for _ in range(REPLICATES):
        pick = rng.integers(0, len(labels), len(labels))
        rows_ci.append(precision_recall(labels[pick], alarms[pick]))
        chosen = np.concatenate([members[g] for g in rng.choice(unique, len(unique))])
        objects_ci.append(precision_recall(labels[chosen], alarms[chosen]))
    rows_ci, objects_ci = np.array(rows_ci), np.array(objects_ci)
    return dict(rows=dict(precision=interval(rows_ci[:, 0]), recall=interval(rows_ci[:, 1])),
                objects=dict(precision=interval(objects_ci[:, 0]), recall=interval(objects_ci[:, 1])))


def alerts_per_day(alarms, day):
    days = np.unique(day)
    per_day = np.array([alarms[day == value].sum() for value in days])
    return dict(days=int(len(days)), mean=round(float(per_day.mean()), 2), median=float(np.median(per_day)), max=int(per_day.max()))


def validate_target(connection, columns, test_columns, directory, episodes, hours):
    matrix, history, _, _, day, present, label, _ = vo.frames(connection, columns, directory, episodes, hours)
    train, evaluation = e8.masks(day, present, hours)
    model = ex.make_model('hgb').fit(matrix[train], label[train])
    threshold, _ = e8.margin_threshold(label[evaluation], model.predict_proba(matrix[evaluation])[:, 1])
    week_threshold, week_chosen_by = vo.fixed_threshold(label[evaluation], history[evaluation][:, 1])
    t_matrix, t_history, _, t_objects, t_day, t_present, t_label, t_rule = vo.frames(connection, test_columns, directory, episodes, hours)
    rows = t_present & (t_day <= t_day[t_present].max() - (hours - 24) * e7.HOUR)
    labels, day, objects = t_label[rows], t_day[rows], t_objects[rows]
    alarms = model.predict_proba(t_matrix[rows])[:, 1] >= threshold
    rule_alarms = t_rule[rows] >= 1.0
    week_alarms = t_history[rows][:, 1] >= week_threshold
    return dict(total=vo.scored(labels, alarms.astype(np.float64), 0.5), months=by_month(labels, alarms, day),
                bootstrap_95=bootstrap(labels, alarms, objects), replicates=REPLICATES,
                alerts_per_day=alerts_per_day(alarms, day),
                rule_alarm_24h=dict(total=vo.scored(labels, rule_alarms.astype(np.float64), 0.5), months=by_month(labels, rule_alarms, day)),
                target_starts_7d=dict(threshold=float(week_threshold), chosen_by=week_chosen_by,
                                      total=vo.scored(labels, week_alarms.astype(np.float64), 0.5), months=by_month(labels, week_alarms, day)))


if __name__ == '__main__':
    examples, directory, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    report = {}
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        test_columns = final_v2.load(connection, examples, 'test')
        for name, episodes, hours in e11.parse_targets(sys.argv[4:]):
            report[name] = validate_target(connection, columns, test_columns, directory, episodes, hours)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
