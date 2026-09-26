import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
import eval_units as eu
import experiment_v2 as ex
import explore_v7 as e7
import final_v2
import run_v2 as rv
from exp_metrics_core import confusion, recall_at_precision, threshold_curve

SUMS = ['event_count_24h', 'event_count_7d', 'alarm_count_24h', 'alarm_count_7d', 'alarm_count_30d', 'episode_starts_7d', 'episode_starts_30d']
MAXIMA = ['alarm_count_24h', 'alarm_count_7d', 'episode_starts_7d', 'episode_starts_30d']
MINIMA = ['last_event_age_hours', 'last_alarm_age_hours_30d']
ACTIVE = ['alarm_count_24h', 'alarm_count_7d', 'episode_starts_30d']
WINDOWS_HOURS = (24, 168, 720)
AGE_CAP_HOURS = 720


def channel_info(connection, directory, channel_ids):
    codes, code = eu.object_codes(connection, directory, channel_ids)
    mapping = dict(connection.execute('SELECT ид_канала_данных, тип_датчика FROM read_csv(?, all_varchar=true)', [str(directory)]).fetchall())
    type_index = {name: index for index, name in enumerate(sorted(set(mapping.values())))}
    unique, inverse = np.unique(channel_ids, return_inverse=True)
    sensor = np.array([type_index.get(mapping.get(channel), -1) for channel in unique], dtype=np.int64)[inverse]
    return codes, code, sensor, len(type_index)


def key_moments(keys):
    return (keys % 1_000_000).astype('datetime64[D]').astype('datetime64[us]')


def aggregate(columns, codes, sensor, type_count):
    keys, group = np.unique(eu.day_keys(codes, columns['as_of']), return_inverse=True)
    n = len(keys)
    value = {name: np.asarray(columns[name], dtype=np.float64) for name in ex.V2_FEATURES}

    def reduce(function, name, start):
        out = np.full(n, start)
        function.at(out, group, value[name])
        return out

    channels = np.bincount(group, minlength=n).astype(np.float64)
    active = [np.bincount(group, np.nan_to_num(value[name]) > 0, n) for name in ACTIVE]
    typed = sensor >= 0
    cell = group[typed] * type_count + sensor[typed]
    per_type = [np.bincount(cell, weights, n * type_count).reshape(n, type_count) for weights in (None, np.nan_to_num(value['episode_starts_7d'])[typed])]
    order = np.lexsort((np.nan_to_num(value['last_alarm_age_hours_30d'], nan=np.inf), -value['episode_starts_30d'], -value['alarm_count_7d'], group))
    lead = order[np.r_[True, group[order][1:] != group[order][:-1]]]
    day = key_moments(keys).astype('datetime64[D]')
    parts = ([channels] + [np.bincount(group, np.nan_to_num(value[name]), n) for name in SUMS] + [reduce(np.fmax, name, -np.inf) for name in MAXIMA]
             + [reduce(np.fmin, name, np.inf) for name in MINIMA] + active + [count / channels for count in active] + per_type
             + [ex.matrix_v2({name: value[name][lead] for name in ex.V2_FEATURES}, False),
                (day.astype('datetime64[M]').astype(np.int64) % 12 + 1).astype(np.float64), ((day.astype(np.int64) + 3) % 7).astype(np.float64)])
    matrix = np.column_stack(parts)
    matrix[~np.isfinite(matrix)] = np.nan
    return keys, matrix


def target_history(connection, episodes, directory, keys):
    raw = connection.execute('SELECT channel_id, first_alarm_at FROM read_parquet(?)', [str(episodes)]).fetchnumpy()
    episode_codes, _ = eu.object_codes(connection, directory, np.asarray(raw['channel_id']).astype(str))
    starts = np.sort(e7.keys(episode_codes, raw['first_alarm_at']))
    object_code, moment = keys // 1_000_000, key_moments(keys)
    end = e7.keys(object_code, moment)
    right = np.searchsorted(starts, end, 'right')
    counts = [right - np.searchsorted(starts, e7.keys(object_code, moment - hours * e7.HOUR), 'right') for hours in WINDOWS_HOURS]
    previous = starts[np.maximum(right - 1, 0)]
    found = (right > 0) & ((previous >> e7.SHIFT) == object_code)
    age = np.where(found, np.minimum((end - previous) / 3.6e9, AGE_CAP_HOURS), np.nan)
    return np.column_stack(counts + [age]).astype(np.float64)


def object_labels(keys, row_keys, labels, alarm_24h):
    known = ~np.isnan(labels)
    position = np.searchsorted(keys, row_keys[known])
    present, label, rule = np.zeros(len(keys), bool), np.zeros(len(keys), np.int64), np.zeros(len(keys))
    present[position] = True
    np.maximum.at(label, position, labels[known].astype(np.int64))
    np.maximum.at(rule, position, (alarm_24h[known] > 0).astype(np.float64))
    return present, label, rule


def margin_threshold(labels, scores):
    thresholds, tp, fp, positives = threshold_curve(labels, scores)
    margin = np.minimum(tp / (tp + fp) - 0.7, tp / positives - 0.5)
    best = int(np.argmax(margin))
    return float(thresholds[best]), float(margin[best])


def selection(labels, scores, rule):
    threshold, margin = margin_threshold(labels, scores)
    at, ruled = confusion(labels, scores, threshold), confusion(labels, rule, 1.0)
    base = float(labels.mean())
    return dict(rows=int(len(labels)), base_rate=base, precision_at_recall_0_5=e7.precision_at_recall(labels, scores, 0.5)[0], threshold=threshold, margin=margin,
                precision=at['precision'], recall=at['recall'], f1=at['f1'], recall_at_precision_0_7=recall_at_precision(labels, scores, 0.7),
                rule_precision=ruled['precision'], rule_recall=ruled['recall'], rule_f1=ruled['f1'],
                qualifies=bool(margin >= 0 and base <= 0.35 and at['f1'] > ruled['f1']))


def choose(results, targets=None):
    qualified = [row for row in results if row['object_day']['qualifies'] and (targets is None or row['target'] in targets)]
    return max(qualified, key=lambda row: row['object_day']['margin']) if qualified else None


def frame(connection, columns, directory, episodes, hours):
    codes, _, sensor, type_count = channel_info(connection, directory, columns['channel_id'])
    keys, matrix = aggregate(columns, codes, sensor, type_count)
    labels = e7.horizon_labels(connection, episodes, columns['channel_id'], columns['as_of'], hours)
    present, label, rule = object_labels(keys, eu.day_keys(codes, columns['as_of']), labels, columns['alarm_count_24h'])
    return np.column_stack([matrix, target_history(connection, episodes, directory, keys)]), key_moments(keys), present, label, rule


def masks(day, present, hours):
    return present & (day < e7.VALIDATION[0] - hours * e7.HOUR), present & (day >= e7.VALIDATION[0]) & (day + hours * e7.HOUR <= e7.VALIDATION[1])


def fit_2025(connection, columns, directory, episodes, hours):
    matrix, day, present, label, rule = frame(connection, columns, directory, episodes, hours)
    train, evaluation = masks(day, present, hours)
    model = ex.make_model('hgb').fit(matrix[train], label[train])
    return model, selection(label[evaluation], model.predict_proba(matrix[evaluation])[:, 1], rule[evaluation])


def search(examples, directory, output, *targets):
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        results = []
        for target in targets:
            name, episodes = target.split('=')
            for hours in e7.HORIZONS:
                results.append(dict(target=name, episodes=episodes, horizon_hours=hours, object_day=fit_2025(connection, columns, directory, Path(episodes), hours)[1]))
                row = results[-1]['object_day']
                print(json.dumps(dict(target=name, h=hours, p05=round(row['precision_at_recall_0_5'], 3), base=round(row['base_rate'], 3), m=round(row['margin'], 3), q=row['qualifies'])), flush=True)
    chosen = choose(results)
    output.write_text(json.dumps(dict(results=results, chosen=chosen and dict(target=chosen['target'], horizon_hours=chosen['horizon_hours']), test_evaluation=False), indent=2) + '\n')


def confirm(examples, directory, search_report, output, *targets):
    if output.exists():
        raise FileExistsError(str(output))
    chosen = choose(json.loads(search_report.read_text())['results'], targets or None)
    if chosen is None:
        raise ValueError('no qualified variant')
    episodes, hours = Path(chosen['episodes']), chosen['horizon_hours']
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        model, again = fit_2025(connection, rv.load(connection, examples), directory, episodes, hours)
        if not np.isclose(again['threshold'], chosen['object_day']['threshold']) or not np.isclose(again['margin'], chosen['object_day']['margin']):
            raise ValueError('retrained model does not reproduce the search result')
        matrix, day, present, label, rule = frame(connection, final_v2.load(connection, examples, 'test'), directory, episodes, hours)
    rows = present & (day <= day[present].max() - (hours - 24) * e7.HOUR)
    days = int((day[rows].max() - day[rows].min()) / (24 * e7.HOUR)) + 1
    result = dict(target=chosen['target'], horizon_hours=hours, unit='object_day', year_2025=chosen['object_day'],
                  test_object_day=eu.unit_report(label[rows], model.predict_proba(matrix[rows])[:, 1], rule[rows], chosen['object_day']['threshold'], days),
                  test_object_days=int(rows.sum()), test_last_as_of=str(day[rows].max()),
                  protocol='object-level HGB trained on as_of < 2025-01-01 minus horizon, margin threshold fixed on 2025, test 2026 evaluated once; test already opened in run-002, run-006, run-007')
    output.mkdir(parents=True)
    joblib.dump(model, output / f"hgb_object_{chosen['target']}_{hours}h.joblib")
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'search':
        search(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), *sys.argv[5:])
    elif sys.argv[1] == 'confirm':
        confirm(*map(Path, sys.argv[2:6]), *sys.argv[6:])
    else:
        raise ValueError('mode')
