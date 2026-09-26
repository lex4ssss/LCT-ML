import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
from sklearn.metrics import average_precision_score
import eval_units as eu
import experiment_v2 as ex
import explore_v7 as e7
import explore_v8 as e8
import final_v2
import run_v2 as rv
from exp_metrics_core import threshold_curve

MIN_AP_GAIN = 0.01
COUNT_WINDOWS_DAYS = (90, 365)
ACTIVE_WINDOWS_DAYS = (7, 30, 90)
DAY = np.timedelta64(1, 'D')


def episode_starts(connection, episodes, directory):
    raw = connection.execute('SELECT channel_id, first_alarm_at FROM read_parquet(?)', [str(episodes)]).fetchnumpy()
    codes, _ = eu.object_codes(connection, directory, np.asarray(raw['channel_id']).astype(str))
    return codes, np.asarray(raw['first_alarm_at']).astype('datetime64[us]')


def rhythm_features(start_codes, start_times, object_codes, days, hours):
    keys = np.sort(e7.keys(start_codes, start_times))
    moment = days.astype('datetime64[us]')

    def count(begin, end, side_begin, side_end):
        return np.searchsorted(keys, e7.keys(object_codes, end), side_end) - np.searchsorted(keys, e7.keys(object_codes, begin), side_begin)

    horizon = hours * e7.HOUR
    columns = [count(moment - w * DAY, moment, 'right', 'right') for w in COUNT_WINDOWS_DAYS]
    columns += [count(moment - w * DAY, moment - w * DAY + horizon, 'left', 'left') for w in (7, 14)]
    start_days = start_times.astype('datetime64[D]').astype(np.int64)
    query_days = days.astype('datetime64[D]').astype(np.int64)
    first = min(start_days.min(), query_days.min()) - 400
    width = max(start_days.max(), query_days.max()) - first + 1
    grid = np.zeros((max(start_codes.max(), object_codes.max()) + 1, width), dtype=np.int64)
    np.add.at(grid, (start_codes, start_days - first), 1)
    active = (grid > 0).astype(np.int64)
    cumulative = np.concatenate([np.zeros((grid.shape[0], 1), np.int64), np.cumsum(active, axis=1)], axis=1)
    index = query_days - first
    columns += [cumulative[object_codes, index] - cumulative[object_codes, index - w] for w in ACTIVE_WINDOWS_DAYS]
    streak = np.zeros_like(active)
    for day in range(1, width):
        streak[:, day] = np.where(active[:, day - 1] > 0, streak[:, day - 1] + 1, 0)
    columns.append(streak[object_codes, index])
    columns.append(sum(grid[object_codes, index - 7 * k] for k in range(1, 5)))
    return np.column_stack(columns).astype(np.float64)


def frames(connection, columns, directory, episodes, hours):
    codes, _, sensor, type_count = e8.channel_info(connection, directory, columns['channel_id'])
    keys, matrix = e8.aggregate(columns, codes, sensor, type_count)
    labels = e7.horizon_labels(connection, episodes, columns['channel_id'], columns['as_of'], hours)
    present, label, rule = e8.object_labels(keys, eu.day_keys(codes, columns['as_of']), labels, columns['alarm_count_24h'])
    reference = np.column_stack([matrix, e8.target_history(connection, episodes, directory, keys)])
    day = e8.key_moments(keys)
    start_codes, start_times = episode_starts(connection, episodes, directory)
    rhythm = rhythm_features(start_codes, start_times, keys // 1_000_000, day, hours)
    return reference, np.column_stack([reference, rhythm]), day, present, label, rule


def precision_threshold(labels, scores, min_recall=0.5):
    thresholds, tp, fp, positives = threshold_curve(labels, scores)
    precision, recall = tp / (tp + fp), tp / positives
    eligible = recall >= min_recall
    return float(thresholds[np.flatnonzero(eligible)[np.argmax(precision[eligible])]])


def compare(labels, reference_scores, rhythm_scores, rule):
    results = {}
    for name, scores in (('R', reference_scores), ('H', rhythm_scores)):
        row = e8.selection(labels, scores, rule)
        row['ap'] = float(average_precision_score(labels, scores))
        row['precision_threshold'] = precision_threshold(labels, scores)
        results[name] = row
    gain = results['H']['ap'] - results['R']['ap']
    accepted = bool(gain >= MIN_AP_GAIN and results['H']['precision_at_recall_0_5'] >= results['R']['precision_at_recall_0_5'] and results['H']['qualifies'])
    return dict(variants=results, ap_gain=gain, accepted=accepted, chosen='H' if accepted else 'R')


def fit(connection, columns, directory, episodes, hours):
    reference, rhythm, day, present, label, rule = frames(connection, columns, directory, episodes, hours)
    train, evaluation = e8.masks(day, present, hours)
    models = dict(R=ex.make_model('hgb').fit(reference[train], label[train]), H=ex.make_model('hgb').fit(rhythm[train], label[train]))
    result = compare(label[evaluation], models['R'].predict_proba(reference[evaluation])[:, 1], models['H'].predict_proba(rhythm[evaluation])[:, 1], rule[evaluation])
    return models, result


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
            _, result = fit(connection, columns, directory, episodes, hours)
            report[name] = dict(episodes=str(episodes), horizon_hours=hours, **result)
            v = result['variants']
            print(json.dumps(dict(target=name, ap=(round(v['R']['ap'], 4), round(v['H']['ap'], 4)),
                                  p05=(round(v['R']['precision_at_recall_0_5'], 4), round(v['H']['precision_at_recall_0_5'], 4)), chosen=result['chosen'])), flush=True)
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
            models, again = fit(connection, columns, directory, episodes, hours)
            chosen = report[name]['chosen']
            saved = report[name]['variants'][chosen]
            if again['chosen'] != chosen or not np.isclose(again['variants'][chosen]['threshold'], saved['threshold']):
                raise ValueError('retrained models do not reproduce the search result')
            reference, rhythm, day, present, label, rule = frames(connection, test_columns, directory, episodes, hours)
            rows = present & (day <= day[present].max() - (hours - 24) * e7.HOUR)
            days = int((day[rows].max() - day[rows].min()) / (24 * e7.HOUR)) + 1
            chosen_scores = models[chosen].predict_proba((rhythm if chosen == 'H' else reference)[rows])[:, 1]
            reference_scores = models['R'].predict_proba(reference[rows])[:, 1]
            result[name] = dict(horizon_hours=hours, chosen=chosen, year_2025=report[name], test_object_days=int(rows.sum()),
                                test_margin=eu.unit_report(label[rows], chosen_scores, rule[rows], saved['threshold'], days),
                                test_precision=eu.unit_report(label[rows], chosen_scores, rule[rows], saved['precision_threshold'], days),
                                test_reference=eu.unit_report(label[rows], reference_scores, rule[rows], report[name]['variants']['R']['threshold'], days))
            joblib.dump(models[chosen], output / f'hgb_object_{name}_{hours}h_{chosen}.joblib')
    (output / 'report.json').write_text(json.dumps(dict(targets=result, protocol='explore-v11, seventh look at test 2026'), indent=2) + '\n')


if __name__ == '__main__':
    mode, examples, directory = sys.argv[1:4]
    if mode == 'search':
        search(Path(examples), Path(directory), Path(sys.argv[4]), sys.argv[5:])
    elif mode == 'confirm':
        confirm(Path(examples), Path(directory), Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6:])
    else:
        raise ValueError('mode')
