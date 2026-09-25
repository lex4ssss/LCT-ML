import json
from pathlib import Path
import sys
import duckdb
import numpy as np
import eval_units as eu
import experiment_v2 as ex
from exp_metrics_core import confusion, threshold_curve
import run_v2 as rv

HORIZONS = (24, 72, 168)
HOUR = np.timedelta64(3600 * 10**6, 'us')
EPOCH = np.datetime64('2018-01-01', 'us')
VALIDATION = (np.datetime64('2025-01-01', 'us'), np.datetime64('2026-01-01', 'us'))
SHIFT = 48


def keys(channel_codes, moments):
    return (channel_codes.astype(np.int64) << SHIFT) + (np.asarray(moments).astype('datetime64[us]') - EPOCH).astype(np.int64)


def horizon_labels(connection, episodes, channel_ids, as_of, hours):
    raw = connection.execute('SELECT channel_id, first_alarm_at, left_boundary FROM read_parquet(?)', [str(episodes)]).fetchnumpy()
    names = np.unique(np.concatenate([channel_ids, np.asarray(raw['channel_id']).astype(str)]))
    row_codes, episode_codes = np.searchsorted(names, channel_ids), np.searchsorted(names, np.asarray(raw['channel_id']).astype(str))
    episode_keys = keys(episode_codes, raw['first_alarm_at'])
    known = np.sort(episode_keys[np.asarray(raw['left_boundary']) == 'observed_quiet_gap'])
    unknown = np.sort(episode_keys[np.asarray(raw['left_boundary']) != 'observed_quiet_gap'])
    start, end = keys(row_codes, as_of), keys(row_codes, as_of + hours * HOUR)
    count = lambda table: np.searchsorted(table, end, 'right') - np.searchsorted(table, start, 'right')
    labels = (count(known) > 0).astype(np.float64)
    labels[count(unknown) > 0] = np.nan
    return labels


def precision_at_recall(labels, scores, target):
    thresholds, tp, fp, positives = threshold_curve(labels, scores)
    eligible = tp / positives >= target
    precision = tp / (tp + fp)
    best = np.flatnonzero(eligible)[np.argmax(precision[eligible])]
    return float(precision[best]), float(thresholds[best])


def unit_metrics(labels, scores, rule):
    precision, threshold = precision_at_recall(labels, scores, 0.5)
    at = confusion(labels, scores, threshold)
    ruled = confusion(labels, rule, 1.0)
    return dict(rows=int(len(labels)), base_rate=float(labels.mean()), precision_at_recall_0_5=precision, threshold=threshold, recall=at['recall'], f1=at['f1'],
                rule_precision=ruled['precision'], rule_recall=ruled['recall'], rule_f1=ruled['f1'],
                qualifies=bool(precision >= 0.7 and labels.mean() <= 0.35 and at['f1'] > ruled['f1']))


def main(examples, directory, output, *targets):
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        object_code, _ = eu.object_codes(connection, directory, columns['channel_id'])
        matrix, as_of = ex.matrix_v2(columns, True), columns['as_of']
        results = []
        for target in targets:
            name, episodes = target.split('=')
            for hours in HORIZONS:
                labels = horizon_labels(connection, Path(episodes), columns['channel_id'], as_of, hours)
                known = ~np.isnan(labels)
                train = known & (as_of < VALIDATION[0] - hours * HOUR)
                evaluation = known & (as_of >= VALIDATION[0]) & (as_of + hours * HOUR <= VALIDATION[1])
                y = np.where(known, labels, 0).astype(np.int64)
                scores = ex.make_model('hgb').fit(matrix[train], y[train]).predict_proba(matrix[evaluation])[:, 1]
                rule = (columns['alarm_count_24h'][evaluation] > 0).astype(np.float64)
                table = eu.object_days(object_code[evaluation], as_of[evaluation], y[evaluation], scores, columns['alarm_count_24h'][evaluation])
                row = dict(target=name, horizon_hours=hours, channel_day=unit_metrics(y[evaluation], scores, rule),
                           object_day=unit_metrics(table['label'], table['score'], table['rule']))
                results.append(row)
                print(json.dumps(dict(target=name, h=hours, ch=round(row['channel_day']['precision_at_recall_0_5'], 3), ch_base=round(row['channel_day']['base_rate'], 3),
                                      obj=round(row['object_day']['precision_at_recall_0_5'], 3), obj_base=round(row['object_day']['base_rate'], 3),
                                      q=[row['channel_day']['qualifies'], row['object_day']['qualifies']])), flush=True)
    output.write_text(json.dumps(dict(results=results, test_evaluation=False), indent=2) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), *sys.argv[4:])
