import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
import eval_units as eu
import experiment_v2 as ex
import final_v2
import run_v2 as rv

TARGETS = ('failure', 'incident')
FILTERS = dict(fit="split IN ('train','validation') AND as_of < TIMESTAMP '2026-01-01'", test="split = 'test' AND as_of >= TIMESTAMP '2026-01-01'")


def class_labels(connection, labels_path, columns, part):
    raw = connection.execute(f'SELECT channel_id, as_of, label_failure, label_incident FROM read_parquet(?) WHERE label IS NOT NULL AND {FILTERS[part]} '
                             'ORDER BY channel_id, as_of', [str(labels_path)]).fetchnumpy()
    if not (np.array_equal(np.asarray(raw['channel_id']).astype(str), columns['channel_id']) and np.array_equal(np.asarray(raw['as_of']).astype('datetime64[us]'), columns['as_of'])):
        raise ValueError('label rows do not match examples')
    return {name: np.ma.filled(np.ma.asarray(raw['label_' + name]).astype(np.float64), np.nan) for name in TARGETS}


def days(as_of):
    return int((as_of.max() - as_of.min()).astype('timedelta64[D]').astype(np.int64)) + 1


def period_report(labels, scores, alarm_24h, codes, as_of, channel_threshold, object_threshold, incidents):
    rule = (alarm_24h > 0).astype(np.float64)
    table = eu.object_days(codes, as_of, labels, scores, alarm_24h)
    return dict(channel_day=eu.unit_report(labels, scores, rule, channel_threshold, days(as_of)),
                object_day=eu.unit_report(table['label'], table['score'], table['rule'], object_threshold, days(as_of)),
                incident=eu.incident_report(*incidents, table, object_threshold),
                incident_rule_alarm_24h=eu.incident_report(*incidents, dict(keys=table['keys'], score=table['rule']), 1.0)), table


def run_target(connection, name, data, episodes, directory, output):
    columns, test, matrix, test_matrix, codes, test_codes, code = (data[key] for key in ('columns', 'test', 'matrix', 'test_matrix', 'codes', 'test_codes', 'code'))
    y, y_test = data['labels'][name], data['test_labels'][name]
    known, known_test = ~np.isnan(y), ~np.isnan(y_test)
    (train_a, eval_a), (train_b, eval_b) = (tuple(mask & known for mask in ex.fold_masks(columns['as_of'], fold)) for fold in ('A', 'B'))
    labels = np.where(known, y, 0).astype(np.int64)
    scores_a = ex.make_model('hgb').fit(matrix[train_a], labels[train_a]).predict_proba(matrix[eval_a])[:, 1]
    model = ex.make_model('hgb').fit(matrix[train_b], labels[train_b])
    scores_b = model.predict_proba(matrix[eval_b])[:, 1]
    channel_2024, _ = ex.max_f1_threshold(labels[eval_a], scores_a)
    table_2024 = eu.object_days(codes[eval_a], columns['as_of'][eval_a], labels[eval_a], scores_a, columns['alarm_count_24h'][eval_a])
    object_2024, _ = ex.max_f1_threshold(table_2024['label'], table_2024['score'])
    incidents_2025 = eu.incident_keys(connection, episodes, directory, code, '2025-01-01', '2026-01-01')
    year_2025, table_2025 = period_report(labels[eval_b], scores_b, columns['alarm_count_24h'][eval_b], codes[eval_b], columns['as_of'][eval_b], channel_2024, object_2024, incidents_2025)
    channel_2025, _ = ex.max_f1_threshold(labels[eval_b], scores_b)
    object_2025, _ = ex.max_f1_threshold(table_2025['label'], table_2025['score'])
    labels_test = y_test[known_test].astype(np.int64)
    scores_test = model.predict_proba(test_matrix[known_test])[:, 1]
    as_of_test = test['as_of'][known_test]
    incidents_test = eu.incident_keys(connection, episodes, directory, code, '2026-01-01', str(as_of_test.max().astype('datetime64[D]') + np.timedelta64(1, 'D')))
    test_2026, _ = period_report(labels_test, scores_test, test['alarm_count_24h'][known_test], test_codes[known_test], as_of_test, channel_2025, object_2025, incidents_test)
    joblib.dump(model, output / f'hgb_{name}.joblib')
    return dict(rows=dict(train_a=int(train_a.sum()), eval_2024=int(eval_a.sum()), train_b=int(train_b.sum()), eval_2025=int(eval_b.sum()), test=int(known_test.sum()),
                          unknown_boundary_dropped=int((~known).sum() + (~known_test).sum())),
                thresholds=dict(channel_2024=channel_2024, object_2024=object_2024, channel_2025=channel_2025, object_2025=object_2025),
                selection_2024=ex.evaluate(labels[eval_a], scores_a, channel_2024), year_2025=year_2025, test_2026=test_2026)


def main(examples, labels_path, failure_episodes, incident_episodes, directory, output):
    if output.exists():
        raise FileExistsError(str(output))
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns, test = rv.load(connection, examples), final_v2.load(connection, examples, 'test')
        codes, code = eu.object_codes(connection, directory, columns['channel_id'])
        data = dict(columns=columns, test=test, matrix=ex.matrix_v2(columns, True), test_matrix=ex.matrix_v2(test, True), codes=codes,
                    test_codes=eu.object_codes(connection, directory, test['channel_id'])[0], code=code,
                    labels=class_labels(connection, labels_path, columns, 'fit'), test_labels=class_labels(connection, labels_path, test, 'test'))
        output.mkdir(parents=True)
        targets = {name: run_target(connection, name, data, episodes, directory, output) for name, episodes in (('failure', failure_episodes), ('incident', incident_episodes))}
    digest = lambda path: hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    result = dict(input_sha256=digest(examples), labels_sha256=digest(labels_path), targets=targets,
                  flags=dict(features='v2', object_features=False, weather=False, threshold_2025='max F1 on 2024 with the fold A model', threshold_test='max F1 on 2025 with the fold B model',
                             test_evaluation=True, test_note='first look for these targets, same 2026 period as run-002'))
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:7]))
