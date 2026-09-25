import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
from exp_metrics_core import recall_at_precision
import experiment_v2 as ex
import final_v2
import run_v2 as rv

INCIDENT_GAP_MINUTES = 60
UNKNOWN_OBJECT = 'unknown'


def object_codes(connection, directory, channel_ids):
    mapping = dict(connection.execute('SELECT ид_канала_данных, ид_объект FROM read_csv(?, all_varchar=true)', [str(directory)]).fetchall())
    names = sorted(set(mapping.values())) + [UNKNOWN_OBJECT]
    code = {name: index for index, name in enumerate(names)}
    unique, inverse = np.unique(channel_ids, return_inverse=True)
    return np.array([code[mapping.get(channel, UNKNOWN_OBJECT)] for channel in unique], dtype=np.int64)[inverse], code


def day_keys(codes, as_of):
    return codes * 1_000_000 + np.asarray(as_of).astype('datetime64[D]').astype(np.int64)


def object_days(codes, as_of, labels, scores, alarm_24h):
    keys, inverse = np.unique(day_keys(codes, as_of), return_inverse=True)
    label, score, rule = np.zeros(len(keys), np.int64), np.full(len(keys), -np.inf), np.zeros(len(keys))
    np.maximum.at(label, inverse, labels)
    np.maximum.at(score, inverse, scores)
    np.maximum.at(rule, inverse, (alarm_24h > 0).astype(np.float64))
    return dict(keys=keys, label=label, score=score, rule=rule)


def incident_keys(connection, episodes, directory, code, start, end):
    rows = connection.execute(f"""
        WITH e AS (SELECT coalesce(d.ид_объект, '{UNKNOWN_OBJECT}') AS object_id, first_alarm_at FROM read_parquet(?) e
                   LEFT JOIN read_csv(?, all_varchar=true) d ON e.channel_id = d.ид_канала_данных
                   WHERE first_alarm_at > ?::TIMESTAMP AND first_alarm_at <= ?::TIMESTAMP),
        m AS (SELECT *, CASE WHEN first_alarm_at - lag(first_alarm_at) OVER w <= INTERVAL {INCIDENT_GAP_MINUTES} MINUTE THEN 0 ELSE 1 END AS new_incident
              FROM e WINDOW w AS (PARTITION BY object_id ORDER BY first_alarm_at)),
        g AS (SELECT object_id, first_alarm_at, sum(new_incident) OVER (PARTITION BY object_id ORDER BY first_alarm_at ROWS UNBOUNDED PRECEDING) AS incident FROM m)
        SELECT object_id, date_trunc('day', min(first_alarm_at) - INTERVAL 1 MICROSECOND) AS as_of, count(*) AS episodes FROM g GROUP BY object_id, incident""",
        [str(episodes), str(directory), start, end]).fetchall()
    codes = np.array([code[row[0]] for row in rows], dtype=np.int64)
    as_of = np.array([row[1] for row in rows], dtype='datetime64[us]')
    return day_keys(codes, as_of), np.array([row[2] for row in rows], dtype=np.int64)


def unit_report(labels, scores, rule, threshold, days):
    model = ex.evaluate(labels, scores, threshold)
    model['alerts_per_day'] = (model['tp'] + model['fp']) / days
    model['recall_at_precision_0_7'] = recall_at_precision(labels, scores, 0.7)
    return dict(model=model, always_alert_precision=float(labels.mean()), rule_alarm_24h=ex.evaluate(labels, rule, 1.0))


def incident_report(keys, sizes, table, threshold):
    eligible, alerted = np.isin(keys, table['keys']), np.isin(keys, table['keys'][table['score'] >= threshold])
    return dict(incidents=int(len(keys)), episodes=int(sizes.sum()), single_channel_share=float((sizes == 1).mean()), with_eligible_object_day=int(eligible.sum()),
                in_alerted_object_day=int(alerted.sum()), recall=float(alerted.mean()), recall_among_eligible=float(alerted[eligible].mean()))


def period(codes, columns, mask, scores):
    return object_days(codes[mask], columns['as_of'][mask], columns['label'][mask], scores[mask], columns['alarm_count_24h'][mask])


def main(examples, episodes, directory, decision_path, output):
    if output.exists():
        raise FileExistsError(str(output))
    decision = json.loads(decision_path.read_text())
    model_path = decision_path.parent / decision['model_path']
    if hashlib.file_digest(model_path.open('rb'), 'sha256').hexdigest() != decision['model_sha256']:
        raise ValueError('model file does not match decision.json')
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns, test = rv.load(connection, examples), final_v2.load(connection, examples, 'test')
        codes, code = object_codes(connection, directory, columns['channel_id'])
        test_codes, _ = object_codes(connection, directory, test['channel_id'])
        (train_a, eval_a), (_, eval_b) = ex.fold_masks(columns['as_of'], 'A'), ex.fold_masks(columns['as_of'], 'B')
        matrix = ex.matrix_v2(columns, True)
        model_b = joblib.load(model_path)
        scores_a, scores_b = np.zeros(len(matrix)), np.zeros(len(matrix))
        scores_a[eval_a] = ex.make_model('hgb').fit(matrix[train_a], columns['label'][train_a]).predict_proba(matrix[eval_a])[:, 1]
        scores_b[eval_b] = model_b.predict_proba(matrix[eval_b])[:, 1]
        scores_test = model_b.predict_proba(ex.matrix_v2(test, True))[:, 1]
        channel_threshold_2024, _ = ex.max_f1_threshold(columns['label'][eval_a], scores_a[eval_a])
        table_2024, table_2025 = period(codes, columns, eval_a, scores_a), period(codes, columns, eval_b, scores_b)
        table_test = object_days(test_codes, test['as_of'], test['label'], scores_test, test['alarm_count_24h'])
        object_threshold_2024, _ = ex.max_f1_threshold(table_2024['label'], table_2024['score'])
        object_threshold_2025, _ = ex.max_f1_threshold(table_2025['label'], table_2025['score'])
        test_days = int((test['as_of'].max() - test['as_of'].min()).astype('timedelta64[D]').astype(np.int64)) + 1
        incidents_2025 = incident_keys(connection, episodes, directory, code, '2025-01-01', '2026-01-01')
        incidents_test = incident_keys(connection, episodes, directory, code, '2026-01-01', str(test['as_of'].max().astype('datetime64[D]') + np.timedelta64(1, 'D')))
    rule_2025 = (columns['alarm_count_24h'][eval_b] > 0).astype(np.float64)
    rule_test = (test['alarm_count_24h'] > 0).astype(np.float64)
    result = dict(
        channel_threshold_2024=channel_threshold_2024, object_threshold_2024=object_threshold_2024, object_threshold_2025=object_threshold_2025,
        year_2025=dict(channel_day=unit_report(columns['label'][eval_b], scores_b[eval_b], rule_2025, channel_threshold_2024, 365),
                       object_day=unit_report(table_2025['label'], table_2025['score'], table_2025['rule'], object_threshold_2024, 365),
                       incident=incident_report(*incidents_2025, table_2025, object_threshold_2024),
                       incident_rule_alarm_24h=incident_report(*incidents_2025, dict(keys=table_2025['keys'], score=table_2025['rule']), 1.0)),
        test_2026=dict(channel_day=unit_report(test['label'], scores_test, rule_test, decision['threshold'], test_days),
                       object_day=unit_report(table_test['label'], table_test['score'], table_test['rule'], object_threshold_2025, test_days),
                       incident=incident_report(*incidents_test, table_test, object_threshold_2025),
                       incident_rule_alarm_24h=incident_report(*incidents_test, dict(keys=table_test['keys'], score=table_test['rule']), 1.0)),
        protocol=dict(model_2025='run-002 hgb, trained on as_of < 2024-12-31', threshold_2025='max F1 on 2024 with the fold A model retrained here',
                      model_test='decision.json model', threshold_test='channel: decision.json; object: max F1 on 2025 object-days with the same model',
                      object_day='max score and any label over eligible channel snapshots of one object', incident_gap_minutes=INCIDENT_GAP_MINUTES,
                      test_second_look=True, retraining_for_test=False))
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:6]))
