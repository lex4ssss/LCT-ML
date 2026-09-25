import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
import experiment_v2 as ex

YEARS = {'validation': ("TIMESTAMP '2025-01-01'", "TIMESTAMP '2026-01-01'"), 'test': ("TIMESTAMP '2026-01-01'", "TIMESTAMP '2100-01-01'")}


def sha256(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def load(connection, examples, part):
    start, end = YEARS[part]
    query = ('SELECT channel_id, as_of, label, ' + ', '.join(ex.V2_FEATURES) + f" FROM read_parquet(?) WHERE label IS NOT NULL AND split = '{part}' "
             f'AND as_of >= {start} AND as_of < {end} ORDER BY channel_id, as_of')
    raw = connection.execute(query, [str(examples)]).fetchnumpy()
    columns = {name: np.ma.filled(np.ma.asarray(raw[name]).astype(np.float64), np.nan) for name in ex.V2_FEATURES}
    columns['as_of'] = np.asarray(raw['as_of']).astype('datetime64[us]')
    columns['label'] = np.asarray(raw['label']).astype(np.int64)
    columns['channel_id'] = np.asarray(raw['channel_id']).astype(str)
    return columns


def scores(model_path, columns):
    return joblib.load(model_path).predict_proba(ex.matrix_v2(columns, True))[:, 1]


def decide(examples, model_path, output):
    if output.exists():
        raise FileExistsError(str(output))
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = load(connection, examples, 'validation')
    labels = columns['label']
    threshold, f1 = ex.max_f1_threshold(labels, scores(model_path, columns))
    decision = dict(model='hgb', model_path=str(model_path), model_sha256=sha256(model_path), examples_sha256=sha256(examples), features=ex.V2_FEATURES + ['month', 'weekday'],
                    trained_on='as_of < 2024-12-31', threshold=threshold, threshold_selection='max row F1 on 2025 validation with this model', validation_f1=f1,
                    probability_output='raw hgb score; out-of-time reliability checked on 2025 with the fold A model', validation_rows=int(len(labels)),
                    test_evaluated=False)
    output.write_text(json.dumps(decision, indent=2) + '\n')


def coverage(connection, episodes, columns, alerted, last_as_of):
    connection.execute("CREATE TEMP TABLE onsets AS SELECT channel_id, date_trunc('day', first_alarm_at - INTERVAL '1 microsecond') AS as_of FROM read_parquet(?) "
                       "WHERE first_alarm_at > TIMESTAMP '2026-01-01' AND first_alarm_at <= ?::TIMESTAMP + INTERVAL '1 day'", [str(episodes), last_as_of])
    connection.execute('CREATE TEMP TABLE eligible AS SELECT unnest(?) AS channel_id, unnest(?)::TIMESTAMP AS as_of, unnest(?) AS alerted',
                       [columns['channel_id'].tolist(), columns['as_of'].astype(object).tolist(), alerted.tolist()])
    total, matched, hit = connection.execute('SELECT count(*), count(e.channel_id), count(*) FILTER(WHERE e.alerted) FROM onsets o LEFT JOIN eligible e USING(channel_id, as_of)').fetchone()
    return dict(episodes_total=int(total), episodes_with_eligible_snapshot=int(matched), episodes_in_alerted_windows=int(hit))


def test(examples, episodes, decision_path, output):
    if output.exists():
        raise FileExistsError(str(output))
    decision = json.loads(decision_path.read_text())
    if decision['model_sha256'] != sha256(decision['model_path']) or decision['examples_sha256'] != sha256(examples):
        raise ValueError('decision does not match files')
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = load(connection, examples, 'test')
        probability = scores(decision['model_path'], columns)
        labels, common = columns['label'], columns['event_count_24h'] > 0
        alarm_rule = (columns['alarm_count_24h'] > 0).astype(np.float64)
        report = dict(decision=decision, rows=int(len(labels)), first_as_of=str(columns['as_of'].min()), last_as_of=str(columns['as_of'].max()),
                      hgb=dict(all=ex.evaluate(labels, probability, decision['threshold']), common=ex.evaluate(labels[common], probability[common], decision['threshold'])),
                      rule_alarm_24h=ex.evaluate(labels, alarm_rule, 1.0), reliability=ex.reliability(labels, probability),
                      coverage=coverage(connection, episodes, columns, probability >= decision['threshold'], columns['as_of'].max().astype(object)),
                      flags=dict(test_evaluation=True, retraining=False, threshold_changed=False))
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'decide':
        decide(*map(Path, sys.argv[2:5]))
    elif sys.argv[1] == 'test':
        test(*map(Path, sys.argv[2:6]))
    else:
        raise ValueError('mode')
