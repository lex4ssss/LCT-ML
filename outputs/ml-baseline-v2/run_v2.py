import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
import experiment_v2 as ex

TRAINABLE = ('logistic_v1', 'logistic_v2', 'hgb')
RULES = {'rule_alarm_24h': 'alarm_count_24h', 'rule_alarm_7d': 'alarm_count_7d'}


def load(connection, examples):
    query = ('SELECT channel_id, as_of, label, ' + ', '.join(ex.V2_FEATURES) + ' FROM read_parquet(?) '
             "WHERE label IS NOT NULL AND split IN ('train','validation') AND as_of < TIMESTAMP '2026-01-01' ORDER BY channel_id, as_of")
    raw = connection.execute(query, [str(examples)]).fetchnumpy()
    columns = {name: np.ma.filled(np.ma.asarray(raw[name]).astype(np.float64), np.nan) for name in ex.V2_FEATURES}
    columns['as_of'] = np.asarray(raw['as_of']).astype('datetime64[us]')
    columns['label'] = np.asarray(raw['label']).astype(np.int64)
    columns['channel_id'] = np.asarray(raw['channel_id']).astype(str)
    return columns


def fitted_scores(kind, columns, train, common):
    if kind == 'logistic_v1':
        matrix = ex.matrix_v1(columns)
        model = ex.make_model(kind).fit(matrix[train & common], columns['label'][train & common])
        scores = np.zeros(len(matrix))
        scores[common] = model.predict_proba(matrix[common])[:, 1]
        return model, scores
    matrix = ex.matrix_v2(columns, kind == 'hgb')
    model = ex.make_model(kind).fit(matrix[train], columns['label'][train])
    return model, model.predict_proba(matrix)[:, 1]


def subsets(labels, scores, rows, common, threshold):
    return {'all': ex.evaluate(labels[rows], scores[rows], threshold), 'common': ex.evaluate(labels[rows & common], scores[rows & common], threshold)}


def calibration(labels, scores_a, eval_a, eval_b):
    isotonic = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(scores_a[eval_a], labels[eval_a])
    calibrated = isotonic.predict(scores_a[eval_b])
    return dict(brier_before=float(brier_score_loss(labels[eval_b], scores_a[eval_b])), brier_after=float(brier_score_loss(labels[eval_b], calibrated)),
                reliability_before=ex.reliability(labels[eval_b], scores_a[eval_b]), reliability_after=ex.reliability(labels[eval_b], calibrated))


def coverage(connection, episodes, columns, eval_b, scores_b, threshold):
    connection.execute('CREATE OR REPLACE TEMP TABLE onsets AS SELECT channel_id, date_trunc(\'day\', first_alarm_at - INTERVAL \'1 microsecond\') AS as_of FROM read_parquet(?) '
                       "WHERE first_alarm_at > TIMESTAMP '2025-01-01' AND first_alarm_at <= TIMESTAMP '2026-01-01'", [str(episodes)])
    connection.execute('CREATE OR REPLACE TEMP TABLE eligible AS SELECT unnest(?) AS channel_id, unnest(?)::TIMESTAMP AS as_of, unnest(?) AS alerted',
                       [columns['channel_id'][eval_b].tolist(), columns['as_of'][eval_b].astype(object).tolist(), (scores_b[eval_b] >= threshold).tolist()])
    total, matched, alerted = connection.execute('SELECT count(*), count(e.channel_id), count(*) FILTER(WHERE e.alerted) FROM onsets o LEFT JOIN eligible e USING(channel_id, as_of)').fetchone()
    return dict(episodes_total=int(total), episodes_with_eligible_snapshot=int(matched), episodes_in_alerted_windows=int(alerted))


def main(examples, episodes, output):
    if output.exists():
        raise FileExistsError(str(output))
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = load(connection, examples)
        labels, common = columns['label'], columns['event_count_24h'] > 0
        masks = {fold: ex.fold_masks(columns['as_of'], fold) for fold in ('A', 'B')}
        (train_a, eval_a), (train_b, eval_b) = masks['A'], masks['B']
        candidates, models = {}, {}
        for name in tuple(RULES) + TRAINABLE:
            if name in RULES:
                scores_a = scores_b = (columns[RULES[name]] > 0).astype(np.float64)
            else:
                _, scores_a = fitted_scores(name, columns, train_a, common)
                models[name], scores_b = fitted_scores(name, columns, train_b, common)
            threshold, _ = ex.max_f1_threshold(labels[eval_a], scores_a[eval_a])
            report = dict(threshold_A=threshold, selection_2024=ex.evaluate(labels[eval_a], scores_a[eval_a], threshold),
                          frozen=subsets(labels, scores_a, eval_b, common, threshold), retrained=subsets(labels, scores_b, eval_b, common, threshold),
                          coverage=coverage(connection, episodes, columns, eval_b, scores_b, threshold))
            if name in TRAINABLE:
                report['calibration'] = calibration(labels, scores_a, eval_a, eval_b)
            candidates[name] = report
    rows = {fold: dict(train=int(train.sum()), eval=int(evaluation.sum())) for fold, (train, evaluation) in masks.items()}
    digest = hashlib.file_digest(examples.open('rb'), 'sha256').hexdigest()
    result = dict(input_sha256=digest, rows=rows, candidates=candidates,
                  flags=dict(test_evaluation=False, training=True, calibration_fit_on='2024', threshold_selected_on='2024'))
    output.mkdir(parents=True)
    for name, model in models.items():
        joblib.dump(model, output / (name + '.joblib'))
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:4]))
