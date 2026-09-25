import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
import eval_units as eu
import experiment_v2 as ex
import run_v2 as rv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-dataset-v2'))
from object_features import CATEGORIES, OBJECT_FEATURES

DIRECTORY_COLUMNS = dict(sensor_type='тип_датчика', system_type='тип_инж_системы')


def vocabulary(connection, directory):
    return {name: sorted(value for (value,) in connection.execute(f'SELECT DISTINCT {column} FROM read_csv(?, all_varchar=true) WHERE {column} IS NOT NULL', [str(directory)]).fetchall())
            for name, column in DIRECTORY_COLUMNS.items()}


def codes(values, known):
    lookup = {value: float(index) for index, value in enumerate(known)}
    unique, inverse = np.unique(np.array(['' if value is None else value for value in values], dtype=object).astype(str), return_inverse=True)
    return np.array([lookup.get(value, np.nan) for value in unique])[inverse]


def load_extra(connection, features, directory, columns):
    query = ('SELECT channel_id, as_of, ' + ', '.join(OBJECT_FEATURES + CATEGORIES) + " FROM read_parquet(?) WHERE label IS NOT NULL AND split IN ('train','validation') "
             "AND as_of < TIMESTAMP '2026-01-01' ORDER BY channel_id, as_of")
    raw = connection.execute(query, [str(features)]).fetchnumpy()
    if not (np.array_equal(np.asarray(raw['channel_id']).astype(str), columns['channel_id']) and np.array_equal(np.asarray(raw['as_of']).astype('datetime64[us]'), columns['as_of'])):
        raise ValueError('feature rows do not match examples')
    numeric = [np.ma.filled(np.ma.asarray(raw[name]).astype(np.float64), np.nan) for name in OBJECT_FEATURES]
    known = vocabulary(connection, directory)
    categorical = [codes(np.ma.filled(np.ma.asarray(raw[name], dtype=object), None), known[name]) for name in CATEGORIES]
    return np.column_stack(numeric + categorical)


def make_model(width):
    mask = np.zeros(width, dtype=bool)
    mask[-len(CATEGORIES):] = True
    return ex.make_model('hgb').set_params(categorical_features=mask)


def main(examples, features, episodes, directory, baseline_report, output):
    if output.exists():
        raise FileExistsError(str(output))
    baseline = json.loads(baseline_report.read_text())
    digest = hashlib.file_digest(examples.open('rb'), 'sha256').hexdigest()
    if digest != baseline['input_sha256']:
        raise ValueError('examples differ from the baseline run')
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        matrix = np.column_stack([ex.matrix_v2(columns, True), load_extra(connection, features, directory, columns)])
        labels, common = columns['label'], columns['event_count_24h'] > 0
        masks = {fold: ex.fold_masks(columns['as_of'], fold) for fold in ('A', 'B')}
        (train_a, eval_a), (train_b, eval_b) = masks['A'], masks['B']
        scores_a = make_model(matrix.shape[1]).fit(matrix[train_a], labels[train_a]).predict_proba(matrix)[:, 1]
        model = make_model(matrix.shape[1]).fit(matrix[train_b], labels[train_b])
        scores_b = model.predict_proba(matrix)[:, 1]
        threshold, _ = ex.max_f1_threshold(labels[eval_a], scores_a[eval_a])
        report = dict(threshold_A=threshold, selection_2024=ex.evaluate(labels[eval_a], scores_a[eval_a], threshold),
                      frozen=rv.subsets(labels, scores_a, eval_b, common, threshold), retrained=rv.subsets(labels, scores_b, eval_b, common, threshold),
                      coverage=rv.coverage(connection, episodes, columns, eval_b, scores_b, threshold), calibration=rv.calibration(labels, scores_a, eval_a, eval_b))
        object_code, _ = eu.object_codes(connection, directory, columns['channel_id'])
        table_2024, table_2025 = eu.period(object_code, columns, eval_a, scores_a), eu.period(object_code, columns, eval_b, scores_b)
        object_threshold, _ = ex.max_f1_threshold(table_2024['label'], table_2024['score'])
        report['object_day_2025'] = eu.unit_report(table_2025['label'], table_2025['score'], table_2025['rule'], object_threshold, 365)
    rows = {fold: dict(train=int(train.sum()), eval=int(evaluation.sum())) for fold, (train, evaluation) in masks.items()}
    result = dict(input_sha256=digest, features_sha256=hashlib.file_digest(features.open('rb'), 'sha256').hexdigest(), rows=rows,
                  added_features=OBJECT_FEATURES + CATEGORIES, rows_missing_object=int(np.isnan(matrix[:, -len(CATEGORIES) - 1]).sum()),
                  candidates=dict(hgb_objects=report), baseline=dict(source=baseline_report.name, hgb=baseline['candidates']['hgb']),
                  flags=dict(test_evaluation=False, training=True, weather=False, object_features=True, categorical=CATEGORIES, calibration_fit_on='2024', threshold_selected_on='2024'))
    output.mkdir(parents=True)
    joblib.dump(model, output / 'hgb_objects.joblib')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:7]))
