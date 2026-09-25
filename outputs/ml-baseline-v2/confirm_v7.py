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

DAY = np.timedelta64(86400 * 10**6, 'us')


def main(examples, episodes, directory, search_report, target, hours, output):
    if output.exists():
        raise FileExistsError(str(output))
    chosen = next(row for row in json.loads(search_report.read_text())['results'] if row['target'] == target and row['horizon_hours'] == hours)
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns, test = rv.load(connection, examples), final_v2.load(connection, examples, 'test')
        labels = e7.horizon_labels(connection, episodes, columns['channel_id'], columns['as_of'], hours)
        known = ~np.isnan(labels)
        train = known & (columns['as_of'] < e7.VALIDATION[0] - hours * e7.HOUR)
        evaluation = known & (columns['as_of'] >= e7.VALIDATION[0]) & (columns['as_of'] + hours * e7.HOUR <= e7.VALIDATION[1])
        y = np.where(known, labels, 0).astype(np.int64)
        matrix = ex.matrix_v2(columns, True)
        model = ex.make_model('hgb').fit(matrix[train], y[train])
        codes, _ = eu.object_codes(connection, directory, columns['channel_id'])
        scores = model.predict_proba(matrix[evaluation])[:, 1]
        table = eu.object_days(codes[evaluation], columns['as_of'][evaluation], y[evaluation], scores, columns['alarm_count_24h'][evaluation])
        precision_2025, threshold = e7.precision_at_recall(table['label'], table['score'], 0.5)
        if not np.isclose(threshold, chosen['object_day']['threshold']) or not np.isclose(precision_2025, chosen['object_day']['precision_at_recall_0_5']):
            raise ValueError('retrained model does not reproduce the search result')
        test_labels = e7.horizon_labels(connection, episodes, test['channel_id'], test['as_of'], hours)
        complete = test['as_of'] <= test['as_of'].max() - 2 * DAY
        rows = complete & ~np.isnan(test_labels)
        test_codes, _ = eu.object_codes(connection, directory, test['channel_id'])
        test_scores = model.predict_proba(ex.matrix_v2(test, True)[rows])[:, 1]
        test_table = eu.object_days(test_codes[rows], test['as_of'][rows], test_labels[rows].astype(np.int64), test_scores, test['alarm_count_24h'][rows])
    days = int((test['as_of'][rows].max() - test['as_of'][rows].min()) / DAY) + 1
    result = dict(target=target, horizon_hours=hours, unit='object_day', threshold_from_2025=threshold, precision_at_recall_0_5_2025=precision_2025,
                  test_object_day=eu.unit_report(test_table['label'], test_table['score'], test_table['rule'], threshold, days),
                  test_rows=int(rows.sum()), test_last_as_of=str(test['as_of'][rows].max()),
                  protocol='model trained on as_of < 2025-01-01 minus horizon, threshold fixed on 2025, test 2026 evaluated once')
    output.mkdir(parents=True)
    joblib.dump(model, output / f'hgb_{target}_{hours}h.joblib')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), sys.argv[5], int(sys.argv[6]), Path(sys.argv[7]))
