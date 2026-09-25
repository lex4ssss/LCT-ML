import hashlib
import json
from pathlib import Path
import sys
import duckdb
import joblib
import numpy as np
import experiment_v2 as ex
import run_v2 as rv

WEATHER_FEATURES = ['temp_mean_24h', 'temp_min_24h', 'temp_max_24h', 'freeze_thaw_72h', 'pressure_last', 'pressure_delta_24h', 'wind_max_24h']
HOUR = np.timedelta64(3600 * 10**6, 'us')
AVAILABILITY_LAG = 3 * HOUR
READINGS_PER_HOUR = 1 / 3


def read_weather(connection, path):
    raw = connection.execute('SELECT datetime, temperature, pressure, wind_speed FROM read_csv_auto(?) ORDER BY datetime', [str(path)]).fetchnumpy()
    weather = {name: np.ma.filled(np.ma.asarray(raw[name]).astype(np.float64), np.nan) for name in ('temperature', 'pressure', 'wind_speed')}
    weather['datetime'] = np.asarray(raw['datetime']).astype('datetime64[us]')
    return weather


def window(times, cutoff, hours, per_hour=READINGS_PER_HOUR):
    start, end = np.searchsorted(times, cutoff - hours * HOUR, 'right'), np.searchsorted(times, cutoff, 'right')
    return slice(start, end) if end - start == round(hours * per_hour) else None


def at(times, values, moment):
    index = np.searchsorted(times, moment)
    return values[index] if index < len(times) and times[index] == moment else np.nan


def features_at(weather, moment, per_hour=READINGS_PER_HOUR):
    times, temperature, pressure = weather['datetime'], weather['temperature'], weather['pressure']
    cutoff = moment - AVAILABILITY_LAG
    day, three_days = window(times, cutoff, 24, per_hour), window(times, cutoff, 72, per_hour)
    values = [np.nan] * len(WEATHER_FEATURES)
    if day is not None:
        values[0:3] = temperature[day].mean(), temperature[day].min(), temperature[day].max()
        values[6] = weather['wind_speed'][day].max()
    if three_days is not None:
        frozen = temperature[three_days] <= 0
        values[3] = float(np.count_nonzero(frozen[1:] != frozen[:-1]))
    values[4] = at(times, pressure, cutoff)
    values[5] = values[4] - at(times, pressure, cutoff - 24 * HOUR)
    return values


def weather_matrix(weather, as_of, features=features_at, names=WEATHER_FEATURES):
    moments, inverse = np.unique(np.asarray(as_of).astype('datetime64[us]'), return_inverse=True)
    table = np.array([features(weather, moment) for moment in moments], dtype=np.float64).reshape(len(moments), len(names))
    return table[inverse]


def main(examples, episodes, weather_csv, baseline_report, output, reader=read_weather, features=features_at, names=WEATHER_FEATURES,
         flags=dict(precipitation=False, weather_lag_hours=3), candidate='hgb_weather'):
    if output.exists():
        raise FileExistsError(str(output))
    baseline = json.loads(baseline_report.read_text())
    digest = hashlib.file_digest(examples.open('rb'), 'sha256').hexdigest()
    if digest != baseline['input_sha256']:
        raise ValueError('examples differ from the baseline run')
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        extra = weather_matrix(reader(connection, weather_csv), columns['as_of'], features, names)
        matrix = np.column_stack([ex.matrix_v2(columns, True), extra])
        labels, common = columns['label'], columns['event_count_24h'] > 0
        masks = {fold: ex.fold_masks(columns['as_of'], fold) for fold in ('A', 'B')}
        (train_a, eval_a), (train_b, eval_b) = masks['A'], masks['B']
        scores_a = ex.make_model('hgb').fit(matrix[train_a], labels[train_a]).predict_proba(matrix)[:, 1]
        model = ex.make_model('hgb').fit(matrix[train_b], labels[train_b])
        scores_b = model.predict_proba(matrix)[:, 1]
        threshold, _ = ex.max_f1_threshold(labels[eval_a], scores_a[eval_a])
        report = dict(threshold_A=threshold, selection_2024=ex.evaluate(labels[eval_a], scores_a[eval_a], threshold),
                      frozen=rv.subsets(labels, scores_a, eval_b, common, threshold), retrained=rv.subsets(labels, scores_b, eval_b, common, threshold),
                      coverage=rv.coverage(connection, episodes, columns, eval_b, scores_b, threshold), calibration=rv.calibration(labels, scores_a, eval_a, eval_b))
    rows = {fold: dict(train=int(train.sum()), eval=int(evaluation.sum())) for fold, (train, evaluation) in masks.items()}
    result = dict(input_sha256=digest, weather_sha256=hashlib.file_digest(weather_csv.open('rb'), 'sha256').hexdigest(), rows=rows,
                  weather_features=names, rows_missing_weather=int(np.isnan(extra).any(axis=1).sum()),
                  candidates={candidate: report}, baseline=dict(source=baseline_report.name, hgb=baseline['candidates']['hgb']),
                  flags=dict(test_evaluation=False, training=True, **flags, calibration_fit_on='2024', threshold_selected_on='2024'))
    output.mkdir(parents=True)
    joblib.dump(model, output / (candidate + '.joblib'))
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:6]))
