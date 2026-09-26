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

PUMP_SENSOR = 'Состояние насоса'
PUMP_ON = 'Включен'
PUMP_WINDOWS_HOURS = (24, 168, 720)
FLOOD_WINDOWS_HOURS = (24, 168)
MIN_AP_GAIN = 0.01


def object_parents(connection, objects_directory, code):
    parent = dict(connection.execute('SELECT ид_объект, родитель FROM read_csv(?, all_varchar=true)', [str(objects_directory)]).fetchall())
    count = max(code.values()) + 1
    parents = np.full(count, '', dtype=object)
    for name, index in code.items():
        parents[index] = parent.get(name) or ''
    return parents


def sibling_table(parents):
    groups = {}
    for index, parent in enumerate(parents):
        if parent:
            groups.setdefault(parent, []).append(index)
    width = max([len(members) - 1 for members in groups.values()] + [1])
    table = np.full((len(parents), width), -1, dtype=np.int64)
    for members in groups.values():
        for index in members:
            others = [other for other in members if other != index]
            table[index, :len(others)] = others
    return table


def pump_events(connection, channels_directory, events_glob, code):
    pumps = dict(connection.execute('SELECT ид_канала_данных, ид_объект FROM read_csv(?, all_varchar=true) WHERE тип_датчика = ?',
                                    [str(channels_directory), PUMP_SENSOR]).fetchall())
    raw = connection.execute('SELECT channel_id, occurred_at FROM read_parquet(?) WHERE raw_value = ? AND channel_id IN (SELECT unnest(?))',
                             [events_glob, PUMP_ON, list(pumps)]).fetchnumpy()
    objects = np.array([code[pumps[channel]] for channel in np.asarray(raw['channel_id']).astype(str)], dtype=np.int64)
    has_pumps = np.zeros(max(code.values()) + 1, dtype=np.float64)
    has_pumps[[code[name] for name in set(pumps.values()) if name in code]] = 1.0
    return np.sort(e7.keys(objects, np.asarray(raw['occurred_at']).astype('datetime64[us]'))), has_pumps


def window_counts(sorted_keys, objects, moments, windows_hours):
    end = np.searchsorted(sorted_keys, e7.keys(objects, moments), 'right')
    return [end - np.searchsorted(sorted_keys, e7.keys(objects, moments - hours * e7.HOUR), 'right') for hours in windows_hours]


def own_block(pump_keys, has_pumps, objects, moments):
    counts = window_counts(pump_keys, objects, moments, PUMP_WINDOWS_HOURS)
    daily = counts[2] / 30
    flap = np.divide(counts[0], daily, out=np.zeros(len(objects)), where=daily > 0)
    return np.column_stack(counts + [flap, has_pumps[objects]]).astype(np.float64)


def pump_features(pump_keys, has_pumps, flood_keys, siblings, object_codes, moments):
    own = own_block(pump_keys, has_pumps, object_codes, moments)
    sibling_sum = np.zeros_like(own)
    flood = np.zeros((len(object_codes), len(FLOOD_WINDOWS_HOURS)))
    for slot in range(siblings.shape[1]):
        other = siblings[object_codes, slot]
        valid = other >= 0
        if not valid.any():
            continue
        sibling_sum[valid] += own_block(pump_keys, has_pumps, other[valid], moments[valid])
        flood[valid] += np.column_stack(window_counts(flood_keys, other[valid], moments[valid], FLOOD_WINDOWS_HOURS))
    return np.column_stack([own, sibling_sum, flood])


def flood_keys(connection, episodes, channels_directory):
    raw = connection.execute('SELECT channel_id, first_alarm_at FROM read_parquet(?)', [str(episodes)]).fetchnumpy()
    codes, _ = eu.object_codes(connection, channels_directory, np.asarray(raw['channel_id']).astype(str))
    return np.sort(e7.keys(codes, np.asarray(raw['first_alarm_at']).astype('datetime64[us]')))


def frames(connection, columns, channels_directory, objects_directory, events_glob, episodes, hours):
    codes, code, sensor, type_count = e8.channel_info(connection, channels_directory, columns['channel_id'])
    keys, matrix = e8.aggregate(columns, codes, sensor, type_count)
    labels = e7.horizon_labels(connection, episodes, columns['channel_id'], columns['as_of'], hours)
    present, label, rule = e8.object_labels(keys, eu.day_keys(codes, columns['as_of']), labels, columns['alarm_count_24h'])
    reference = np.column_stack([matrix, e8.target_history(connection, episodes, channels_directory, keys)])
    pump_keys, has_pumps = pump_events(connection, channels_directory, events_glob, code)
    siblings = sibling_table(object_parents(connection, objects_directory, code))
    day = e8.key_moments(keys)
    extra = pump_features(pump_keys, has_pumps, flood_keys(connection, episodes, channels_directory), siblings, keys // 1_000_000, day)
    return reference, np.column_stack([reference, extra]), day, present, label, rule


def compare(labels, reference_scores, variant_scores, rule):
    reference, variant = e8.selection(labels, reference_scores, rule), e8.selection(labels, variant_scores, rule)
    reference['ap'] = float(average_precision_score(labels, reference_scores))
    variant['ap'] = float(average_precision_score(labels, variant_scores))
    gain = variant['ap'] - reference['ap']
    accepted = bool(gain >= MIN_AP_GAIN and variant['precision_at_recall_0_5'] >= reference['precision_at_recall_0_5'] and variant['qualifies'])
    return dict(reference=reference, variant=variant, ap_gain=gain, accepted=accepted)


def fit_2025(connection, columns, paths, hours):
    reference, variant, day, present, label, rule = frames(connection, columns, *paths, hours)
    train, evaluation = e8.masks(day, present, hours)
    models = [ex.make_model('hgb').fit(matrix[train], label[train]) for matrix in (reference, variant)]
    scores = [model.predict_proba(matrix[evaluation])[:, 1] for model, matrix in zip(models, (reference, variant))]
    return models, compare(label[evaluation], *scores, rule[evaluation])


def search(examples, paths, output):
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        columns = rv.load(connection, examples)
        results = []
        for hours in e7.HORIZONS:
            result = dict(horizon_hours=hours, **fit_2025(connection, columns, paths, hours)[1])
            results.append(result)
            print(json.dumps(dict(h=hours, ap_ref=round(result['reference']['ap'], 4), ap_var=round(result['variant']['ap'], 4),
                                  p05_ref=round(result['reference']['precision_at_recall_0_5'], 4),
                                  p05_var=round(result['variant']['precision_at_recall_0_5'], 4), accepted=result['accepted'])), flush=True)
    accepted = [row for row in results if row['accepted']]
    chosen = max(accepted, key=lambda row: row['ap_gain'])['horizon_hours'] if accepted else None
    output.write_text(json.dumps(dict(results=results, chosen_horizon_hours=chosen, test_evaluation=False), indent=2) + '\n')


def confirm(examples, paths, search_report, output):
    if output.exists():
        raise FileExistsError(str(output))
    report = json.loads(search_report.read_text())
    hours = report['chosen_horizon_hours']
    if hours is None:
        raise ValueError('no accepted horizon')
    chosen = next(row for row in report['results'] if row['horizon_hours'] == hours)
    with duckdb.connect(config={'threads': 4, 'memory_limit': '4GB'}) as connection:
        models, again = fit_2025(connection, rv.load(connection, examples), paths, hours)
        if not np.isclose(again['variant']['threshold'], chosen['variant']['threshold']):
            raise ValueError('retrained model does not reproduce the search result')
        reference, variant, day, present, label, rule = frames(connection, final_v2.load(connection, examples, 'test'), *paths, hours)
    rows = present & (day <= day[present].max() - (hours - 24) * e7.HOUR)
    days = int((day[rows].max() - day[rows].min()) / (24 * e7.HOUR)) + 1
    result = dict(horizon_hours=hours, year_2025=chosen, test_object_days=int(rows.sum()), test_last_as_of=str(day[rows].max()),
                  test_variant=eu.unit_report(label[rows], models[1].predict_proba(variant[rows])[:, 1], rule[rows], chosen['variant']['threshold'], days),
                  test_reference=eu.unit_report(label[rows], models[0].predict_proba(reference[rows])[:, 1], rule[rows], chosen['reference']['threshold'], days),
                  protocol='explore-v9: flood object model with pump features, AP gain rule fixed on 2025, test 2026 evaluated once')
    output.mkdir(parents=True)
    joblib.dump(models[1], output / f'hgb_object_flood_pumps_{hours}h.joblib')
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    mode, examples, channels_directory, objects_directory, events_glob, episodes = sys.argv[1:7]
    paths = (Path(channels_directory), Path(objects_directory), events_glob, Path(episodes))
    if mode == 'search':
        search(Path(examples), paths, Path(sys.argv[7]))
    elif mode == 'confirm':
        confirm(Path(examples), paths, Path(sys.argv[7]), Path(sys.argv[8]))
    else:
        raise ValueError('mode')
