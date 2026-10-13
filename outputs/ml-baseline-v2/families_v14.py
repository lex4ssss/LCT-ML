import joblib
import numpy as np
import experiment_v2 as ex
import explore_v7 as e7
import explore_v8 as e8
import telemetry_v12 as telemetry


FAMILIES = ('smoke_temperature', 'water', 'gas', 'access', 'other')
TYPES = ({'Датчик дыма', 'Тепловой датчик', 'Датчик температуры'}, {'Датчик затопления'}, {'Газовый датчик'},
         {'КД Дверь', 'КД Люк', 'КД АВ', '9-секционный люк'})
VARIANTS = ('R', 'H_F', 'HEAD_MAX', 'HEAD_OR')


def family_code(sensor_type):
    return next((index for index, names in enumerate(TYPES) if sensor_type in names), 4)


def channel_mapping(directory, object_code):
    rows = telemetry.registry(directory).drop_duplicates('channel_id', keep='last')
    rows['head'] = [family_code(value) for value in rows['sensor_type']]
    rows['object_code'] = [object_code[value] for value in rows['object_id']]
    return rows


def label_table(connection, examples, episodes, mapping, unknown_code, hours=72):
    connection.register('v14_registry', mapping)
    aggregates = [f'coalesce(max(label) FILTER (WHERE head={index}), 0) AS family_{index}' for index in range(5)]
    query = ('WITH e AS (SELECT channel_id, as_of FROM read_parquet(?) WHERE label IS NOT NULL '
             "AND split IN ('train','validation') AND as_of < TIMESTAMP '2026-01-01'), "
             "k AS (SELECT channel_id, first_alarm_at AS first_at FROM read_parquet(?) WHERE left_boundary='observed_quiet_gap'), "
             "u AS (SELECT channel_id, first_alarm_at AS first_at FROM read_parquet(?) WHERE left_boundary IS DISTINCT FROM 'observed_quiet_gap'), "
             's AS (SELECT coalesce(m.object_code, ?) AS object_code, coalesce(m.head, 4) AS head, e.as_of, '
             "CASE WHEN u.first_at<=e.as_of + ? * INTERVAL '1 hour' THEN NULL "
             "WHEN k.first_at<=e.as_of + ? * INTERVAL '1 hour' THEN 1 ELSE 0 END AS label "
             'FROM e LEFT JOIN v14_registry m ON e.channel_id=m.channel_id '
             'ASOF LEFT JOIN k ON e.channel_id=k.channel_id AND e.as_of<k.first_at '
             'ASOF LEFT JOIN u ON e.channel_id=u.channel_id AND e.as_of<u.first_at) '
             "SELECT object_code * 1000000 + date_diff('day', DATE '1970-01-01', as_of) AS object_key, "
             'count(label)>0 AS present, coalesce(max(label), 0) AS label, ' + ', '.join(aggregates) +
             ' FROM s GROUP BY object_key ORDER BY object_key')
    return connection.execute(query, [str(examples), str(episodes), str(episodes), unknown_code, hours, hours]).fetchdf()


def family_history(connection, episodes, mapping, object_keys, unknown_code):
    connection.register('v14_registry', mapping)
    records = connection.execute('SELECT coalesce(m.object_code, ?) AS object_code, coalesce(m.head, 4) AS head, e.first_alarm_at '
                                 'FROM read_parquet(?) e LEFT JOIN v14_registry m ON e.channel_id=m.channel_id',
                                 [int(unknown_code), str(episodes)]).fetchnumpy()
    object_code = object_keys // 1_000_000
    moments = e8.key_moments(object_keys)
    ends = e7.keys(object_code, moments)
    parts, names = [], []
    for head, family in enumerate(FAMILIES):
        rows = np.asarray(records['head']) == head
        starts = np.sort(e7.keys(np.asarray(records['object_code'])[rows], np.asarray(records['first_alarm_at'])[rows]))
        if not len(starts):
            parts.append(np.column_stack([np.zeros((len(object_keys), 3)), np.full(len(object_keys), np.nan)]))
        else:
            right = np.searchsorted(starts, ends, 'right')
            counts = [right - np.searchsorted(starts, e7.keys(object_code, moments - hours * e7.HOUR), 'right') for hours in e8.WINDOWS_HOURS]
            previous = starts[np.maximum(right - 1, 0)]
            found = (right > 0) & ((previous >> e7.SHIFT) == object_code)
            age = np.where(found, np.minimum((ends - previous) / 3.6e9, e8.AGE_CAP_HOURS), np.nan)
            parts.append(np.column_stack(counts + [age]))
        names += [f'{family}_starts_{hours}h' for hours in e8.WINDOWS_HOURS] + [f'{family}_last_start_age']
    return np.column_stack(parts).astype(float), names


class ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, matrix):
        score = np.full(len(matrix), self.value)
        return np.column_stack([1 - score, score])


def fit_head(matrix, labels):
    classes = np.unique(labels)
    if not len(classes) or np.any(~np.isin(classes, [0, 1])):
        raise ValueError('binary head labels required')
    if len(classes) == 1:
        return ConstantModel(classes[0])
    return ex.make_model('hgb').fit(matrix, labels)


def combine(scores, variant):
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] != 5 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('five finite head probabilities required')
    if variant == 'HEAD_MAX':
        return scores.max(axis=1)
    if variant == 'HEAD_OR':
        return 1 - np.prod(1 - scores, axis=1)
    raise ValueError('combination')


def save_verified(model, path, matrix):
    expected = model.predict_proba(matrix)
    joblib.dump(model, path)
    np.testing.assert_array_equal(joblib.load(path).predict_proba(matrix), expected)
