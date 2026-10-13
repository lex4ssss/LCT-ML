import csv
from decimal import Decimal
import re
import numpy as np
import pandas as pd


PICKET = re.compile(r'(?:ПК|пикет)\s*([0-9]+(?:[.,][0-9]+)?)', re.IGNORECASE)
NUMERIC_NAMES = ['rows', 'clock', 'negative', 'temp_n', 'temp_sum', 'temp_squares', 'temp_min', 'temp_max',
                 'gas_n', 'gas_sum', 'gas_squares', 'gas_min', 'gas_max', 'gas_positive']
NUMERIC_WINDOWS = (1, 7, 30)
DOMAIN_WINDOWS = (24, 168, 720)


def parse_picket(name):
    values = {format(Decimal(value.replace(',', '.')).normalize(), 'f') for value in PICKET.findall(name)}
    return next(iter(values)) if len(values) == 1 else None


def registry(directory):
    with open(directory, encoding='utf-8-sig', newline='') as source:
        records = list(csv.DictReader(source))
    rows = [dict(channel_id=row['ид_канала_данных'], object_id=row['ид_объект'],
                 sensor_type=row['тип_датчика'], picket=parse_picket(row['название_датчика'])) for row in records]
    return pd.DataFrame(rows, columns=['channel_id', 'object_id', 'sensor_type', 'picket'])


def numeric_daily(connection, observations, channel_registry, cutoff):
    connection.register('v12_registry', channel_registry)
    expressions = ['count(*) AS rows',
                   "count(*) FILTER (WHERE starts_with(raw_value, '01.01.1970')) AS clock",
                   'count(*) FILTER (WHERE value < 0) AS negative']
    for prefix, sensor_type in [('temp', 'Датчик температуры'), ('gas', 'Газовый датчик')]:
        condition = "sensor_type = '" + sensor_type + "'"
        expressions += [f'count(value) FILTER (WHERE {condition}) AS {prefix}_n',
                        f'coalesce(sum(value) FILTER (WHERE {condition}), 0) AS {prefix}_sum',
                        f'coalesce(sum(value * value) FILTER (WHERE {condition}), 0) AS {prefix}_squares',
                        f'min(value) FILTER (WHERE {condition}) AS {prefix}_min',
                        f'max(value) FILTER (WHERE {condition}) AS {prefix}_max']
    expressions += ["count(*) FILTER (WHERE sensor_type = 'Газовый датчик' AND value > 0) AS gas_positive"]
    query = ('WITH parsed AS (SELECT channel_id, occurred_at, raw_value, '
             "try_cast(replace(raw_value, ',', '.') AS DOUBLE) AS parsed_value FROM read_parquet(?)), "
             'typed AS (SELECT coalesce(r.object_id, ?) AS object_id, r.sensor_type, p.raw_value, '
             "date_trunc('day', p.occurred_at - INTERVAL '1 microsecond') + INTERVAL '1 day' AS as_of, "
             'CASE WHEN isfinite(p.parsed_value) THEN p.parsed_value END AS value '
             'FROM parsed p LEFT JOIN v12_registry r ON r.channel_id=p.channel_id WHERE p.occurred_at<=?) '
             'SELECT object_id, as_of, ' + ', '.join(expressions) + ' FROM typed GROUP BY object_id, as_of '
             'ORDER BY object_id, as_of')
    paths = [str(path) for path in observations] if isinstance(observations, (list, tuple)) else str(observations)
    return connection.execute(query, [paths, 'unknown', pd.Timestamp(cutoff).to_pydatetime()]).fetchdf()


def numeric_matrix(daily, object_ids, moments):
    names = []
    for window in NUMERIC_WINDOWS:
        names += [f'{prefix}_{stat}_{window}d' for prefix in ['temp', 'gas'] for stat in ['n', 'mean', 'std', 'min', 'max']]
        names += [f'{stat}_{window}d' for stat in ['gas_positive_share', 'negative_share', 'clock_share', 'records']]
    names += [f'{prefix}_{stat}' for prefix in ['temp', 'gas'] for stat in ['change_1_7', 'change_7_30', 'standardized_change_1_7']]
    result = np.full((len(moments), len(names)), np.nan)
    query_days = np.asarray(moments).astype('datetime64[D]').astype(np.int64)
    object_ids = np.asarray(object_ids).astype(str)
    for object_id in np.unique(object_ids):
        indices = np.flatnonzero(object_ids == object_id)
        rows = daily[daily['object_id'] == object_id]
        if rows.empty:
            continue
        row_days = rows['as_of'].to_numpy().astype('datetime64[D]').astype(np.int64)
        first = min(row_days.min(), query_days[indices].min()) - 31
        last = max(row_days.max(), query_days[indices].max())
        width = int(last - first + 1)
        grid = pd.DataFrame(0.0, index=np.arange(width), columns=NUMERIC_NAMES)
        for name in NUMERIC_NAMES:
            if name.endswith(('_min', '_max')):
                grid[name] = np.nan
            grid.loc[row_days - first, name] = rows[name].to_numpy(dtype=float)
        parts, summaries = [], {}
        for window in NUMERIC_WINDOWS:
            sums = grid.rolling(window, min_periods=1).sum()
            for prefix in ['temp', 'gas']:
                count = sums[prefix + '_n'].to_numpy()
                mean = np.divide(sums[prefix + '_sum'], count, out=np.full(width, np.nan), where=count > 0)
                squares = np.divide(sums[prefix + '_squares'], count, out=np.full(width, np.nan), where=count > 0)
                deviation = np.where(count == 1, 0.0, np.sqrt(np.maximum(squares - mean * mean, 0)))
                low = grid[prefix + '_min'].rolling(window, min_periods=1).min().to_numpy()
                high = grid[prefix + '_max'].rolling(window, min_periods=1).max().to_numpy()
                parts += [count, mean, deviation, low, high]
                summaries[prefix, window] = mean, deviation
            gas_count = sums['gas_n'].to_numpy()
            total = sums['rows'].to_numpy()
            parts += [np.divide(sums['gas_positive'], gas_count, out=np.full(width, np.nan), where=gas_count > 0),
                      np.divide(sums['negative'], total, out=np.full(width, np.nan), where=total > 0),
                      np.divide(sums['clock'], total, out=np.full(width, np.nan), where=total > 0), total]
        for prefix in ['temp', 'gas']:
            mean1, _ = summaries[prefix, 1]
            mean7, std7 = summaries[prefix, 7]
            mean30, _ = summaries[prefix, 30]
            parts += [mean1 - mean7, mean7 - mean30, (mean1 - mean7) / (std7 + 1e-6)]
        result[indices] = np.column_stack(parts)[query_days[indices] - first]
    result[~np.isfinite(result)] = np.nan
    return result, names


def domain_matrix(columns, channel_registry, object_keys, object_code):
    rows = channel_registry.dropna(subset=['picket']).copy()
    sizes = rows.groupby(['object_id', 'picket']).size()
    domains = sorted(key for key, size in sizes.items() if size >= 2)
    domain_code = {key: index for index, key in enumerate(domains)}
    channel_domain = {row.channel_id: domain_code.get((row.object_id, row.picket), -1) for row in rows.itertuples()}
    channel_ids, inverse = np.unique(columns['channel_id'], return_inverse=True)
    group_code = np.array([channel_domain.get(channel_id, -1) for channel_id in channel_ids], dtype=np.int64)[inverse]
    valid = group_code >= 0
    names = ['domains_present']
    names += [f'domain_{stat}_{window}h' for window in DOMAIN_WINDOWS for stat in
              ['observed_multi', 'confirmed', 'singleton', 'ambiguous', 'max_alarm_channels', 'max_alarm_share']]
    result = np.zeros((len(object_keys), len(names)))
    if not valid.any():
        result[:, 6::6] = np.nan
        return result, names
    day = np.asarray(columns['as_of']).astype('datetime64[D]').astype(np.int64)
    domain_keys, group = np.unique(group_code[valid] * 1_000_000 + day[valid], return_inverse=True)
    domain_objects = np.array([object_code[key[0]] for key in domains], dtype=np.int64)
    row_object_keys = domain_objects[domain_keys // 1_000_000] * 1_000_000 + domain_keys % 1_000_000
    positions = np.searchsorted(object_keys, row_object_keys)
    if np.any(positions >= len(object_keys)) or np.any(object_keys[positions] != row_object_keys):
        raise ValueError('domain object-day not present')
    np.add.at(result[:, 0], positions, 1)
    alarm_names = ['alarm_count_24h', 'alarm_count_7d', 'alarm_count_30d']
    for window_index, (window, alarm_name) in enumerate(zip(DOMAIN_WINDOWS, alarm_names)):
        age = np.asarray(columns['last_event_age_hours'])[valid]
        observed_channels = (age >= 0) & (age < window)
        observed = np.bincount(group, weights=observed_channels, minlength=len(domain_keys))
        active = np.bincount(group, weights=(np.asarray(columns[alarm_name])[valid] > 0) & observed_channels, minlength=len(domain_keys))
        start = 1 + window_index * 6
        indicators = [observed >= 2, active >= 2, (active == 1) & (observed >= 2), (active >= 1) & (observed < 2)]
        for offset, indicator in enumerate(indicators):
            np.add.at(result[:, start + offset], positions, indicator)
        np.maximum.at(result[:, start + 4], positions, active)
        share = np.divide(active, observed, out=np.zeros(len(domain_keys)), where=observed > 0)
        np.maximum.at(result[:, start + 5], positions, share)
        result[result[:, 0] == 0, start + 5] = np.nan
    return result, names
