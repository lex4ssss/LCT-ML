import numpy as np
import pandas as pd
from states_v15 import STATES


WINDOWS = (1, 7, 30)
TYPES = (('temp', 'Датчик температуры'), ('gas', 'Газовый датчик'))
TRANSITIONS = ('switch', 'reversal', 'diverse', 'endpoint_change', 'alarm_entry', 'alarm_to_non_alarm',
               'alarm_to_alarm', 'neispraven_entry', 'incident_entry', 'other_alarm_entry', 'ambiguous')
NUMERIC_STATS = ('recent_channels', 'baseline_channels', 'missing_baseline_channels', 'max_abs', 'mean_abs',
                 'p90_abs', 'max_signed', 'min_signed', 'above_3_share', 'below_minus_3_share')


def feature_names():
    return dict(trajectories=[f'channel_{stat}_{window}d' for window in WINDOWS for stat in TRANSITIONS] +
                [f'channel_{direction}_state_{state}_7d' for direction in ('incoming', 'outgoing') for state in range(16)] +
                [f'channel_last_state_{state}_{stat}' for stat in ('count', 'share') for state in range(19)] +
                [f'channel_last_age_{stat}' for stat in ('min', 'mean', 'max')],
                numeric=[f'channel_{prefix}_{stat}_{window}d' for prefix, _ in TYPES for window in (1, 7) for stat in NUMERIC_STATS])


def daily_channels(connection, observations, mapping, cutoff):
    connection.register('v16_registry', mapping[['channel_id', 'object_id', 'sensor_type']].drop_duplicates('channel_id', keep='last'))
    query = ('WITH encoded AS (SELECT o.channel_id, o.occurred_at, coalesce(r.object_id, ?) AS object_id, r.sensor_type, '
             "date_trunc('day', o.occurred_at - INTERVAL '1 microsecond') + INTERVAL '1 day' AS as_of, "
             'CASE WHEN o.is_alarm IS NULL THEN 18 WHEN NOT o.is_alarm THEN 16 '
             'ELSE coalesce(list_position(?, o.raw_value) - 1, 17) END AS code, '
             "try_cast(replace(o.raw_value, ',', '.') AS DOUBLE) AS value "
             'FROM read_parquet(?) o LEFT JOIN v16_registry r ON r.channel_id=o.channel_id WHERE o.occurred_at<=?), '
             'grouped AS (SELECT channel_id, object_id, sensor_type, as_of, '
             'min(struct_pack(ts := occurred_at, code := code)) AS first_low, '
             'min(struct_pack(ts := occurred_at, code := -code)) AS first_high, '
             'max(struct_pack(ts := occurred_at, code := code)) AS last_high, '
             'max(struct_pack(ts := occurred_at, code := -code)) AS last_low, count(DISTINCT code) AS distinct_codes, '
             "count(value) FILTER (WHERE isfinite(value) AND sensor_type IN (?, ?)) AS numeric_n, "
             'avg(value) FILTER (WHERE isfinite(value) AND abs(value)<=1e150 AND sensor_type IN (?, ?)) AS numeric_mean, '
             'var_pop(value) FILTER (WHERE isfinite(value) AND abs(value)<=1e150 AND sensor_type IN (?, ?)) AS numeric_var, '
             'max(abs(value)) FILTER (WHERE isfinite(value) AND sensor_type IN (?, ?)) AS numeric_scale '
             'FROM encoded GROUP BY channel_id, object_id, sensor_type, as_of) '
             'SELECT channel_id, object_id, sensor_type, as_of, first_low.ts AS first_at, last_high.ts AS last_at, '
             'CASE WHEN first_low.code = -first_high.code THEN first_low.code ELSE 18 END AS first_code, '
             'CASE WHEN last_high.code = -last_low.code THEN last_high.code ELSE 18 END AS last_code, '
             'distinct_codes, numeric_n, numeric_mean, numeric_var, numeric_scale FROM grouped ORDER BY channel_id, as_of')
    paths = [str(path) for path in observations] if isinstance(observations, (tuple, list)) else str(observations)
    types = [sensor_type for _, sensor_type in TYPES]
    cutoff = pd.Timestamp(cutoff).to_pydatetime()
    daily = connection.execute(query, ['unknown', list(STATES), paths, cutoff, *types, *types, *types, *types]).fetchdf()
    large = daily[daily['numeric_scale'] > 1e150]
    if not large.empty:
        connection.register('v16_large_numeric', large[['channel_id', 'as_of', 'numeric_scale']])
        query = ("WITH parsed AS (SELECT channel_id, date_trunc('day', occurred_at - INTERVAL '1 microsecond') + INTERVAL '1 day' AS as_of, "
                 "try_cast(replace(raw_value, ',', '.') AS DOUBLE) AS value FROM read_parquet(?) WHERE occurred_at<=?), "
                 'scaled AS (SELECT p.channel_id, p.as_of, p.value/l.numeric_scale AS value, l.numeric_scale '
                 'FROM parsed p JOIN v16_large_numeric l USING (channel_id, as_of) WHERE isfinite(p.value)) '
                 'SELECT channel_id, as_of, avg(value)*numeric_scale AS numeric_mean, '
                 'var_pop(value)*numeric_scale*numeric_scale AS numeric_var '
                 'FROM scaled GROUP BY channel_id, as_of, numeric_scale')
        replacements = connection.execute(query, [paths, cutoff]).fetchdf().set_index(['channel_id', 'as_of'])
        for column in ('numeric_mean', 'numeric_var'):
            daily.loc[large.index, column] = replacements.loc[pd.MultiIndex.from_frame(large[['channel_id', 'as_of']]), column].to_numpy()
    return daily


def query_index(object_ids, moments):
    moments = np.asarray(moments).astype('datetime64[us]')
    object_ids = np.asarray(object_ids).astype(str)
    if len(moments) != len(object_ids) or np.isnat(moments).any() or np.any(moments != moments.astype('datetime64[D]')):
        raise ValueError('aligned object-day midnight timestamps required')
    days = moments.astype('datetime64[D]').astype(np.int64)
    return days, {object_id: np.flatnonzero(object_ids == object_id) for object_id in np.unique(object_ids)}


def transition_flags(rows):
    last = rows['last_code'].to_numpy(dtype=int)
    first = rows['first_code'].to_numpy(dtype=int)
    observed = rows['last_at'].to_numpy().astype('datetime64[us]').astype(np.int64)
    previous = np.roll(last, 1)
    valid = (last != 18) & (previous != 18) & ((observed - np.roll(observed, 1)) <= 720 * 3600 * 1_000_000)
    valid[0] = False
    switch = valid & (last != previous)
    reversal = switch & np.roll(switch, 1) & (last == np.roll(last, 2))
    reversal[:2] = False
    alarm = last < 16
    old_alarm = previous < 16
    flags = np.column_stack([switch, reversal, rows['distinct_codes'].to_numpy() > 1,
                            (first != last) & (first != 18) & (last != 18), switch & alarm,
                            switch & old_alarm & (last == 16), switch & old_alarm & alarm,
                            switch & (last == 0), switch & (last >= 7) & (last < 16), switch & (last == 17),
                            (first == 18) | (last == 18)])
    by_state = np.column_stack([switch & (codes == state) for codes in (last, previous) for state in range(16)])
    return flags, by_state


def window_sum(values, row_days, query_days, window):
    prefix = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    upper = np.searchsorted(row_days, query_days, side='right')
    lower = np.searchsorted(row_days, query_days - window, side='right')
    return prefix[upper] - prefix[lower]


def trajectory_matrix(daily, object_ids, moments):
    days, indices = query_index(object_ids, moments)
    result = np.zeros((len(days), 106))
    age_min = np.full(len(days), np.inf)
    age_sum = np.zeros(len(days))
    age_max = np.full(len(days), -np.inf)
    for _, rows in daily.groupby('channel_id', sort=False):
        object_id = rows['object_id'].iloc[0]
        selected = indices.get(object_id)
        if selected is None:
            continue
        rows = rows.sort_values('as_of')
        row_days = rows['as_of'].to_numpy().astype('datetime64[D]').astype(np.int64)
        rows = rows[row_days <= days[selected].max()]
        row_days = row_days[row_days <= days[selected].max()]
        if rows.empty:
            continue
        flags, state_flags = transition_flags(rows)
        for index, window in enumerate(WINDOWS):
            result[selected, index * 11:(index + 1) * 11] += window_sum(flags, row_days, days[selected], window)
        result[selected, 33:65] += window_sum(state_flags, row_days, days[selected], 7)
        positions = np.searchsorted(row_days, days[selected], side='right') - 1
        available = positions >= 0
        age = (days[selected] * 86400_000_000 - rows['last_at'].to_numpy().astype('datetime64[us]').astype(np.int64)[np.maximum(positions, 0)]) / 3600_000_000
        if np.any(age[available] < 0):
            raise ValueError('future channel observation')
        active = available & (age < 720)
        selected_active = selected[active]
        codes = rows['last_code'].to_numpy(dtype=int)[positions[active]]
        result[selected_active, 65 + codes] += 1
        age_min[selected_active] = np.minimum(age_min[selected_active], age[active])
        age_sum[selected_active] += age[active]
        age_max[selected_active] = np.maximum(age_max[selected_active], age[active])
    total = result[:, 65:84].sum(axis=1)
    result[:, 84:103] = np.divide(result[:, 65:84], total[:, None], out=np.full((len(days), 19), np.nan), where=total[:, None] > 0)
    result[:, 103] = np.where(total > 0, age_min, np.nan)
    result[:, 104] = np.divide(age_sum, total, out=np.full(len(days), np.nan), where=total > 0)
    result[:, 105] = np.where(total > 0, age_max, np.nan)
    return result


def channel_numeric_scores(rows, query_days, recent_days):
    row_days = rows['as_of'].to_numpy().astype('datetime64[D]').astype(np.int64)
    selected = (row_days <= query_days.max()) & (rows['numeric_n'].to_numpy() > 0)
    if not np.any(selected):
        return np.zeros(len(query_days), dtype=bool), np.full(len(query_days), np.nan)
    rows = rows[selected]
    row_days = row_days[selected]
    first = min(int(row_days.min()), int(query_days.min()) - 29)
    width = int(query_days.max() - first + 1)
    counts = np.zeros(width, dtype=np.longdouble)
    means = np.zeros(width, dtype=np.longdouble)
    variances = np.zeros(width, dtype=np.longdouble)
    positions = row_days - first
    counts[positions] = rows['numeric_n'].to_numpy()
    anchor = np.longdouble(rows['numeric_mean'].iloc[0])
    means[positions] = rows['numeric_mean'].to_numpy(dtype=np.longdouble) - anchor
    variances[positions] = rows['numeric_var'].to_numpy(dtype=np.longdouble)
    grid = query_days[:, None] - first - np.arange(29, -1, -1)[None, :]
    present = grid >= 0
    grid = np.maximum(grid, 0)
    n = np.where(present, counts[grid], 0)
    m = np.where(n > 0, means[grid], 0)
    v = np.where(n > 0, variances[grid], 0)
    recent_n = n[:, -recent_days:].sum(axis=1)
    baseline_n = n[:, :-recent_days].sum(axis=1)
    recent_mean = np.divide((n[:, -recent_days:] * m[:, -recent_days:]).sum(axis=1), recent_n,
                            out=np.full(len(query_days), np.nan, dtype=np.longdouble), where=recent_n > 0)
    baseline_mean = np.divide((n[:, :-recent_days] * m[:, :-recent_days]).sum(axis=1), baseline_n,
                              out=np.full(len(query_days), np.nan, dtype=np.longdouble), where=baseline_n > 0)
    with np.errstate(over='ignore', invalid='ignore'):
        variance_sum = (n[:, :-recent_days] * (v[:, :-recent_days] + (m[:, :-recent_days] - baseline_mean[:, None]) ** 2)).sum(axis=1)
        variance = np.divide(variance_sum, baseline_n, out=np.full(len(query_days), np.nan, dtype=np.longdouble), where=baseline_n > 0)
        difference = recent_mean - baseline_mean
    valid = (recent_n > 0) & (baseline_n >= 10) & ((n[:, :-recent_days] > 0).sum(axis=1) >= 7)
    valid &= np.isfinite(difference) & np.isfinite(variance) & (variance >= 0)
    score = np.full(len(query_days), np.nan, dtype=np.longdouble)
    constant = valid & (variance == 0)
    score[constant] = np.sign(difference[constant]) * 20
    varying = valid & (variance > 0)
    score[varying] = np.clip(difference[varying] / np.sqrt(variance[varying]), -20, 20)
    return recent_n > 0, score.astype(float)


def numeric_matrix(daily, object_ids, moments):
    days, indices = query_index(object_ids, moments)
    result = np.full((len(days), 40), np.nan)
    for object_id, selected in indices.items():
        own = daily[daily['object_id'] == object_id]
        for type_index, (_, sensor_type) in enumerate(TYPES):
            typed = own[own['sensor_type'] == sensor_type]
            channels = [rows.sort_values('as_of') for _, rows in typed.groupby('channel_id', sort=False)]
            for window_index, window in enumerate((1, 7)):
                offset = type_index * 20 + window_index * 10
                result[selected, offset:offset + 3] = 0
                if not channels:
                    continue
                channel_scores = [channel_numeric_scores(rows, days[selected], window) for rows in channels]
                recent = np.column_stack([row[0] for row in channel_scores])
                scores = np.column_stack([row[1] for row in channel_scores])
                valid = np.isfinite(scores)
                valid_n = valid.sum(axis=1)
                result[selected, offset] = recent.sum(axis=1)
                result[selected, offset + 1] = valid_n
                result[selected, offset + 2] = (recent & ~valid).sum(axis=1)
                for row in np.flatnonzero(valid_n > 0):
                    values = scores[row, valid[row]]
                    absolute = np.abs(values)
                    result[selected[row], offset + 3:offset + 10] = [absolute.max(), absolute.mean(), np.quantile(absolute, 0.9),
                                                                  values.max(), values.min(), (values > 3).mean(), (values < -3).mean()]
    return result
