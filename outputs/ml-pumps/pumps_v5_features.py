import numpy as np
import pandas as pd

import pumps_data

SIGNALS = ['vibration_mean', 'temperature_mean', 'motor_current', 'pressure_diff_filter', 'pressure_suction', 'pressure_discharge']
MEAN_WINDOWS = (12, 36, 96, 288)
SLOPE_WINDOWS = (12, 36, 96)
ONSET_NOISE_UNITS = 4.0


def signals(values):
    return np.column_stack([values[:, 0:4].mean(axis=1), values[:, 8:12].mean(axis=1), values[:, 4], values[:, 5], values[:, 6], values[:, 7]])


def normal_stats(frames):
    normal = pd.concat([frame[frame['trajectory'] < 0] for frame in frames])
    table = pd.DataFrame(signals(normal[pumps_data.SENSORS].to_numpy(dtype=np.float64)), columns=SIGNALS)
    table['pump_id'] = normal['pump_id'].to_numpy()
    grouped = table.groupby('pump_id')[SIGNALS]
    return dict(median={pump: row.to_numpy() for pump, row in grouped.median().iterrows()},
                std={pump: row.to_numpy() for pump, row in grouped.std().iterrows()})


def groups_and_positions(trajectory):
    group = np.where(trajectory >= 0, trajectory, -1 - np.arange(len(trajectory)))
    position = pd.Series(group).groupby(group).cumcount().to_numpy()
    return group, position


def rolling_sum(values, group, position, window):
    cumulative = pd.DataFrame(values).groupby(group).cumsum().to_numpy()
    earlier = np.arange(len(values)) - window
    has_earlier = position >= window
    return cumulative - np.where(has_earlier[:, None], cumulative[np.where(has_earlier, earlier, 0)], 0.0)


def rolling_mean(values, profile, group, position, window):
    inside = np.minimum(position + 1, window)
    return (rolling_sum(values, group, position, window) + (window - inside)[:, None] * profile) / window


def rolling_slope(values, profile, group, position, window):
    inside = np.minimum(position + 1, window)
    padded = (window - inside)[:, None]
    sum_x = rolling_sum(values, group, position, window) + padded * profile
    sum_px = rolling_sum(values * position[:, None], group, position, window)
    sum_dx = position[:, None] * rolling_sum(values, group, position, window) - sum_px
    sum_kx = (window - 1) * (sum_x - padded * profile) - sum_dx + profile * padded * (padded - 1) / 2
    sum_k = window * (window - 1) / 2
    sum_kk = (window - 1) * window * (2 * window - 1) / 6
    return (window * sum_kx - sum_k * sum_x) / (window * sum_kk - sum_k ** 2)


def onset_age(deviation, group, position):
    abnormal = (np.abs(deviation) > ONSET_NOISE_UNITS).any(axis=1)
    first = pd.Series(np.where(abnormal, position, np.iinfo(np.int64).max)).groupby(group).cummin().to_numpy()
    return np.where(first <= position, position - first, -1)


def feature_frame(frame, stats):
    values = frame[pumps_data.SENSORS].to_numpy(dtype=np.float64)
    pumps = frame['pump_id'].to_numpy()
    profile = np.vstack([stats['median'][pump] for pump in pumps])
    noise = np.vstack([stats['std'][pump] for pump in pumps])
    group, position = groups_and_positions(frame['trajectory'].to_numpy())
    signal = signals(values)
    columns = {name: values[:, i] for i, name in enumerate(pumps_data.SENSORS)}
    columns['pump_type'] = pd.Categorical(pumps, categories=pumps_data.PUMPS).codes.astype(np.float64)
    columns['vibration_max'] = values[:, 0:4].max(axis=1)
    columns['temperature_min'] = values[:, 8:12].min(axis=1)
    means = {window: rolling_mean(signal, profile, group, position, window) for window in MEAN_WINDOWS}
    for window, mean in means.items():
        for i, name in enumerate(SIGNALS):
            columns[f'{name}_mean_{window}'] = mean[:, i]
    for window in SLOPE_WINDOWS:
        slope = rolling_slope(signal, profile, group, position, window)
        for i, name in enumerate(SIGNALS):
            columns[f'{name}_slope_{window}'] = slope[:, i]
    deviation = (means[12] - profile) / noise
    for i, name in enumerate(SIGNALS):
        columns[f'{name}_deviation_12'] = deviation[:, i]
    columns['onset_age'] = onset_age((signal - profile) / noise, group, position).astype(np.float64)
    return pd.DataFrame(columns, index=frame.index).astype(np.float32)


def prediction_rows(reconstructed, stats, horizons_minutes):
    frame = reconstructed.sort_values(['trajectory', 'step'], kind='stable')
    in_trajectory = frame['trajectory'].to_numpy() >= 0
    last_step = frame.groupby('trajectory')['step'].transform('max').to_numpy()
    is_fault_row = in_trajectory & (frame['step'].to_numpy() == last_step)
    fault_type = frame['fault_type_name'].where(is_fault_row).groupby(frame['trajectory']).transform('first')
    remaining = np.where(in_trajectory, pumps_data.STEP_MINUTES * (last_step - frame['step'].to_numpy()), np.inf)
    keep = ~is_fault_row
    features = feature_frame(frame, stats)[keep].reset_index(drop=True)
    labels = {h: ((remaining > 0) & (remaining <= h))[keep].astype(np.int8) for h in horizons_minutes}
    meta = pd.DataFrame({'pump_id': frame['pump_id'].to_numpy(), 'trajectory': frame['trajectory'].to_numpy(),
                         'step': frame['step'].to_numpy(),
                         'fault_type': fault_type.where(in_trajectory).fillna('normal').to_numpy()})[keep].reset_index(drop=True)
    return features, labels, meta
