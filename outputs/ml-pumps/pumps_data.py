import numpy as np
import pandas as pd

ORIGIN = pd.Timestamp('2025-07-14 00:00:00')
STEP_MINUTES = 5
TRAJECTORY_LENGTHS = (12, 96, 288)
SENSORS = [
    'vibration_pump_bearing_drive', 'vibration_pump_bearing_nondrive',
    'vibration_motor_bearing_drive', 'vibration_motor_bearing_nondrive',
    'motor_current', 'pressure_diff_filter', 'pressure_suction', 'pressure_discharge',
    'temperature_motor_bearing_drive', 'temperature_motor_bearing_nondrive',
    'temperature_pump_bearing_drive', 'temperature_pump_bearing_nondrive',
]
PUMPS = ['NPV_2_1', 'NPV_2_2', 'NPV_2_3', 'NPV_2_4']
CHANGE_WINDOW = 12


class NotReconstructable(ValueError):
    pass


def load(path):
    columns = ['timestamp', 'pump_id', 'fault_type_name', *SENSORS]
    frame = pd.read_csv(path, usecols=columns, dtype={'fault_type_name': 'string'}, parse_dates=['timestamp'])
    return frame.reset_index(drop=True)


def reconstruct(frame):
    step = ((frame['timestamp'] - ORIGIN).dt.total_seconds() // (STEP_MINUTES * 60)).astype(np.int64).to_numpy()
    labelled = frame['fault_type_name'].notna().to_numpy()
    pump = frame['pump_id'].to_numpy()
    trajectory = np.full(len(frame), -1, dtype=np.int64)
    next_id = 0
    max_length = max(TRAJECTORY_LENGTHS)
    for pump_id in pd.unique(pump):
        in_window = np.flatnonzero((pump == pump_id) & (step < max_length))
        by_step = in_window[np.argsort(step[in_window], kind='stable')]
        steps_here = step[by_step]
        open_ids = np.array([], dtype=np.int64)
        for current in range(max_length):
            rows = by_step[steps_here == current]
            if current == 0:
                open_ids = np.arange(next_id, next_id + len(rows))
                next_id += len(rows)
            if len(rows) != len(open_ids):
                raise NotReconstructable(f'{pump_id}: {len(rows)} rows at step {current}, {len(open_ids)} open trajectories')
            trajectory[rows] = open_ids
            open_ids = open_ids[~labelled[rows]]
        if len(open_ids):
            raise NotReconstructable(f'{pump_id}: {len(open_ids)} trajectories without a fault label')
    result = frame.assign(step=step, trajectory=trajectory)
    in_trajectory = result[result['trajectory'] >= 0]
    lengths = in_trajectory.groupby('trajectory')['step'].agg(['size', 'max'])
    if not lengths['size'].isin(TRAJECTORY_LENGTHS).all() or (lengths['size'] != lengths['max'] + 1).any():
        raise NotReconstructable('trajectory lengths outside 12/96/288 or with gaps')
    labels_per_trajectory = in_trajectory['fault_type_name'].notna().groupby(in_trajectory['trajectory']).sum()
    if (labels_per_trajectory != 1).any():
        raise NotReconstructable('trajectory without exactly one fault label')
    if (labelled & (trajectory < 0)).any():
        raise NotReconstructable('fault label outside trajectories')
    return result


def prediction_rows(reconstructed, horizons_minutes):
    frame = reconstructed.sort_values(['trajectory', 'step'], kind='stable')
    in_trajectory = frame['trajectory'].to_numpy() >= 0
    last_step = frame.groupby('trajectory')['step'].transform('max').to_numpy()
    is_fault_row = in_trajectory & (frame['step'].to_numpy() == last_step)
    fault_type = frame['fault_type_name'].where(is_fault_row).groupby(frame['trajectory']).transform('first')
    remaining = np.where(in_trajectory, STEP_MINUTES * (last_step - frame['step'].to_numpy()), np.inf)
    features = feature_frame(frame, in_trajectory)
    keep = ~is_fault_row
    labels = {h: ((remaining > 0) & (remaining <= h))[keep].astype(np.int8) for h in horizons_minutes}
    meta = pd.DataFrame({
        'pump_id': frame['pump_id'].to_numpy(),
        'timestamp': frame['timestamp'].to_numpy(),
        'trajectory': frame['trajectory'].to_numpy(),
        'step': frame['step'].to_numpy(),
        'fault_type': fault_type.where(in_trajectory).to_numpy(),
    })[keep].reset_index(drop=True)
    return features[keep].reset_index(drop=True), labels, meta


def feature_frame(frame, in_trajectory):
    values = frame[SENSORS].to_numpy(dtype=np.float64)
    trajectory = frame['trajectory'].to_numpy()
    position = frame.groupby('trajectory').cumcount().to_numpy()
    lag = np.minimum(position, CHANGE_WINDOW)
    earlier = values[np.arange(len(frame)) - lag]
    change = np.where(in_trajectory[:, None], values - earlier, 0.0)
    mean_step = np.divide(change, np.maximum(lag, 1)[:, None])
    features = pd.DataFrame(values, columns=SENSORS, index=frame.index)
    for i, name in enumerate(SENSORS):
        features[f'{name}_change_{CHANGE_WINDOW}'] = change[:, i]
        features[f'{name}_mean_step_{CHANGE_WINDOW}'] = mean_step[:, i]
    features['pump_type'] = pd.Categorical(frame['pump_id'], categories=PUMPS).codes
    assert (trajectory[np.arange(len(frame)) - lag] == trajectory).all()
    return features


def rule_alarm(features):
    vibration = features[SENSORS[:4]].max(axis=1) > 2.8
    temperature = features[SENSORS[8:]].max(axis=1) > 80
    return (vibration | (features['motor_current'] > 82) | (features['pressure_diff_filter'] > 0.15)
            | temperature).to_numpy()


def split(meta):
    part = np.empty(len(meta), dtype=object)
    for pump_id, rows in meta.groupby('pump_id').groups.items():
        rows = np.asarray(rows)
        sub = meta.loc[rows]
        trajectory_rows = rows[sub['trajectory'].to_numpy() >= 0]
        ids = np.sort(meta.loc[trajectory_rows, 'trajectory'].unique())
        part[trajectory_rows] = share_labels(meta.loc[trajectory_rows, 'trajectory'].to_numpy(), ids)
        normal_rows = rows[sub['trajectory'].to_numpy() < 0]
        order = normal_rows[np.argsort(meta.loc[normal_rows, 'timestamp'].to_numpy(), kind='stable')]
        part[order] = share_labels(np.arange(len(order)), np.arange(len(order)))
    return part


def share_labels(keys, ordered_keys):
    n = len(ordered_keys)
    rank = pd.Series(np.arange(n), index=ordered_keys).reindex(keys).to_numpy()
    return np.where(rank < int(0.6 * n), 'train', np.where(rank < int(0.8 * n), 'validation', 'test'))
