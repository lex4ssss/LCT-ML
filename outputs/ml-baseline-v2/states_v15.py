from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml-dataset-v2'))
from class_labels import FAILURE_STATES, INCIDENT_STATES


STATES = tuple(FAILURE_STATES + INCIDENT_STATES)
WINDOWS = (1, 7, 30)


def daily_states(connection, observations, mapping, cutoff):
    connection.register('v15_registry', mapping[['channel_id', 'object_id']].drop_duplicates('channel_id', keep='last'))
    query = ('SELECT coalesce(r.object_id, ?) AS object_id, o.raw_value AS state, '
             "date_trunc('day', o.occurred_at - INTERVAL '1 microsecond') + INTERVAL '1 day' AS as_of, "
             'count(*) AS events, count(DISTINCT o.channel_id) AS channel_days, max(o.occurred_at) AS last_at '
             'FROM read_parquet(?) o LEFT JOIN v15_registry r ON o.channel_id=r.channel_id '
             'WHERE o.is_alarm AND o.raw_value IN (SELECT unnest(?)) AND o.occurred_at<=? '
             'GROUP BY object_id, state, as_of ORDER BY object_id, as_of, state')
    paths = [str(path) for path in observations] if isinstance(observations, (tuple, list)) else str(observations)
    return connection.execute(query, ['unknown', paths, list(STATES), pd.Timestamp(cutoff).to_pydatetime()]).fetchdf()


def state_matrix(daily, object_ids, moments):
    moments = np.asarray(moments).astype('datetime64[us]')
    if np.isnat(moments).any() or np.any(moments != moments.astype('datetime64[D]')):
        raise ValueError('object-day midnight timestamps required')
    object_ids = np.asarray(object_ids).astype(str)
    days = moments.astype('datetime64[D]').astype(np.int64)
    primary = np.zeros((len(days), 4 * len(STATES)))
    primary[:, 3::4] = np.nan
    context = np.zeros((len(days), 6 * len(STATES)))
    context[:, 0::2] = np.nan
    names = dict(primary=[f'state_{state}_{stat}' for state in range(len(STATES)) for stat in ['events_1d', 'events_7d', 'events_30d', 'age']],
                 context=[f'state_{state}_{stat}_{window}d' for state in range(len(STATES)) for window in WINDOWS for stat in ['share', 'channel_days']])
    state_code = {state: index for index, state in enumerate(STATES)}
    for object_id in np.unique(object_ids):
        selected = np.flatnonzero(object_ids == object_id)
        rows = daily[daily['object_id'] == object_id]
        row_days = rows['as_of'].to_numpy().astype('datetime64[D]').astype(np.int64)
        rows = rows[row_days <= days[selected].max()]
        row_days = row_days[row_days <= days[selected].max()]
        if rows.empty:
            continue
        first = min(int(row_days.min()), int(days[selected].min())) - 31
        width = int(days[selected].max() - first + 1)
        counts = np.zeros((width, len(STATES)))
        channels = counts.copy()
        last_at = np.full_like(counts, np.nan)
        positions = row_days - first
        state = np.array([state_code[value] for value in rows['state']])
        counts[positions, state] = rows['events'].to_numpy()
        channels[positions, state] = rows['channel_days'].to_numpy()
        last_at[positions, state] = rows['last_at'].to_numpy().astype('datetime64[us]').astype(np.int64) / 1e6
        seen = pd.DataFrame(last_at).ffill().to_numpy()
        age = np.minimum(((np.arange(width) + first)[:, None] * 86400 - seen) / 3600, 720)
        if np.any(age < 0):
            raise ValueError('future state observation in day bucket')
        event_windows = {window: pd.DataFrame(counts).rolling(window, min_periods=1).sum().to_numpy() for window in WINDOWS}
        channel_windows = {window: pd.DataFrame(channels).rolling(window, min_periods=1).sum().to_numpy() for window in WINDOWS}
        shares = {}
        for window, events in event_windows.items():
            total = events.sum(axis=1)
            shares[window] = np.divide(events, total[:, None], out=np.full_like(events, np.nan), where=total[:, None] > 0)
        main = np.column_stack([value for state in range(len(STATES)) for value in
                                [event_windows[1][:, state], event_windows[7][:, state], event_windows[30][:, state], age[:, state]]])
        extra = np.column_stack([value for state in range(len(STATES)) for window in WINDOWS for value in
                                 [shares[window][:, state], channel_windows[window][:, state]]])
        primary[selected] = main[days[selected] - first]
        context[selected] = extra[days[selected] - first]
    return primary, context, names
