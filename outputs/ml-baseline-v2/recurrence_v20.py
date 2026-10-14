import numpy as np


HOUR = 3_600_000_000
WINDOWS = (2160, 4320, 8760)
LAGS = (168, 336, 672, 720, 2160, 4320, 8760)
GAP_WINDOWS = (8, 32)
AGE_CAP = 17520


def feature_names():
    names = [f"recorded_{kind}_count_{hours}h" for kind in ("start", "burst") for hours in WINDOWS]
    names += [f"recorded_start_echo_{lag}h" for lag in LAGS]
    names += ["first_recorded_burst_age_h", "last_recorded_burst_age_h", "last_recorded_start_age_h"]
    for size in GAP_WINDOWS:
        names += [f"recorded_gap_last{size}_{stat}" for stat in ("median_h", "mean_h", "cv", "min_h", "max_h")]
        names += [f"recorded_burst_age_over_last{size}_median"]
    names += [f"recorded_start_log_count_change_{hours}h" for hours in WINDOWS[:2]]
    return names


def moments(values):
    array = np.asarray(values, dtype="datetime64[us]")
    if array.ndim != 1 or np.isnat(array).any():
        raise ValueError("Finite timestamp vector required")
    return array.astype(np.int64)


def burst_starts(starts):
    starts = np.unique(moments(starts))
    if not len(starts):
        return starts
    return starts[np.r_[True, np.diff(starts) > 900_000_000]]


def object_features(starts, days, hours):
    if hours not in (72, 168):
        raise ValueError("Original target horizon required")
    starts = np.unique(moments(starts))
    days = moments(days)
    bursts = burst_starts(starts.astype("datetime64[us]"))
    parts = []
    raw_right = np.searchsorted(starts, days, "right")
    burst_right = np.searchsorted(bursts, days, "right")
    for events, right in ((starts, raw_right), (bursts, burst_right)):
        for window in WINDOWS:
            parts.append(right - np.searchsorted(events, days - window * HOUR, "right"))
    for lag in LAGS:
        lower, upper = days - lag * HOUR, days - (lag - hours) * HOUR
        parts.append(np.searchsorted(starts, upper, "right") - np.searchsorted(starts, lower, "right"))
    ages = []
    for events, right, first in ((bursts, burst_right, True), (bursts, burst_right, False),
                                (starts, raw_right, False)):
        age = np.full(len(days), np.nan)
        found = right > 0
        if len(events):
            previous = np.zeros(found.sum(), dtype=int) if first else right[found] - 1
            age[found] = np.minimum((days[found] - events[previous]) / HOUR, AGE_CAP)
        ages.append(age)
    parts.extend(ages)
    for size in GAP_WINDOWS:
        statistics = np.full((len(days), 6), np.nan)
        for index, end in enumerate(burst_right):
            if end < 2:
                continue
            gaps = np.diff(bursts[max(0, end - size - 1):end]) / HOUR
            median, mean = np.median(gaps), np.mean(gaps)
            statistics[index] = (median, mean, np.std(gaps) / mean, gaps.min(), gaps.max(),
                                 min(ages[1][index] / median, 20))
        parts.extend(statistics.T)
    for window in WINDOWS[:2]:
        middle = np.searchsorted(starts, days - window * HOUR, "right")
        left = np.searchsorted(starts, days - 2 * window * HOUR, "right")
        parts.append(np.log1p(raw_right - middle) - np.log1p(middle - left))
    matrix = np.column_stack(parts).astype(np.float64)
    if matrix.shape != (len(days), len(feature_names())) or np.isinf(matrix).any():
        raise ValueError("Invalid recurrence features")
    return matrix


def read_history(connection, episodes, registry, cutoff):
    connection.register("v20_registry", registry[["channel_id", "object_id"]].drop_duplicates("channel_id", keep="last"))
    return connection.execute(
        "SELECT DISTINCT coalesce(m.object_id,'unknown') AS object_id, e.first_alarm_at "
        "FROM read_parquet(?) e LEFT JOIN v20_registry m ON cast(e.channel_id AS VARCHAR)=m.channel_id "
        "WHERE e.first_alarm_at<=CAST(? AS TIMESTAMP) ORDER BY object_id, first_alarm_at",
        [str(episodes), str(np.datetime64(cutoff, "us"))],
    ).fetchdf()


def feature_matrix(history, object_ids, days, hours):
    object_ids = np.asarray(object_ids).astype(str)
    days = np.asarray(days, dtype="datetime64[us]")
    if object_ids.ndim != 1 or days.ndim != 1 or len(object_ids) != len(days):
        raise ValueError("Aligned object and timestamp vectors required")
    groups = {str(key): group["first_alarm_at"].to_numpy(dtype="datetime64[us]")
              for key, group in history.groupby("object_id", sort=False)}
    matrix = np.empty((len(days), len(feature_names())), dtype=np.float64)
    for object_id in np.unique(object_ids):
        selected = object_ids == object_id
        matrix[selected] = object_features(groups.get(object_id, []), days[selected], hours)
    return matrix
