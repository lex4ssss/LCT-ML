import numpy as np
from families_v14 import ConstantModel
import models_v12 as models


INTERVALS = {72: (24, 72), 168: (24, 72, 168)}


def label_table(connection, examples, episodes, mapping, unknown_code, hours):
    if hours not in INTERVALS:
        raise ValueError('target horizon')
    connection.register('v18_registry', mapping[['channel_id', 'object_code']].drop_duplicates('channel_id', keep='last'))
    query = ('WITH e AS (SELECT channel_id, as_of FROM read_parquet(?) WHERE label IS NOT NULL '
             "AND split IN ('train','validation') AND as_of < TIMESTAMP '2026-01-01'), "
             "k AS (SELECT channel_id, first_alarm_at FROM read_parquet(?) WHERE left_boundary='observed_quiet_gap'), "
             "u AS (SELECT channel_id, first_alarm_at FROM read_parquet(?) WHERE left_boundary IS DISTINCT FROM 'observed_quiet_gap'), "
             'joined AS (SELECT coalesce(m.object_code, ?) AS object_code, e.as_of, '
             "e.as_of + ? * INTERVAL '1 hour' AS horizon_end, k.first_alarm_at AS known_at, u.first_alarm_at AS unknown_at "
             'FROM e LEFT JOIN v18_registry m ON e.channel_id=m.channel_id '
             'ASOF LEFT JOIN k ON e.channel_id=k.channel_id AND e.as_of<k.first_alarm_at '
             'ASOF LEFT JOIN u ON e.channel_id=u.channel_id AND e.as_of<u.first_alarm_at), '
             'labeled AS (SELECT object_code, as_of, '
             'CASE WHEN unknown_at<=horizon_end THEN NULL WHEN known_at<=horizon_end THEN 1 ELSE 0 END AS label, '
             'CASE WHEN unknown_at<=horizon_end THEN NULL WHEN known_at<=horizon_end '
             "THEN date_diff('microsecond', as_of, known_at)/3600000000.0 ELSE NULL END AS next_hours FROM joined) "
             "SELECT object_code * 1000000 + date_diff('day', DATE '1970-01-01', as_of) AS object_key, "
             'count(label)>0 AS present, coalesce(max(label),0) AS label, min(next_hours) AS next_hours, '
             'count(*) AS source_channels, count(label) AS eligible_channels, count(*)-count(label) AS excluded_channels '
             'FROM labeled GROUP BY object_key ORDER BY object_key')
    return connection.execute(query, [str(examples), str(episodes), str(episodes), int(unknown_code), int(hours)]).fetchdf()


def align(table, shared):
    codes = {name: index for index, name in enumerate(shared['categories'])}
    keys = np.array([codes[name] for name in shared['object_ids']]) * 1_000_000 + shared['days'].astype('datetime64[D]').astype(np.int64)
    if len(np.unique(keys)) != len(keys) or table['object_key'].duplicated().any():
        raise ValueError('unique object-days required')
    rows = table.set_index('object_key').reindex(keys)
    return dict(present=rows['present'].fillna(False).to_numpy(dtype=bool), label=rows['label'].fillna(0).to_numpy(dtype=int),
                next_hours=rows['next_hours'].to_numpy(dtype=float),
                **{name: rows[name].fillna(0).to_numpy(dtype=int) for name in ('source_channels', 'eligible_channels', 'excluded_channels')})


def risk_sets(onset_hours, hours):
    if hours not in INTERVALS:
        raise ValueError('target horizon')
    onset_hours = np.asarray(onset_hours, dtype=float)
    if onset_hours.ndim != 1 or np.isinf(onset_hours).any() or np.any((onset_hours <= 0) | (onset_hours > hours)):
        raise ValueError('onsets must be missing or inside the original horizon')
    lower = 0
    for upper in INTERVALS[hours]:
        at_risk = np.isnan(onset_hours) | (onset_hours > lower)
        positive = (onset_hours > lower) & (onset_hours <= upper)
        yield lower, upper, at_risk, positive.astype(int)
        lower = upper


def cumulative_risk(hazards):
    hazards = np.asarray(hazards, dtype=float)
    if hazards.ndim != 2 or hazards.shape[1] not in (2, 3) or not np.isfinite(hazards).all() or np.any((hazards < 0) | (hazards > 1)):
        raise ValueError('two or three finite conditional probabilities required')
    return 1 - np.cumprod(1 - hazards, axis=1)


class HazardModel:
    def __init__(self, heads, intervals, training):
        self.heads = heads
        self.intervals = intervals
        self.training = training

    def predict_curve(self, matrix):
        return cumulative_risk(np.column_stack([head.predict_proba(matrix)[:, 1] for head in self.heads]))

    def predict_proba(self, matrix):
        risk = self.predict_curve(matrix)[:, -1]
        return np.column_stack([1 - risk, risk])


def fit(variant, matrix, onset_hours, object_ids, categories, days, hours):
    if not variant.startswith(('H_', 'C_')):
        raise ValueError('hazard model family')
    heads, training = [], []
    for lower, upper, at_risk, labels in risk_sets(onset_hours, hours):
        if not np.any(at_risk):
            raise ValueError('empty conditional risk set')
        classes = np.unique(labels[at_risk])
        if len(classes) == 1:
            model = ConstantModel(classes[0])
        else:
            model = models.fit_model(variant, matrix[at_risk], labels[at_risk], object_ids[at_risk], categories, days[at_risk])
        heads.append(model)
        training.append(dict(lower_hours=lower, upper_hours=upper, rows=int(at_risk.sum()),
                             positives=int(labels[at_risk].sum()), base_rate=float(labels[at_risk].mean()), constant=len(classes) == 1))
    return HazardModel(heads, INTERVALS[hours], training)


def error_audit(labels, scores, threshold, onset_hours, history, channels, hours):
    positive = np.asarray(labels).astype(bool)
    predicted = np.asarray(scores) >= threshold
    errors = dict(tp=positive & predicted, fp=~positive & predicted, fn=positive & ~predicted, tn=~positive & ~predicted)
    cohorts = dict(no_target_7d=history[:, 1] == 0, target_7d=history[:, 1] > 0,
                   no_target_30d=history[:, 2] == 0, target_30d=history[:, 2] > 0,
                   excluded_source_channels=channels['excluded_channels'] > 0)
    intervals = []
    for lower, upper, _, _ in risk_sets(onset_hours, hours):
        selected = (onset_hours > lower) & (onset_hours <= upper)
        intervals.append(dict(lower_hours=lower, upper_hours=upper, positives=int((positive & selected).sum()),
                              false_negatives=int((errors['fn'] & selected).sum())))
    return dict(rows=len(labels), positives=int(positive.sum()), base_rate=float(positive.mean()),
                confusion={name: int(mask.sum()) for name, mask in errors.items()}, onset_intervals=intervals,
                cohorts={name: dict(rows=int(mask.sum()), **{error: int((mask & selected).sum()) for error, selected in errors.items()})
                         for name, mask in cohorts.items()},
                source_channels=int(channels['source_channels'].sum()), eligible_channels=int(channels['eligible_channels'].sum()),
                excluded_channels=int(channels['excluded_channels'].sum()))
