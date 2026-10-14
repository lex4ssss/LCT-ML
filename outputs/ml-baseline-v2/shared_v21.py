import numpy as np
import pandas as pd
from families_v14 import ConstantModel
import models_v12 as models
import time_v18 as timing


def expand_training(matrix, onsets, object_ids, days, hours):
    matrix = np.asarray(matrix, dtype=np.float64)
    object_ids = np.asarray(object_ids).astype(str)
    days = np.asarray(days, dtype="datetime64[us]")
    if matrix.ndim != 2 or object_ids.ndim != 1 or days.ndim != 1:
        raise ValueError("Aligned matrix and vectors required")
    if len(matrix) != len(onsets) or len(object_ids) != len(matrix) or len(days) != len(matrix):
        raise ValueError("Training row counts differ")
    if np.isinf(matrix).any() or np.isnat(days).any():
        raise ValueError("Invalid training values")
    indices, intervals, outcomes, training = [], [], [], []
    for interval, (lower, upper, at_risk, positive) in enumerate(timing.risk_sets(onsets, hours)):
        selected = np.flatnonzero(at_risk)
        if not len(selected):
            raise ValueError("Empty conditional interval")
        indices.append(selected)
        intervals.append(np.full(len(selected), interval, dtype=int))
        outcomes.append(positive[selected])
        training.append(dict(lower_hours=lower, upper_hours=upper, rows=len(selected),
                             positives=int(positive[selected].sum()), base_rate=float(positive[selected].mean())))
    indices, intervals, outcomes = np.concatenate(indices), np.concatenate(intervals), np.concatenate(outcomes)
    order = np.lexsort((intervals, indices, days[indices].astype(np.int64)))
    indices, intervals, outcomes = indices[order], intervals[order], outcomes[order]
    if outcomes.sum() != np.isfinite(np.asarray(onsets, dtype=float)).sum():
        raise ValueError("Onset count did not reconcile")
    return dict(matrix=np.column_stack([matrix[indices], intervals]), labels=outcomes,
                object_ids=object_ids[indices], days=days[indices], source_rows=indices,
                interval_codes=intervals, training=training)


class SharedHazardModel:
    def __init__(self, model, numeric_features, categorical, hours, training):
        if hours not in timing.INTERVALS:
            raise ValueError("Original horizon required")
        self.model = model
        self.numeric_features = numeric_features
        self.categorical = categorical
        self.intervals = timing.INTERVALS[hours]
        self.training = training
        self.n_features_in_ = numeric_features + int(categorical)

    def predict_curve(self, matrix):
        if self.categorical:
            numeric = [f"f{index}" for index in range(self.numeric_features)]
            expected = numeric + ["object_id"]
            if not isinstance(matrix, pd.DataFrame) or len(matrix.columns) != len(expected) or set(matrix.columns) != set(expected):
                raise ValueError("Named categorical model input required")
            matrix = matrix.loc[:, expected]
        else:
            matrix = np.asarray(matrix, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[1] != self.numeric_features or np.isinf(matrix).any():
                raise ValueError("Numeric prediction feature count differs")
        if not len(matrix):
            return np.empty((0, len(self.intervals)))
        hazards = []
        for interval in range(len(self.intervals)):
            if self.categorical:
                expanded = matrix.copy()
                expanded[f"f{self.numeric_features}"] = interval
                expanded = expanded.loc[:, numeric + [f"f{self.numeric_features}", "object_id"]]
            else:
                expanded = np.column_stack([matrix, np.full(len(matrix), interval)])
            hazards.append(self.model.predict_proba(expanded)[:, 1])
        return timing.cumulative_risk(np.column_stack(hazards))

    def predict_proba(self, matrix):
        risk = self.predict_curve(matrix)[:, -1]
        return np.column_stack([1 - risk, risk])


def fit(variant, matrix, onsets, object_ids, categories, days, hours):
    if not variant.startswith(("H_", "C_")):
        raise ValueError("Shared hazard model family")
    expanded = expand_training(matrix, onsets, object_ids, days, hours)
    classes = np.unique(expanded["labels"])
    if len(classes) == 1:
        model = ConstantModel(classes[0])
    else:
        model = models.fit_model(variant, expanded["matrix"], expanded["labels"], expanded["object_ids"],
                                 categories, expanded["days"])
    return SharedHazardModel(model, expanded["matrix"].shape[1] - 1, variant.startswith("C_"), hours, expanded["training"])
