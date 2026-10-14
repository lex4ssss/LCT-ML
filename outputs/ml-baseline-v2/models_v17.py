import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import models_v12 as models


HGB_GRID = {name: dict(max_iter=iterations, learning_rate=rate, max_leaf_nodes=leaves,
                       min_samples_leaf=minimum, l2_regularization=regularization,
                       early_stopping=False, random_state=0)
            for name, iterations, rate, leaves, minimum, regularization in
            [('H0', 300, 0.05, 31, 200, 1), ('H1', 300, 0.05, 31, 50, 1),
             ('H2', 600, 0.05, 63, 50, 10), ('H3', 600, 0.03, 31, 200, 10)]}
CAT_GRID = {name: dict(iterations=iterations, learning_rate=rate, depth=depth, l2_leaf_reg=10,
                       loss_function='Logloss', random_seed=0, thread_count=4, has_time=True,
                       allow_writing_files=False, verbose=False)
            for name, iterations, rate, depth in
            [('C0', 600, 0.05, 6), ('C1', 600, 0.05, 8), ('C2', 1200, 0.03, 6), ('C3', 1200, 0.03, 8)]}


def inner_masks(days, present, hours):
    days = np.asarray(days).astype('datetime64[us]')
    if np.isnat(days).any() or np.any(days >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('invalid or test timestamps')
    start = np.datetime64('2024-01-01', 'us')
    end = np.datetime64('2025-01-01', 'us')
    horizon = np.timedelta64(hours, 'h')
    return present & (days + horizon < start), present & (days >= start) & (days + horizon <= end)


def grid():
    return {**HGB_GRID, **CAT_GRID}


def estimator(name):
    if name in HGB_GRID:
        return HistGradientBoostingClassifier(**HGB_GRID[name])
    if name in CAT_GRID:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**CAT_GRID[name])
    raise ValueError('grid candidate')


def fit(name, matrix, labels, object_ids, categories, days):
    order = np.argsort(days, kind='stable')
    matrix, labels, object_ids = matrix[order], labels[order], object_ids[order]
    model = estimator(name)
    if name.startswith('C'):
        return model.fit(models.model_frame(matrix, object_ids, categories), labels, cat_features=['object_id'])
    return model.fit(matrix, labels)


def predict(name, model, matrix, object_ids, categories):
    variant = 'C_17' if name.startswith('C') else 'H_17'
    return models.predict_model(variant, model, matrix, object_ids, categories)


def choose(inner, family):
    candidates = HGB_GRID if family == 'H' else CAT_GRID if family == 'C' else None
    if candidates is None:
        raise ValueError('model family')
    if any(f'{name}-inner' not in inner for name in candidates):
        raise ValueError('incomplete inner grid')
    def preference(name):
        parameters = candidates[name]
        iterations = parameters.get('max_iter', parameters.get('iterations'))
        complexity = parameters.get('max_leaf_nodes', parameters.get('depth'))
        return inner[f'{name}-inner']['ap'], -iterations, -complexity, parameters.get('min_samples_leaf', 0)
    return max(candidates, key=preference)
