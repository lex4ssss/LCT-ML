import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

TEST_START = np.datetime64('2026-01-01T00:00:00', 'us')
V2_FEATURES = ['event_count_24h', 'alarm_count_24h', 'event_count_7d', 'alarm_count_7d', 'event_count_30d', 'alarm_count_30d', 'active_days_30d', 'episode_starts_7d', 'episode_starts_30d', 'last_event_age_hours', 'last_alarm_age_hours_30d', 'channel_age_days', 'log_gap_hours_30d']
FOLDS = {'A': ('2023-12-31', '2024-01-01', '2025-01-01'), 'B': ('2024-12-31', '2025-01-01', '2026-01-01')}


def fold_masks(as_of, fold):
    as_of = np.asarray(as_of).astype('datetime64[us]')
    if (as_of >= TEST_START).any():
        raise ValueError('test rows present')
    if fold not in FOLDS:
        raise ValueError('fold')
    train_end, eval_start, eval_end = (np.datetime64(value, 'us') for value in FOLDS[fold])
    return as_of < train_end, (as_of >= eval_start) & (as_of < eval_end)


def matrix_v1(columns):
    alarm_age = np.asarray(columns['last_alarm_age_hours_30d'], dtype=np.float64)
    recent_alarm_age = np.where(alarm_age < 24, alarm_age, np.nan)
    base = [np.asarray(columns[name], dtype=np.float64) for name in ('event_count_24h', 'alarm_count_24h', 'last_event_age_hours')]
    return np.column_stack(base + [recent_alarm_age])


def matrix_v2(columns, calendar):
    features = [np.asarray(columns[name], dtype=np.float64) for name in V2_FEATURES]
    age = V2_FEATURES.index('channel_age_days')
    features[age] = np.minimum(features[age], 365)
    if calendar:
        days = np.asarray(columns['as_of']).astype('datetime64[D]')
        features.append((days.astype('datetime64[M]').astype(np.int64) % 12 + 1).astype(np.float64))
        features.append(((days.astype(np.int64) + 3) % 7).astype(np.float64))
    return np.column_stack(features)


def logistic(fill_value, max_iter):
    return make_pipeline(SimpleImputer(strategy='constant', fill_value=fill_value), FunctionTransformer(np.log1p), StandardScaler(),
                         LogisticRegression(C=1, max_iter=max_iter, tol=1e-6, solver='lbfgs'))


def make_model(kind):
    if kind == 'logistic_v1':
        return logistic(24, 500)
    if kind == 'logistic_v2':
        return logistic(720, 1000)
    if kind == 'hgb':
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=0)
    raise ValueError('kind')
