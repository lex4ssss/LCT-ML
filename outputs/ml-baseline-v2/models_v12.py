from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import experiment_v2 as ex
import explore_v8 as e8


VARIANTS = ('R', 'H_N', 'H_D', 'H_ND', 'C_B', 'C_ND', 'C_ND_2022', 'L_ND_2022', 'E')


def toggles(variant):
    if variant not in VARIANTS:
        raise ValueError('variant')
    return dict(numeric=variant in ('H_N', 'H_ND', 'C_ND', 'C_ND_2022', 'L_ND_2022', 'E'),
                domains=variant in ('H_D', 'H_ND', 'C_ND', 'C_ND_2022', 'L_ND_2022', 'E'),
                object_identity=variant.startswith(('C_', 'L_')) or variant == 'E',
                train_from='2022-01-01' if variant.endswith('_2022') or variant == 'E' else None,
                class_weights=False, weather=False, synthetic=False, neural_network=False,
                online_retraining=False, calibration=False)


def feature_matrix(variant, base, numeric, domains):
    flags = toggles(variant)
    parts = [base]
    if flags['numeric']:
        parts.append(numeric)
    if flags['domains']:
        parts.append(domains)
    return np.column_stack(parts) if len(parts) > 1 else base


def model_frame(matrix, object_ids, categories, lightgbm=False):
    frame = pd.DataFrame(matrix, columns=[f'f{index}' for index in range(matrix.shape[1])])
    frame['object_id'] = pd.Categorical(object_ids, categories=categories) if lightgbm else np.asarray(object_ids).astype(str)
    if frame['object_id'].isna().any():
        raise ValueError('object category absent')
    return frame


def fit_model(variant, matrix, labels, object_ids, categories, days):
    order = np.argsort(days, kind='stable')
    matrix, labels, object_ids = matrix[order], labels[order], object_ids[order]
    if variant.startswith('H_'):
        return ex.make_model('hgb').fit(matrix, labels)
    if variant.startswith('C_'):
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05, l2_leaf_reg=10,
                                   loss_function='Logloss', random_seed=0, thread_count=4, has_time=True,
                                   allow_writing_files=False, verbose=False)
        return model.fit(model_frame(matrix, object_ids, categories), labels, cat_features=['object_id'])
    if variant.startswith('L_'):
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=31,
                               min_child_samples=200, reg_lambda=10, random_state=0, n_jobs=4, verbosity=-1)
        return model.fit(model_frame(matrix, object_ids, categories, True), labels, categorical_feature=['object_id'])
    raise ValueError('trainable variant')


def predict_model(variant, model, matrix, object_ids, categories):
    if variant.startswith(('C_', 'L_')):
        matrix = model_frame(matrix, object_ids, categories, variant.startswith('L_'))
    return model.predict_proba(matrix)[:, 1]


def training_masks(days, present, hours, variant):
    train, evaluation = e8.masks(days, present, hours)
    start = toggles(variant)['train_from']
    if start:
        train &= days >= np.datetime64(start, 'us')
    if np.any(days >= np.datetime64('2026-01-01', 'us')):
        raise ValueError('test rows in search')
    return train, evaluation


def metrics(labels, scores, rule, strong_rule, threshold=None):
    result = e8.selection(labels, scores, rule)
    result['ap'] = float(average_precision_score(labels, scores))
    result['rule_7d'] = e8.confusion(labels, strong_rule, 1.0)
    if threshold is not None:
        at = e8.confusion(labels, scores, threshold)
        result.update({name: at[name] for name in ('precision', 'recall', 'f1')})
        result['threshold'] = float(threshold)
        result['margin'] = min(at['precision'] - 0.7, at['recall'] - 0.5)
        result['qualifies'] = bool(result['margin'] >= 0 and result['base_rate'] <= 0.35 and at['f1'] > result['rule_f1'])
    return result


def accepted(candidate, reference):
    return bool(candidate['ap'] >= reference['ap'] + 0.01 and
                candidate['precision_at_recall_0_5'] >= reference['precision_at_recall_0_5'] and
                candidate['qualifies'] and candidate['f1'] >= reference['f1'])


def winner(results):
    reference = results['R']['validation']
    eligible = [(name, row) for name, row in results.items() if name != 'R' and accepted(row['validation'], reference)]
    if not eligible:
        return 'R'
    return max(eligible, key=lambda item: (item[1]['validation']['ap'], item[1]['validation']['margin'], -item[1]['features']))[0]


def save_verified(model, path, variant, matrix, object_ids, categories):
    expected = predict_model(variant, model, matrix, object_ids, categories)
    joblib.dump(model, Path(path))
    loaded = joblib.load(Path(path))
    actual = predict_model(variant, loaded, matrix, object_ids, categories)
    np.testing.assert_array_equal(actual, expected)
