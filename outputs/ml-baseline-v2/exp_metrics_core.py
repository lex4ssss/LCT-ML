import numpy as np


def threshold_curve(labels, scores):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind='stable')
    sorted_scores, sorted_labels = scores[order], labels[order]
    last_of_group = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    tp = np.cumsum(sorted_labels)[last_of_group]
    fp = np.cumsum(1 - sorted_labels)[last_of_group]
    return sorted_scores[last_of_group], tp, fp, int(labels.sum())


def max_f1_threshold(labels, scores):
    thresholds, tp, fp, positives = threshold_curve(labels, scores)
    denominator = tp + fp + positives
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(tp)), where=denominator > 0)
    best = np.flatnonzero(f1 == f1.max())[0]
    return float(thresholds[best]), float(f1[best])


def confusion(labels, scores, threshold):
    labels = np.asarray(labels).astype(np.int64)
    predicted = np.asarray(scores, dtype=np.float64) >= threshold
    tp = int((predicted & (labels == 1)).sum())
    fp = int((predicted & (labels == 0)).sum())
    fn = int((~predicted & (labels == 1)).sum())
    tn = int((~predicted & (labels == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                alerts_per_1000=1000 * (tp + fp) / len(labels))


def recall_at_precision(labels, scores, target):
    _, tp, fp, positives = threshold_curve(labels, scores)
    if positives == 0:
        return 0.0
    eligible = tp / (tp + fp) >= target
    return float(tp[eligible].max() / positives) if eligible.any() else 0.0
