import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from exp_metrics_core import confusion, recall_at_precision


def reliability(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    bins = []
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        inside = (scores >= lower) & ((scores < upper) if index < 9 else (scores <= 1.0))
        if inside.any():
            bins.append(dict(lower=round(lower, 1), upper=round(upper, 1), rows=int(inside.sum()),
                             mean_score=float(scores[inside].mean()), observed_fraction=float(labels[inside].mean())))
    return bins


def evaluate(labels, scores, threshold):
    labels = np.asarray(labels).astype(np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    report = dict(rows=int(len(labels)), positives=int(labels.sum()), prevalence=float(labels.mean()),
                  average_precision=float(average_precision_score(labels, scores)),
                  brier=float(brier_score_loss(labels, scores)), roc_auc=float(roc_auc_score(labels, scores)),
                  recall_at_precision_0_3=recall_at_precision(labels, scores, 0.3),
                  recall_at_precision_0_5=recall_at_precision(labels, scores, 0.5), threshold=float(threshold))
    report.update(confusion(labels, scores, threshold))
    return report
