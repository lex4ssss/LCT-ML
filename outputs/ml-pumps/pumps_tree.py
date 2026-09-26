import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class Tree:
    def __init__(self, router, experts):
        self.router, self.experts = router, experts

    def expert_score(self, fault, features):
        expert = self.experts[fault]
        if isinstance(expert, float):
            return np.full(len(features), expert)
        return expert.predict_proba(features)[:, 1]

    def predict_proba(self, features):
        routed = self.router.predict_proba(features)
        index = {name: i for i, name in enumerate(self.router.classes_)}
        probability = np.zeros(len(features))
        for fault in self.experts:
            if fault in index:
                probability += routed[:, index[fault]] * self.expert_score(fault, features)
        return np.column_stack([1 - probability, probability])


def fit_expert(features, labels):
    if labels.min() == labels.max():
        return float(labels[0])
    return HistGradientBoostingClassifier(random_state=0).fit(features, labels)
