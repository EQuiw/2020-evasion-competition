import numpy as np
import typing


class ClassificationResult:

    def __init__(self, prob_score: float, threshold: float, features: typing.Optional[np.ndarray]):
        self.prob_score = prob_score
        self.threshold = threshold
        self.features = features

    @staticmethod
    def get_max_decision(classificationresults: typing.List['ClassificationResult']):
        preds = []
        for classificationresult in classificationresults:
            preds.append(int(classificationresult.prob_score > classificationresult.threshold))
        return int(np.max(preds) > 0)
