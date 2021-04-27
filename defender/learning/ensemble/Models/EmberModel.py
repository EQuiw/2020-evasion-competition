import numpy as np
import os
import typing
import sys

from xgboost import XGBClassifier
from learning.emberboost.ember import PEFeatureExtractor

from learning.emberboost.ember.boost.boost_testing import predict_class
from learning.ensemble.Models.ClassifierInterface import ClassifierInterface
from .ClassificationResult import ClassificationResult


class EmberModel(ClassifierInterface):

    def __init__(self, embermodelpath: str, threshold: float, feature_version: int,
                 include_header: bool = False, remove_strings_stream: bool = True,
                 truncate: bool = True):
        self.embermodelpath = embermodelpath
        self.threshold = threshold

        self.clf: XGBClassifier = XGBClassifier()
        self.clf.load_model(fname=os.path.join(embermodelpath, "model_xgboost.dat"))

        self.extractor = PEFeatureExtractor(feature_version=feature_version,
                                            include_header=include_header,
                                            remove_strings_stream=remove_strings_stream)
        self.truncate = truncate

    def extract_features(self, bytez, pe_binary=None):
        features = np.array(self.extractor.feature_vector(bytez, pe_binary=pe_binary, truncate=self.truncate), dtype=np.float32)
        features = np.expand_dims(features, axis=0)
        return features

    def predict(self, bytez, pe_binary=None) -> list:
        try:
            features = self.extract_features(bytez=bytez, pe_binary=pe_binary)
            return self.predict_from_features(features=features)
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [0]

    def predict_from_features(self, features) -> list:
        try:
            return [predict_class(clf=self.clf, samples=features, threshold=self.threshold)[0]]
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [0]

    def predict_proba(self, bytez, pe_binary=None) -> list:
        try:
            features = self.extract_features(bytez=bytez, pe_binary=pe_binary)
            return self.predict_proba_from_features(features=features)
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [0.0]

    def predict_proba_from_features(self, features) -> list:
        try:
            return [self.clf.predict_proba(features)[:, 1][0]]
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [0.0]

    def predict_all_proba(self, bytez, pe_binary=None):
        try:
            features = self.extract_features(bytez=bytez, pe_binary=pe_binary)
            res = ClassificationResult(prob_score=self.predict_proba_from_features(features=features)[0],
                                       threshold=self.threshold,
                                       features=features)
            return [res]
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [ClassificationResult(prob_score=0.0, threshold=self.threshold, features=None)]
