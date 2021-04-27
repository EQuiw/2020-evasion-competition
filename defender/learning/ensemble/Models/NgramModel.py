import os
import sys
from xgboost import XGBClassifier
from types import SimpleNamespace
import json

import numpy as np

from learning.emberboost.ember.boost.boost_testing import predict_class
from learning.ensemble.Models.ClassifierInterface import ClassifierInterface
from learning.ensemble.Models.ClassificationResult import ClassificationResult
from features.ngrams.ngrams import load_config, main as ngrams_main


class NgramModel(ClassifierInterface):

    def __init__(self, ngrammodelpath: str, cfg_path: str, dim_file: str, threshold: float, command: str, nlen: int,
                 skiplen: int = None, nocutting: bool = False, binary: bool = True):
        self.ngrammodelpath = ngrammodelpath
        self.threshold = threshold

        self.clf: XGBClassifier = XGBClassifier()
        self.clf.load_model(fname=os.path.join(ngrammodelpath, "model_xgboost.dat"))

        # provide mock-up config obj
        cfg = MockupConf(config=cfg_path, command=command, nlen=nlen, libsvm=False,
                         skiplen=skiplen, nocutting=nocutting, dim_file=None)
        self.cfg = load_config(cfg)
        self.dims = json.load(open(dim_file, 'r'))
        self.binary = binary

    def extract_features(self, bytez, pe_binary=None):
        # non-optimized version
        ngrams = ngrams_main(self.cfg, bytez=bytez)
        features = np.zeros((1, 256**self.cfg.nlen))
        for term, count in ngrams.items():
            dim = sum(term[i]*256**(3-(i+1)) for i in range(self.cfg.nlen))
            features[0, dim] = count
        features = features[:, self.dims]
        return features

        # optimized version in cython was not working in Docker

    def predict(self, bytez, pe_binary=None):
        features = self.extract_features(bytez=bytez, pe_binary=pe_binary)
        pred = [predict_class(clf=self.clf, samples=features, threshold=self.threshold)[0]]
        try:
            return pred
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [0]

    def predict_proba(self, bytez, pe_binary=None):
        features = self.extract_features(bytez=bytez, pe_binary=pe_binary)
        pred = [self.clf.predict_proba(features)[:, 1][0]]
        try:
            return pred
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


class MockupConf(SimpleNamespace):
    """ For a mockup conf object, we need a namespace obj that is iterable like a list. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx = {i: k for i, k in enumerate(self.__dict__)}

    def __getitem__(self, i):
        if i not in self.idx:
            raise IndexError()
        return self.__dict__[self.idx[i]]
