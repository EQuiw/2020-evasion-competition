import numpy as np
import sys
import os
import pickle
import argparse

from learning.ensemble.Models.ClassifierInterface import ClassifierInterface
from learning.expert.expert import extract_peid, extract_yaru, load_config, init_expert
from .ClassificationResult import ClassificationResult


class ExpertModel(ClassifierInterface):

    def __init__(self, dictvectorizerpath: str, featureindicespath: str,
                 features: str, configpath: str):

        # For this type of classifier, we re-use the API of learning.expert.expert, so that
        # we define here the arguments that we would otherwise pass on the command line.
        # Simple, quick solution here...
        prog = "export-yaru-peid"
        descr = "Classify by using peid or yaru features"
        parser = argparse.ArgumentParser(prog=prog, description=descr)
        parser.add_argument('features', metavar='<features>', help='feature set: peid, yaru')
        parser.add_argument('--config', metavar='<file>',
                            help='set config file',
                            default='config.yml')

        cfg = parser.parse_args(['--config', configpath, features])

        if cfg.features not in ["peid", "yaru"]:
            raise NotImplementedError("Error: Unknown feature set {}".format(cfg.features))
        # Sanity checks
        if not os.path.exists(cfg.config):
            raise NotImplementedError("Error: File {} does not exist".format(cfg.config))

        self.dictv = pickle.load(open(os.path.join(dictvectorizerpath), "rb"))
        self.featureindices = pickle.load(open(os.path.join(featureindicespath), "rb"))

        # 1. Extract features from file
        self.cfg = load_config(cfg)
        init_expert(self.cfg)

    def _extract_features(self, bytez, pe_binary=None):
        cfg = self.cfg
        if cfg.features == "peid":
            feats = extract_peid(cfg, data=bytez)
        elif cfg.features == "yaru":
            feats = extract_yaru(cfg, data=bytez)
        else:
            raise NotImplementedError()

        curreport_dict = {}
        for x in feats:
            curreport_dict[x] = 1

        learning_features = self.dictv.transform(curreport_dict)
        features = learning_features[:, self.featureindices]

        return features

    def predict(self, bytez, pe_binary=None):
        try:
            features = self._extract_features(bytez=bytez, pe_binary=pe_binary)
            y_pred = np.sum(features, axis=1) > 0
            return [int(y_pred[0])]
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [0]

    def predict_proba(self, bytez, pe_binary=None):
        return [float(x) for x in self.predict(bytez=bytez, pe_binary=pe_binary)]

    def predict_all_proba(self, bytez, pe_binary=None):
        try:
            features = self._extract_features(bytez=bytez, pe_binary=pe_binary)
            y_pred = np.sum(features, axis=1) > 0
            return [ClassificationResult(prob_score=float(y_pred[0]), threshold=0, features=features)]
        except Exception as e:
            print("Error with: {}".format(str(e)), file=sys.stderr)
            return [ClassificationResult(prob_score=0.0, threshold=0, features=None)]
