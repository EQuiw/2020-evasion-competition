import os
import typing

import yaml
import sys

from .EmberModel import EmberModel
from .EmberModelTruncated import EmberModelTruncated
from .EmberModelCollection import EmberModelCollection
from .ExpertModel import ExpertModel
from .NgramModel import NgramModel
from .ClassifierInterface import ClassifierInterface
from .ClassificationResult import ClassificationResult


class Ensemble(ClassifierInterface):
    ''' Ensemble of multiple classifiers. '''

    @classmethod
    def from_yaml(cls, filename: str):
        '''Call __init__ with parameters from config file (YAML). '''
        with open(filename) as f:
            cnf = yaml.load(f, Loader=yaml.FullLoader)
        return cls(**cnf)

    def __init__(self, embermodelpath, expertmodelpath, ngrammodelpath, featureversion=2):
        # Now add all classifiers in a list that will be used to get predictions later..
        classifiers: typing.List[ClassifierInterface] = []

        classifiers.append(
            EmberModel(embermodelpath=os.path.join(embermodelpath, "2017/xgboost/mono_False"),
                       feature_version=featureversion, threshold=0.6171156764030457, include_header=False,
                       truncate=False, remove_strings_stream=False)
        )

        # Note! For efficency, we moved some EmberModel and EmberModelTruncated classes to
        # EmberModelCollection. In this class, the features are re-used and we do not need to extract
        # the features multiple times. This is faster. However, you should get the same
        # output by using the code below that is commented out -- instead of using EmberModelCollection.
        classifiers.append(
            EmberModelCollection(embermodelpath=embermodelpath, expertmodelpath=expertmodelpath,
                                 featureversion=featureversion, detectpacker=False)
        )

        # classifiers.append(
        #     EmberModel(embermodelpath=os.path.join(embermodelpath, "2017/xgboost/mono_False"),
        #                feature_version=featureversion, threshold=0.7620967626571655, include_header=False,
        #                truncate=True, remove_strings_stream=True)
        # )
        # classifiers.append(
        #     EmberModelTruncated(embermodelpath=os.path.join(embermodelpath, "2017_small_nodata/xgboost/mono_False"),
        #                         feature_version=featureversion, threshold=0.7652024626731873, truncate=True)
        # )
        # classifiers.append(
        #     EmberModelTruncated(embermodelpath=os.path.join(embermodelpath, "2018_small_nodata/xgboost/mono_False"),
        #                         feature_version=featureversion, threshold=0.7550212740898132, truncate=True)
        # )

        try:
            em = ExpertModel(dictvectorizerpath=os.path.join(expertmodelpath, "dict-vectorizer-yaru.pck"),
                             featureindicespath=os.path.join(expertmodelpath, "feature-indices-yaru.pck"),
                             features="yaru", configpath=os.path.join(expertmodelpath, "config.yml"))
            classifiers.append(em)
        except Exception as e:
            print("Error with init Expert Model {}".format(str(e)), file=sys.stderr)
            pass

        classifiers.append(
            NgramModel(ngrammodelpath=os.path.join(ngrammodelpath, 'skipgram-n3-skip3/xgboost/binary_False/mono_True'),
                       cfg_path=os.path.join(ngrammodelpath, 'config.yml'),
                       dim_file=os.path.join(ngrammodelpath, 'skipgram-n3-skip3_dims.json'),
                       threshold=0.8379436135292053,
                       command='skipgram', nlen=3, skiplen=3, binary=False, nocutting=True)
        )

        self.classifiers = classifiers

    def predict(self, bytez, pe_binary=None) -> list:
        preds = []
        for clf in self.classifiers:
            try:
                preds.extend(clf.predict(bytez=bytez, pe_binary=pe_binary))
            except Exception as e:
                print("Error with: {}".format(str(e)), file=sys.stderr)
                preds.extend([0])
        return preds

    def predict_proba(self, bytez, pe_binary=None) -> list:
        probas = []
        for clf in self.classifiers:
            try:
                probas.extend(clf.predict_proba(bytez=bytez, pe_binary=pe_binary))
            except Exception as e:
                print("Error with: {}".format(str(e)), file=sys.stderr)
                probas.extend([0.0])
        return probas

    def predict_all_proba(self, bytez, pe_binary=None):
        probas = []
        for clf in self.classifiers:
            try:
                probas.extend(clf.predict_all_proba(bytez=bytez, pe_binary=pe_binary))
            except Exception as e:
                print("Error with: {}".format(str(e)), file=sys.stderr)
                probas.append(ClassificationResult(prob_score=0.0, threshold=0, features=None))
        return probas

    def model_info(self):
        return "You are quite curios, aren't you?"
