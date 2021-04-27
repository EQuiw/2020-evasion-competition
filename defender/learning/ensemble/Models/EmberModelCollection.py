import numpy as np
import os
import sys
import typing

from .EmberModel import EmberModel
from .EmberModelTruncated import EmberModelTruncated

from learning.ensemble.Models.ClassifierInterface import ClassifierInterface
from .ClassificationResult import ClassificationResult


class EmberModelCollection(ClassifierInterface):
    """
    This is an optimized ember models collection, where we re-use the features extracted once.
    ONLY modify this class if you know what you're doing, because the features must be compatible!
    """

    def __init__(self, embermodelpath, expertmodelpath, featureversion=2, detectpacker: bool = False):
        self.add_classifiers: typing.List[typing.Tuple[EmberModel, np.ndarray]] = []
        self.detectpacker = detectpacker

        # we will extract the features from this model
        self.base_classifier = \
            EmberModel(embermodelpath=os.path.join(embermodelpath, "2017/xgboost/mono_False"),
                       feature_version=featureversion, threshold=0.7620967626571655, include_header=False,
                       truncate=True, remove_strings_stream=True)

        # and re-use the features from base-model for the following classifiers, but probably remove some features.
        self.add_classifiers.append((
            EmberModelTruncated(embermodelpath=os.path.join(embermodelpath, "2017_small_nodata/xgboost/mono_False"),
                                feature_version=featureversion, threshold=0.7652024626731873, truncate=True),
            np.concatenate((np.arange(512, 515), np.arange(611, 2289))))
        )
        self.add_classifiers.append((
            EmberModelTruncated(embermodelpath=os.path.join(embermodelpath, "2018_small_nodata/xgboost/mono_False"),
                                feature_version=featureversion, threshold=0.7550212740898132, truncate=True),
            np.concatenate((np.arange(512, 515), np.arange(611, 2289))))
        )

        if self.detectpacker is True:
            raise NotImplementedError() # removed here.


    def predict(self, bytez, pe_binary=None) -> list:
        preds = []

        try:
            base_features = self.base_classifier.extract_features(bytez=bytez, pe_binary=pe_binary)
            preds.extend(self.base_classifier.predict_from_features(features=base_features))

            for clf, featrange in self.add_classifiers:
                try:
                    new_feature_vector = base_features[:, featrange]

                    preds.extend(clf.predict_from_features(features=new_feature_vector))
                except Exception as e:
                    print("Error with: {}".format(str(e)), file=sys.stderr)
                    preds.extend([0])

            if self.detectpacker is True:
                raise NotImplementedError()

        except Exception as e:
            print("Severe Error with: {}".format(str(e)), file=sys.stderr)
            preds.extend([0])

        return preds

    def predict_proba(self, bytez, pe_binary=None) -> list:
        preds = self.predict_all_proba(bytez=bytez, pe_binary=pe_binary)
        return [x.prob_score for x in preds]

    def predict_all_proba(self, bytez, pe_binary=None):
        preds: typing.List[ClassificationResult] = []

        try:
            base_features = self.base_classifier.extract_features(bytez=bytez, pe_binary=pe_binary)
            preds.append(ClassificationResult(prob_score=self.base_classifier.predict_proba_from_features(features=base_features)[0],
                                              threshold=self.base_classifier.threshold,
                                              features=base_features))

            for clf, featrange in self.add_classifiers:
                try:
                    new_feature_vector = base_features[:, featrange]

                    preds.append(
                        ClassificationResult(prob_score=clf.predict_proba_from_features(features=new_feature_vector)[0],
                                             threshold=clf.threshold,
                                             features=new_feature_vector)
                    )
                except Exception as e:
                    print("Error with: {}".format(str(e)), file=sys.stderr)
                    preds.append(ClassificationResult(prob_score=0.0, threshold=clf.threshold, features=None))

            if self.detectpacker is True:
                raise NotImplementedError()

        except Exception as e:
            print("Severe Error with: {}".format(str(e)), file=sys.stderr)
            preds.append(ClassificationResult(prob_score=0.0, threshold=0, features=None))
        return preds

