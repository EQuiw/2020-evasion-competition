import os
from xgboost import XGBClassifier
from learning.emberboost.ember.featuresvariant.featurestruncated import PEFeatureExtractorTruncated
from learning.ensemble.Models.EmberModel import EmberModel


class EmberModelTruncated(EmberModel):

    def __init__(self, embermodelpath: str, threshold: float, feature_version: int, truncate: bool = True):
        # no super call!!
        self.embermodelpath = embermodelpath
        self.threshold = threshold

        self.clf: XGBClassifier = XGBClassifier()
        self.clf.load_model(fname=os.path.join(embermodelpath, "model_xgboost.dat"))

        self.extractor = PEFeatureExtractorTruncated(feature_version=feature_version, include_header=False)
        self.truncate = truncate
