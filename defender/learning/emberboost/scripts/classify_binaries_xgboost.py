#!/usr/bin/env python

import os
import numpy as np
import argparse
from xgboost import XGBClassifier
import learning.emberboost.ember.boost.boost_testing as boost_testing
from learning.emberboost.ember import PEFeatureExtractor


def main():
    prog = "classify_binaries using xgboost"
    descr = "Use a trained ember model to make predictions on PE files"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-v", "--featureversion", type=int, default=2, help="EMBER feature version")
    parser.add_argument("-m", "--modelpath", type=str, default=None, required=True, help="Ember model")
    parser.add_argument('--threshold', type=float, default=None, help='if threshold is given, we return binary classification')
    parser.add_argument("binaries", metavar="BINARIES", type=str, nargs="+", help="PE files to classify")
    args = parser.parse_args()

    if not os.path.exists(args.modelpath):
        parser.error("ember model {} does not exist".format(args.modelpath))

    clf: XGBClassifier = XGBClassifier()
    clf.load_model(fname=os.path.join(args.modelpath, "model_xgboost.dat"))

    for binary_path in args.binaries:
        if not os.path.exists(binary_path):
            print("{} does not exist".format(binary_path))

        file_data = open(binary_path, "rb").read()

        # score = ember.predict_sample(lgbm_model, file_data, args.featureversion)
        extractor = PEFeatureExtractor(feature_version=args.featureversion)
        features = np.array(extractor.feature_vector(file_data), dtype=np.float32)
        features = np.expand_dims(features, axis=0)

        if args.threshold is not None:
            score = boost_testing.predict_class(clf=clf, samples=features, threshold=float(args.threshold))
        else:
            score = clf.predict_proba(features)[:, 1]

        # print("\t".join((binary_path, str(score))) # TODO
        print("\t".join((binary_path, str(score), str(clf.predict_proba(features)[0][1]))))


if __name__ == "__main__":
    main()
