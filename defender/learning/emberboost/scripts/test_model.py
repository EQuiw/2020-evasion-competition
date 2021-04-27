import argparse
import numpy
import learning.emberboost.ember.boost.boost_testing as boost_testing
from learning.emberboost.ember.boost.train_boost_utils import read_ember_features_from_disk
from learning.emberboost.ember import PEFeatureExtractor


# Call me like this: PYTHONPATH=/<PATH-TO>/defender/ python learning/emberboost/scripts/test_model.py -d /<PATH-TO>/ember_2017_2 --modelpath=/<PATH-TO>/<MODEL-DIR>
def main():
    prog = "test xgboost model"
    descr = "test xgboost model"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    # parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with vectorized features")
    parser.add_argument("-d", "--datadir", metavar="DATADIR", nargs='+', type=str, required=True,
                        help="Directory with vectorized features")
    parser.add_argument("--modeldir", metavar="MODELDIR", type=str, required=True,
                        help="Directory with model")
    args = parser.parse_args()

    modelpath = args.modeldir
    emberpaths = args.datadir

    extractor = PEFeatureExtractor(feature_version=2)
    X_train, y_train, X_test, y_test = read_ember_features_from_disk(emberpath=emberpaths[0], feature_version=2,
                                                                     extractor=extractor)
    for i in range(1, len(emberpaths)):
        Xtr, ytr, Xte, yte = read_ember_features_from_disk(emberpath=emberpaths[i], feature_version=2,
                                                           extractor=extractor)
        assert Xtr.shape[1] == X_train.shape[1] and len(ytr.shape) == len(y_train.shape)
        X_train = numpy.concatenate((X_train, Xtr), axis=0)
        y_train = numpy.concatenate((y_train, ytr), axis=0)
        X_test = numpy.concatenate((X_test, Xte), axis=0)
        y_test = numpy.concatenate((y_test, yte), axis=0)

    boost_testing.test_model(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, modelpath=modelpath)

if __name__ == "__main__":
    main()