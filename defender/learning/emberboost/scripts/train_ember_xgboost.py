import learning.emberboost.ember as ember
import argparse
import numpy
import learning.emberboost.ember.boost.boost_learning as boost_learning
import os
from learning.emberboost.ember.boost.train_boost_utils import read_ember_features_from_disk
from learning.emberboost.ember import PEFeatureExtractor


# Call me like this: PYTHONPATH=/<PATH-TO>/defender/ python learning/emberboost/scripts/train_ember_xgboost.py -d /<PATH-TO>/ember_2017_2 /<PATH-TO>/ember2018 --resultdir=/dev/shm --mono --gpu=2
def main():
    prog = "train_ember_xgboost"
    descr = "Train an ember model from a directory with raw feature files; BY USING xgboost"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-d", "--datadir", metavar="DATADIR", nargs='+', type=str, required=True, help="Directory with vectorized features")
    parser.add_argument("--resultdir", metavar="RESULTDIR", type=str, required=True,
                        help="Directory to save models. Creates <dir>/xgboost/mono_{True,False}/")
    # parser.add_argument("--optimize", help="gridsearch to find best parameters", action="store_true")
    parser.add_argument("--mono", help="activate monotonic constraint", action="store_true")
    parser.add_argument('--gpu', type=int, default=-1, help='select unique ID of GPU if GPU should be used')

    args = parser.parse_args()

    emberpaths = args.datadir

    extractor = PEFeatureExtractor(feature_version=2)
    X_train, y_train, _, _ = read_ember_features_from_disk(emberpath=emberpaths[0], feature_version=2,
                                                           extractor=extractor)
    for i in range(1, len(emberpaths)):
        Xtr, ytr, _, _ = read_ember_features_from_disk(emberpath=emberpaths[i], feature_version=2, extractor=extractor)
        assert Xtr.shape[1] == X_train.shape[1] and len(ytr.shape) == len(y_train.shape)
        X_train = numpy.concatenate((X_train, Xtr), axis=0)
        y_train = numpy.concatenate((y_train, ytr), axis=0)

    resultdir = os.path.join(args.resultdir, "xgboost", "mono_" + str(args.mono))
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    # decide if we want to use the GPU, number specifies GPU-ID, and we will automatically select gpu_hist as param.
    gpu_id = args.gpu if args.gpu != -1 else None

    boost_learning.train(X_train=X_train, y_train=y_train, usemonotonic=args.mono,
                                     resultsdir=resultdir, bo_iters=250, bo_init_points=10, noofthreads=12,
                                     gpu_id=gpu_id)


if __name__ == "__main__":
    main()