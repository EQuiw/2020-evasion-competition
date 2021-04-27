import os
import argparse
import sys
import json

import numpy as np
import scipy.sparse as sp
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier

import learning.emberboost.ember.boost.boost_learning as boost_learning
import learning.emberboost.ember.boost.boost_testing as boost_testing


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_features(ngrampaths, binary=False, dimfile=None):
    print('[-] loading complete dataset from LIBSVM file. this might take a while...')
    X, Y = load_svmlight_files(ngrampaths)
    if binary:
        X = binarize(X)
    if dimfile is not None:
        dims = np.array(json.load(open(dimfile, 'r')))
        # perform column slicing on column-major format
        X = X.tocsc()
        X = X[:, dims]
        X = X.tocsr()

    sss = StratifiedShuffleSplit(n_splits=1, random_state=42)
    train_idx, test_idx = list(sss.split(X, Y))[0]
    X_train, y_train, X_test, y_test = X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]
    return X_train, y_train, X_test, y_test


def binarize(X):
    if not isinstance(X, sp.csr_matrix):
        eprint('[-] expected matrix to be a sparse CSR matrix, but found {}'.format(type(X)))
    X.data = np.clip(X.data, 0, 1)
    return X


# Call me like this: PYTHONPATH=$(pwd) python scripts/train_ngram_xgboost.py /path/to/libsvm_dataset.libsvm --resultdir=/dev/shm --binary --mono --gpu=2
def main():
    prog = "train_ngram_xgboost"
    descr = "Train an ember model from a directory with raw feature files; BY USING xgboost"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument('inputs', metavar='<file1> ... <fileN>', nargs='+',
                        help='name of the input LIBSVM files')
    parser.add_argument("--resultdir", metavar="RESULTDIR", type=str, required=True,
                        help="Directory to save models. Creates <dir>/xgboost/binary_{True,False}/mono_{True,False}/")
    # parser.add_argument("--optimize", help="gridsearch to find best parameters", action="store_true")
    parser.add_argument("--binary", help="whether to use binary ngrams", action="store_true")
    parser.add_argument("--mono", help="activate monotonic constraint", action="store_true")
    parser.add_argument('--gpu', type=int, default=-1, help='select unique ID of GPU if GPU should be used')
    parser.add_argument('--dimfile', type=str, default=None, help='provide dim file for reduced features')
    parser.add_argument('--figdir', type=str, nargs="?", default=None, help='plot ROC curve to a file in this directory')
    parser.add_argument('--fpr', type=float, help='target false positive rate (FPR)')

    args = parser.parse_args()

    # Sanity check
    for filename in args.inputs:
        if not os.path.exists(filename):
            eprint("Error: File {} does not exist".format(filename))
            sys.exit(-1)

    X_train, y_train, X_test, y_test = read_features(args.inputs, binary=args.binary, dimfile=args.dimfile)

    param_str = os.path.join("xgboost", "binary_" + str(args.binary), "mono_" + str(args.mono))
    resultdir = os.path.join(args.resultdir, param_str)
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    # decide if we want to use the GPU, number specifies GPU-ID, and we will automatically select gpu_hist as param.
    gpu_id = args.gpu if args.gpu != -1 else None

    # skip training if model exists already
    if not os.path.exists(os.path.join(resultdir, "model_xgboost.dat")):
        boost_learning.train(X_train=X_train, y_train=y_train, usemonotonic=args.mono,
                             resultsdir=resultdir, bo_iters=400, bo_init_points=10, noofthreads=12,
                             gpu_id=gpu_id)
    else:
        eprint('[-] model is already trained. Skipping training...')

    # find threshold value at 0.001 FPR
    clf: XGBClassifier = XGBClassifier()
    clf.load_model(fname=os.path.join(resultdir, "model_xgboost.dat"))
    ypred_test = clf.predict(X_test)
    ypred_test_proba = clf.predict_proba(X_test)[:, 1]
    ngram_info = args.input.rsplit(os.sep, 1)[1].rsplit('.', 1)[0]  # remove directory and libsvm extension
    title = os.path.join(ngram_info, param_str).replace(os.sep, '-')
    figfile = None if args.figdir is None else os.path.join(args.figdir, title)
    thrx_test = boost_testing.get_scores(y_test=y_test, ypred=ypred_test, ypred_proba=ypred_test_proba,
                                         title=title, threshold_fpr=args.fpr,
                                         figfile=figfile)
    if figfile is not None:
        eprint('[-] saved ROC figure to {}'.format(figfile))

    eprint('[-] evaluation finished. Threshold value at {}: {}'.format(args.fpr, thrx_test))


if __name__ == "__main__":
    main()
