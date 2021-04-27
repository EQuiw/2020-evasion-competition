#!/usr/bin/env python3
# (c) 2020 Institute of System Security, TU Braunschweig

import os
import sys
import argparse
import json
from functools import partial

from svmlight_loader import load_svmlight_file
from scipy.stats import spearmanr, pearsonr
import scipy.sparse as sp
import numpy as np
import joblib
from tqdm import tqdm


def parse_args():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description='Ngram Feature Selector')
    parser.add_argument('metric', metavar='[spearman|pearson]',
                        help='correlation metric to choose the best features')
    parser.add_argument('inputs', metavar='<file1> ... <fileN>', nargs='+',
                        help='name of the input LIBSVM files')
    parser.add_argument('--threshold', metavar='<float>', type=float, nargs='?', default=0.2,
                        help='threshold defining the minimum value of [spearman|pearson] correlation coefficient')
    parser.add_argument('--n_jobs', metavar='<n_jobs>', type=int, nargs='?', default=1,
                        help='number of parallel processes')
    parser.add_argument('--json', action='store_true',
                        help='write the output to stdout in JSON format')

    cfg = parser.parse_args()

    # Sanity check
    for filename in cfg.inputs:
        if not os.path.exists(filename):
            eprint("Error: File {} does not exist".format(filename))
            sys.exit(-1)

    return cfg


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def fast_libsvm_read_X(cfg, filename, ndims):
    eprint('[-] reading LIBSVM file using {} workers'.format(cfg.n_jobs))
    with open(filename, 'r') as f, joblib.parallel_backend('loky'):
        results = joblib.Parallel(n_jobs=cfg.n_jobs)(joblib.delayed(
            libsvm_parseline)(line, i) for i, line in tqdm(enumerate(f)))
    eprint('[-] transpose results')
    row, col, data = list(zip(*results))
    eprint('[-] merge results')
    row = np.hstack(row)
    col = np.hstack(col)
    data = np.hstack(data)

    eprint('[-] coo transformation')
    X = sp.coo_matrix((data, (row, col)), shape=(max(row) + 1, ndims))
    eprint('[-] csc transformation')
    X = X.tocsc()
    eprint('[-] LIBSVM read X finished')
    return X


def libsvm_parseline(line, row_idx):
    row = []
    col = []
    data = []

    if len(line) == 0 or line.startswith('#'):
        return np.array(row), np.array(col), np.array(data)

    line = line.split()[1:]  # discard label
    for entry in line:
        j, val = entry.split(':')
        j, val = int(j), int(val)
        row.append(row_idx)
        col.append(j)
        data.append(val)
    return np.array(row), np.array(col), np.array(data)


def libsvm_read_Y(filename):
    Y = []
    with open(filename, 'r') as f:
        for line in f:
            if len(line) == 0 or line.startswith('#'):
                continue
            Y.append(int(line[0]))
    return np.array(Y)


def select_features(cfg, X, Y):
    metrics = {
        'spearman': spearmanr,
        'pearson': pearsonr,
    }

    # the scipy import messes with the CPU affinity (this is a known bug)
    #   -> set the affinity manually again to unlock parallel computing
    eprint("[-] setting CPU affinity...")
    os.system("taskset -p 0x{} {} 1>&2".format(''.join(['ff']*cfg.n_jobs), os.getpid()))  # redirect stdout to stderr

    if cfg.metric in metrics:
        eprint("[-] calculating the {} coefficients for the dataset. this might take a while.".format(cfg.metric))

        worker = partial(generic_worker, metrics[cfg.metric])
        X = X.tocsc()
        with joblib.parallel_backend('loky', n_jobs=cfg.n_jobs):
            results = joblib.Parallel(n_jobs=cfg.n_jobs, batch_size=2**14)(joblib.delayed(
                worker)(i, X[:, i], Y) for i in tqdm(range(X.shape[1])))
        # results format: [(dim, value, p-value))]
        results = filter(lambda x: x[1] > cfg.threshold, results)
        # fixed threshold for rejecting null hypothesis (i.e. keeping the feature)
        results = filter(lambda x: x[2] < 0.01, results)

        dims = [x[0] for x in results]

        return dims
    else:
        raise NotImplementedError("Feature selection not implemented for {}!".format(cfg.metric))


def generic_worker(metric_fun, i, col, Y):
    col = col.A.reshape(-1)     # column to dense array
    val, pval = metric_fun(col, Y)   # calculate metric
    return i, val, pval


def export_to_json(dims):
    print(json.dumps(dims, indent=4))


def main(cfg):
    eprint('[-] loading data from {}'.format(cfg.inputs))
    # TODO: below is a nasty quickfix to identify the number of dims from filename
    nlen = int([part[-1] for part in cfg.inputs[0].split('-') if part.startswith('n') and str.isnumeric(part[-1])][0])
    dims = 256**nlen
    X = sp.csr_matrix((0, dims))
    Y = np.zeros((0,))
    for filename in cfg.inputs:
        # X_curr, Y_curr = load_svmlight_file(filename, n_features=dims)
        X_curr = fast_libsvm_read_X(cfg, filename, dims)
        Y_curr = libsvm_read_Y(filename)
        X = sp.vstack([X, X_curr])
        Y = np.hstack([Y, Y_curr])
    dims = select_features(cfg, X, Y)
    if cfg.json:
        export_to_json(dims)


if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
