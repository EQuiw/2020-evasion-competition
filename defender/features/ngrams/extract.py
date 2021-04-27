#!/usr/bin/env python3
# (c) 2020 Institute of System Security, TU Braunschweig

import argparse
import os
import joblib
import sys
import pickle


import scipy.sparse as sp

from ngrams import load_config, main as ngrams_main


def parse_args():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description='Parallelization script for ngram extraction')
    parser.add_argument('command', metavar='<command>', type=str,
                        help='command for ngram extraction [ngram|skipgram]')
    parser.add_argument('nlen', metavar='<len>', type=int,
                        help='n-gram length')
    parser.add_argument('--skiplen', metavar='<skiplen>', type=int, nargs='?', default=2,
                        help='length of skipping window')
    parser.add_argument('--vocab', metavar='<vocab>', nargs='?', default=None,
                        help='Vocabulary file containing the ngrams to consider')
    parser.add_argument('--output', metavar='<output>',
                         help='output file')
    parser.add_argument('--config', metavar='<file>',
                        help='set config file',
                        default='config.yml')
    parser.add_argument('--nocutting', action='store_true',
                        help='Process file without cutting high/low entropy sections')
    parser.add_argument('--inputdirs', metavar='<dir0> <dir1> ...', type=str,
                        nargs='+', help='directories to scan for pe files')
    parser.add_argument('--n_jobs', metavar='<int>', type=int, nargs='?', default=1,
                        help='number of parallel processes')

    cfg = parser.parse_args()

    # Sanity checks
    for d in cfg.inputdirs:
        if not os.path.exists(d):
            print("Error: Directory {} does not exist".format(d))
            sys.exit(-1)

    if not os.path.exists(cfg.config):
        print("Error: File {} does not exist".format(cfg.config))
        sys.exit(-1)

    if cfg.vocab is None:
        print("Error: Vocabulary must be provided!")
        sys.exit(-1)

    return cfg


def scan_dirs(inputdirs):
    for d in inputdirs:
        for root, dirs, files in os.walk(d):
            for f in files:
                filename = os.path.join(root, f)
                if is_pe_file(filename):
                    yield filename


def is_pe_file(filename):
    output = os.popen('file {}'.format(filename)).read()
    return 'PE' in output and 'executable' in output


def process_single(cfg, filepath):
    setattr(cfg, 'input', filepath)
    ngrams = ngrams_main(cfg)
    filename = filepath.split(os.sep)[-1]
    vocab = pickle.load(open(cfg.vocab, 'rb'))
    return cfg, filename, counter2csr(ngrams, vocab)


def counter2csr(ngrams, vocab):
    vec = sp.lil_matrix((1, len(vocab)))
    if ngrams is not None:
        for term, count in ngrams.items():
            if term in vocab:
                vec[0, vocab[term]] += count
    return vec.tocsr()


def extraction_cb(results):
    cfg = results[0][0]
    vocab = pickle.load(open(cfg.vocab, 'rb'))
    if os.path.exists(cfg.output):
        file_dict, ngrams = pickle.load(open(cfg.output, 'rb'))
    else:
        file_dict = {}  # maps filename to matrix row
        ngrams = sp.csr_matrix((0, len(vocab)))

    _, filenames, new_rows = list(map(list, zip(*results)))
    offset = ngrams.shape[0]
    for i, f in enumerate(filenames):
        file_dict[f] = offset + i
    ngrams = sp.vstack([ngrams] + new_rows)

    pickle.dump((file_dict, ngrams), open(cfg.output, 'wb'))


class ExtractionCallback(object):
    """
    Customizable callback for joblib parallelization.
    Requires cb_handler to be set to a function which shall be
    called upon finishing a batch of work.
    """
    cb_handler = None

    def __init__(self, dp_ts, b_size, parallel):
        self.dispatch_timestamp = dp_ts
        self.batch_size = b_size
        self.parallel = parallel

    def __call__(self, out):
        if isinstance(out, joblib._parallel_backends.ImmediateResult):
            # if n_jobs == 1
            results = out.results
        elif isinstance(out, joblib.externals.loky._base.Future):
            results = out.result()
        ExtractionCallback.cb_handler(results)
        if self.parallel._original_iterator is not None:
            self.parallel.dispatch_next()


def main(cfg):
    ExtractionCallback.cb_handler = extraction_cb
    joblib.parallel.BatchCompletionCallBack = ExtractionCallback
    with joblib.parallel_backend('loky', n_jobs=cfg.n_jobs):
        joblib.Parallel()(joblib.delayed(
            process_single)(cfg, f) for f in scan_dirs(cfg.inputdirs))


if __name__ == "__main__":
    # Parse args and initialize
    cfg = parse_args()
    cfg = load_config(cfg)
    main(cfg)
