#!/usr/bin/env python3
# (c) 2020 Institute of System Security, TU Braunschweig

import argparse
import collections
import math
import os
import sys
import json
import yaml


def parse_args():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description='Ngram Feature Extractor')
    subparser = parser.add_subparsers(help='commands')
    parser_ngram = subparser.add_parser('ngram')
    parser_ngram.add_argument('nlen', metavar='<len>', type=int,
                              help='n-gram length')
    parser_ngram.set_defaults(command='ngram')
    parser_skipgram = subparser.add_parser('skipgram')
    parser_skipgram.add_argument('nlen', metavar='<len>', type=int,
                                 help='skip-gram length')
    parser_skipgram.add_argument('skiplen', metavar='<skiplen>', type=int,
                                 help='length of skipping window')
    parser_skipgram.set_defaults(command='skipgram')
    parser.add_argument('input', metavar='<input>',
                        help='PE file to analyze')
    parser.add_argument('--dim_file', metavar='<file>', nargs='?', default=None,
                        help='File containing the dimensions to extract')
    # parser.add_argument('output', metavar='<output>',
    #                     help='File with features')
    parser.add_argument('--config', metavar='<file>',
                        help='set config file',
                        default='config.yml')
    parser.add_argument('--label', metavar='<int>', type=int, default=-1,
                        help='label (only used for libsvm)')
    parser.add_argument('--nocutting', action='store_true',
                        help='Process file without cutting high/low entropy sections')
    parser.add_argument('--libsvm', action='store_true',
                        help='Transform output to libsvm format')

    cfg = parser.parse_args()

    # Sanity checks
    if not os.path.exists(cfg.input):
        print("Error: File {} does not exist".format(cfg.input))
        sys.exit(-1)

    if not os.path.exists(cfg.config):
        print("Error: File {} does not exist".format(cfg.config))
        sys.exit(-1)

    if cfg.libsvm and cfg.label == -1:
        print("Error: Please provide a label when exporting to LIVSVM format!")
        sys.exit(-1)

    return cfg


def load_config(cfg):
    """ Load config """
    with open(cfg.config) as f:
        c = yaml.load(f, Loader=yaml.FullLoader)

    d = vars(cfg)
    for k, v in c.items():
        if k not in cfg:
            d[k] = v

    return cfg


def init_ngrams(cfg):
    """ Initialize n-gram feature extractor """
    pass


def extract_ngrams(cfg, chunks, dims=None):
    """ Extract n-grams from chunk """
    ngrams = collections.Counter()

    for chunk in chunks:
        for i in range(len(chunk) - cfg.nlen + 1):
            ngram = tuple(chunk[i:i + cfg.nlen])
            # if dims is provided, check if the embedding dimension is included
            if dims is None or int.from_bytes(bytes(ngram), 'big') in dims:
                ngrams[ngram] += 1

    return ngrams


def extract_skipgrams(cfg, chunks, dims=None):
    """ Extract q-grams (q - skipping window) from chunk """
    skipgrams = collections.Counter()

    for chunk in chunks:
        for i in range(len(chunk) - (cfg.nlen - 1) * cfg.skiplen):
            skipgram = tuple(chunk[i:i+cfg.nlen*cfg.skiplen:cfg.skiplen])
            # if dims is provided, check if the embedding dimension is included
            if dims is None or int.from_bytes(bytes(skipgram), 'big') in dims:
                skipgrams[skipgram] += 1

    return skipgrams


def get_entropy(cfg, chunk):
    """ Get entropy of chunk """
    byte_counts = collections.Counter()
    byte_ent = {}
    ents = []
    window = cfg.ent_window

    for i in range(len(chunk)):
        byte_counts[chunk[i]] += 1

        # recompute all entropy summands because total changed
        if i < window:
            ent = 0
            total = min(i + 1, window)
            for val in byte_counts.values():
                freq = val / total
                ent = ent + freq * math.log(freq, 2)
        # streaming mode for i >= window
        else:
            # calc all byte entropies on first streaming iteration
            if i == window:
                for b, val in byte_counts.items():
                    freq = val / window
                    byte_ent[b] = freq * math.log(freq, 2)

            c = chunk[i - window]
            byte_counts[c] -= 1
            if byte_counts[c] == 0:
                del byte_counts[c]
                del byte_ent[c]
            else:
                # byte which has left the window (but freq != 0)
                old_freq = byte_counts[c] / window
                byte_ent[c] = old_freq * math.log(old_freq, 2)
            # entropy summand of the current byte
            new_freq = byte_counts[chunk[i]] / window
            byte_ent[chunk[i]] = new_freq * math.log(new_freq, 2)
            ent = sum(byte_ent.values())

        ents.append(-ent)
    return ents


def cut_entropy(cfg, data):
    """ Cut chunk based on entropy """
    ents = get_entropy(cfg, data)
    chunks, chunk = [], []

    for b, e in zip(data, ents):
        if cfg.ent_min < e < cfg.ent_max:
            chunk.append(b)
        elif len(chunk) < cfg.nlen:
            continue
        else:
            chunks.append(chunk)
            chunk = []

    if len(chunk) > 0:
        chunks.append(chunk)

    return chunks


def filter_ngrams(ngrams, vocab):
    """ Remove terms from [ngrams] Counter that are not in the vocabulary. """
    for term in list(ngrams.keys()):
        if term not in vocab:
            del ngrams[term]


def main(cfg, bytez=None):
    init_ngrams(cfg)

    if bytez is None:
        with open(cfg.input, "rb") as f:
            data = f.read()
    else:
        data = bytez

    if cfg.nocutting:
        chunks = [data]
    else:
        chunks = cut_entropy(cfg, data)

    if cfg.dim_file is not None:
        dims = json.load(open(cfg.dim_file, 'r'))
    else:
        dims = None

    if cfg.command == 'ngram':
        output = extract_ngrams(cfg, chunks, dims)
    elif cfg.command == 'skipgram':
        output = extract_skipgrams(cfg, chunks, dims)
    else:
        raise NotImplementedError()

    # dump output to pickle file
    # pickle.dump(output, open(cfg.output, 'wb'))
    # return output
    if cfg.libsvm:
        export_to_libsvm(cfg, output)
    # export_to_jsonl(output=output, inputfile=cfg.input, normalize=False)
    return output


def export_to_jsonl(output, inputfile, normalize: bool):

    # normalize
    if normalize is True:
        total = sum(output.values())
        for key in output:
            output[key] /= total

    # export to jsonl
    output_dict = {}
    for key in output:
        output_dict[str(key).replace(' ', '')] = output[key]

    output_dict['input'] = inputfile

    print(json.dumps(output_dict))


def export_to_libsvm(cfg, output):
    items = []
    for term, count in output.items():
        dim = int.from_bytes(bytes(term), 'big')
        items.append((dim, count))
    items = sorted(items, key=lambda item: item[0])
    libsvm_str = ' '.join(['{}:{}'.format(dim, count) for dim, count in items])
    print('{} {}'.format(cfg.label, libsvm_str))


if __name__ == "__main__":
    # Parse args and initialize
    cfg = parse_args()
    cfg = load_config(cfg)
    main(cfg)