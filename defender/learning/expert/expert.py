#!/usr/bin/env python3
# (c) 2020 Institute of System Security, TU Braunschweig

import argparse
import json
import os
import sys

import yaml
import yara


def parse_args():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description='Expert Feature Extractor')
    parser.add_argument('features', metavar='<features>',
                        help='feature set: peid, yaru, all')
    parser.add_argument('input', metavar='<input>',
                        help='PE file to analyze')
    parser.add_argument('output', metavar='<output>',
                        help='File with features')
    parser.add_argument('--config', metavar='<file>',
                        help='set config file',
                        default='config.yml')

    cfg = parser.parse_args()

    # Sanity checks
    if not os.path.exists(cfg.input):
        print("Error: File {} does not exist".format(cfg.input))
        sys.exit(-1)

    if not os.path.exists(cfg.config):
        print("Error: File {} does not exist".format(cfg.config))
        sys.exit(-1)

    if cfg.features not in ["peid", "yaru", "all"]:
        print("Error: Unknown feature set {}".format(cfg.features))
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


def flatten_data(data, prefix=None, delim='|'):
    """ Flatten a lists and dictionaries """
    flat = []

    if type(data) == dict:
        items = data.items()
    elif type(data) == list:
        items = zip(len(data) * [''], data)
    else:
        print("Error: Unknown data type {}".format(type(data)))

    for k, v in items:
        t = type(v)

        if prefix:
            if len(k) > 0:
                k = "{}{}{}".format(prefix, delim, k)
            else:
                k = prefix

        # Process basic types
        if t == str or t == int or t == float or t == bool:
            if len(k) > 0:
                v = "{}{}{}".format(k, delim, v)
            flat.append(v)

        # Process collections
        elif t == dict or t == list:
            s = flatten_data(v)
            if len(k) > 0:
                s = map(lambda x: "{}{}{}".format(k, delim, x), s)
            flat.extend(s)

        elif v is None:
            pass

        else:
            print("Warning: Unknown data type {}".format(t))

    return flat


def init_expert(cfg):
    """ Initialize expert feature extractor """

    if not os.path.exists(cfg.cache):
        os.makedirs(cfg.cache)

    if cfg.features in ["peid", "all"]:
        yfile = os.path.join(cfg.cache, "peid")
        if not os.path.exists(yfile):
            cfg.peid_rules = yara.compile(filepath=cfg.peid_db)
            cfg.peid_rules.save(yfile)
        else:
            cfg.peid_rules = yara.load(yfile)

    if cfg.features in ["yaru", "all"]:
        yfile = os.path.join(cfg.cache, "yaru")
        if not os.path.exists(yfile):
            cfg.yaru_rules = yara.compile(filepath=cfg.yaru_db)
            cfg.yaru_rules.save(yfile)
        else:
            cfg.yaru_rules = yara.load(yfile)


def extract_peid(cfg, data=None):
    """ Extract PeID features """

    # Check for rules, compiled during init
    assert (cfg.peid_rules)

    # Match input file
    if data is None:
        matches = cfg.peid_rules.match(cfg.input)
    else:
        matches = cfg.peid_rules.match(data=data)

    # Convert matches to names and flatten
    matches = list(map(lambda x: x.rule, matches))
    return flatten_data(matches, prefix="peid")


def extract_yaru(cfg, data=None):
    """ Extract Yara rules features """

    # Check for rules, compiled during init
    assert (cfg.yaru_rules)

    # Match input file
    if data is None:
        matches = cfg.yaru_rules.match(cfg.input)
    else:
        matches = cfg.yaru_rules.match(data=data)

    # Convert matches to names and flatten
    matches = list(map(lambda x: x.rule, matches))
    return flatten_data(matches, prefix="yaru")

if __name__ == "__main__":

    # Parse args and initialize
    cfg = parse_args()
    cfg = load_config(cfg)
    init_expert(cfg)

    try:
        # Extract selected features
        if cfg.features == "peid":
            feats = extract_peid(cfg)
        elif cfg.features == "yaru":
            feats = extract_yaru(cfg)
        elif cfg.features == "all":
            feats = {
                "peid2": extract_peid(cfg),
                "yaru": extract_yaru(cfg),
            }
        else:
            raise NotImplementedError()

        # Store features as json
        with open(cfg.output, "wt") as out:
            json.dump(feats, out)

    except Exception as e:
        print("Error occurred during extraction for: {} {}".format(cfg.input, str(e)), file=sys.stderr)
