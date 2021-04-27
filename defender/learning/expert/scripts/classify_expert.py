import os
import numpy as np

import pickle
import argparse
import sys
from learning.expert.expert import extract_peid, extract_yaru, load_config, init_expert

prog = "export-yaru-peid"
descr = "Classify by using peid or yaru features"
parser = argparse.ArgumentParser(prog=prog, description=descr)
parser.add_argument('input', metavar='<input>', help='PE file to analyze')
parser.add_argument("dictvectorizer", metavar="DICTVEC", type=str, help="Directory with benign features")
parser.add_argument("featureindices", metavar="FEATNAMESVEC", type=str, help="Directory with malware features")
parser.add_argument('features', metavar='<features>', help='feature set: peid, yaru')
parser.add_argument('--config', metavar='<file>',
                        help='set config file',
                        default='config.yml')
cfg = parser.parse_args()

feature_type = cfg.features
if cfg.features not in ["peid", "yaru"]:
    print("Error: Unknown feature set {}".format(cfg.features))
    sys.exit(-1)
# Sanity checks
if not os.path.exists(cfg.input):
    print("Error: File {} does not exist".format(cfg.input))
    sys.exit(-1)

if not os.path.exists(cfg.config):
    print("Error: File {} does not exist".format(cfg.config))
    sys.exit(-1)


try:
    dictv = pickle.load(open(os.path.join(cfg.dictvectorizer), "rb"))
    featureindices = pickle.load(open(os.path.join(cfg.featureindices), "rb"))

    # 1. Extract features from file
    cfg = load_config(cfg)
    init_expert(cfg)

    if cfg.features == "peid":
        feats = extract_peid(cfg)
    elif cfg.features == "yaru":
        feats = extract_yaru(cfg)

    curreport_dict = {}
    for x in feats:
        curreport_dict[x] = 1

    learning_features = dictv.transform(curreport_dict)
    learning_features = learning_features[:, featureindices]

    y_pred = np.sum(learning_features, axis=1) > 0
    print(cfg.input, int(y_pred[0]))

except Exception as e:
    print("Error:", str(e))
    print(cfg.input, 0)


