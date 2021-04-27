"""
Stand-alone script to test stateful defense.
"""

import os
import numpy as np
import sys
import time
import lief

from learning.ensemble.Models.Ensemble import Ensemble
from learning.statefuldefense.stateful.StatefulDefense import StatefulDefense
from learning.statefuldefense.stateful.StatefulAnnoyDefense import StatefulAnnoyDefense

# A. You NEED to specify here path information!
# 1. Specify the directory where data_dirs are located!
DATA_PATH = "<TODO>"
REPO_PATH = "/<TODO>/2020-ms-adv-challenge-public/"

# 2. Specify the directories in DATA_PATH where Windows PE files are located. We will iterate over them!
data_dirs = ["MLSEC_2019_samples_and_variants/",  # just examples for dirs that contain PE files
             "pe-files-benign/",
             "pe-files-malicious/"]
# 3. Max files per dir that are tested
MAX_BINARIES_PER_DIR: int = 1000

# B. Set Paths & Settings
MODELPATH = os.path.join(REPO_PATH, "defender/learning/emberboost/models/")
EXPERTPATH = os.path.join(REPO_PATH, "defender/learning/expert/models/")
NGRAMSPATH = os.path.join(REPO_PATH, "defender/learning/ngrams/models/")

BINARIES = [os.path.join(DATA_PATH, d) for d in data_dirs]
verbose: bool = True

featureversion: int = 2
embermodelpath: str = MODELPATH
expertmodelpath: str = EXPERTPATH
ngrammodelpath: str = NGRAMSPATH

# C. Test Ensemble & Stateful Defense
ensemble: Ensemble = Ensemble(embermodelpath=embermodelpath, expertmodelpath=expertmodelpath,
                              featureversion=featureversion, ngrammodelpath=ngrammodelpath)

for selected_binaries in BINARIES:
    binaries = np.sort(
        [os.path.join(selected_binaries, f) for f in os.listdir(selected_binaries) if
         os.path.isfile(os.path.join(selected_binaries, f)) and f.endswith("")])

    overall_ = []
    time_measures = []

    statefulDefense: StatefulDefense = StatefulAnnoyDefense(seed=4878, verbose=False)

    for binary_path in binaries[:MAX_BINARIES_PER_DIR]:
        if not os.path.exists(binary_path):
            print("{} does not exist".format(binary_path), file=sys.stderr)
            continue

        preptime = time.time()
        file_data = open(binary_path, "rb").read()
        lief_binary = lief.PE.parse(list(file_data))

        # ensembl = ensemble.extract_features(bytez=file_data, pe_binary=lief_binary)
        start = time.time()
        preds = ensemble.predict(bytez=file_data, pe_binary=lief_binary)
        mid = time.time()
        preds_proba = ensemble.predict_proba(bytez=file_data, pe_binary=lief_binary)
        end = time.time()

        stateful_response: tuple = statefulDefense.check(score=np.max(preds_proba), ismalware=np.max(preds) == 1,
                              features=ensemble.classifiers[0].extract_features(bytez=file_data, pe_binary=lief_binary),
                              bytez=file_data, pe_binary=lief_binary)
        if stateful_response[0] is True:
            print(binary_path, "was detected as attack")

