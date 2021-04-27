"""
Stand-alone script to test ensemble.
"""
import os
import numpy as np
import sys
import time
import lief
import scipy.stats

from learning.ensemble.Models.Ensemble import Ensemble

MODELPATH="<TODO>/2020-ms-adv-challenge-public/defender/learning/emberboost/models/"
EXPERTPATH="<TODO>/2020-ms-adv-challenge-public/defender/learning/expert/models/"
NGRAMSPATH = "<TODO>/2020-ms-adv-challenge-public/defender/learning/ngrams/models/"
BINARIES = ["<TODO>/MLSEC_2019_samples_and_variants/",  # just examples for dirs that contain PE files
            "<TODO>/pe-files-benign/",
            "<TODO>/pe-files-malicious/"]
verbose: bool = True


def main():
    featureversion: int = 2
    embermodelpath: str = MODELPATH
    expertmodelpath: str = EXPERTPATH
    ngrammodelpath: str = NGRAMSPATH

    ensemble: Ensemble = Ensemble(embermodelpath=embermodelpath, expertmodelpath=expertmodelpath,
                                  featureversion=featureversion, ngrammodelpath=ngrammodelpath)

    for selected_binaries in BINARIES:
        binaries = np.sort(
            [os.path.join(selected_binaries, f) for f in os.listdir(selected_binaries) if
             os.path.isfile(os.path.join(selected_binaries, f)) and f.endswith("")])

        overall_ = []
        time_measures = []
        for binary_path in binaries:
            if not os.path.exists(binary_path):
                print("{} does not exist".format(binary_path), file=sys.stderr)
                continue

            preptime = time.time()
            file_data = open(binary_path, "rb").read()
            lief_binary = lief.PE.parse(list(file_data))

            start = time.time()
            preds = ensemble.predict(bytez=file_data, pe_binary=lief_binary)
            mid = time.time()
            preds_proba = ensemble.predict_proba(bytez=file_data, pe_binary=lief_binary)
            end = time.time()

            if verbose is True:
                preds_sum = np.sum(np.array(preds) == 1.0)
                print("\t".join((binary_path,
                                 f"{preds_sum}",
                                 f"{preds}",
                                 f"{preds_proba}",
                                 f"{round(start - preptime, 2)}s",
                                 f"{round(end - mid, 2)}s",
                                 f"{round(mid - start, 2)}s",
                                 )))

            overall_.append(np.sum(np.array(preds) == 1.0) > 0)
            time_measures.append((start - preptime) + (mid - start))  # only count parse and 1x predict

        print("Overall: ", selected_binaries, np.mean(overall_))
        print("Time Stats:", selected_binaries, scipy.stats.describe(time_measures))


if __name__ == "__main__":
    main()
