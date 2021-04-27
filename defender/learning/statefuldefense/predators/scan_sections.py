"""
Stand-alone script to test semantic gap checks.
"""
import lief
import typing
import time
import numpy as np
import os

from learning.statefuldefense.predators.DuplicateSimpleScanner import DuplicateSimpleScanner
from learning.statefuldefense.predators.SlackSpaceScanner import SlackSpaceScanner
from learning.statefuldefense.predators.OverlayScanner import OverlayScanner
from learning.statefuldefense.predators.PredatorInterface import PredatorInterface

if __name__ == "__main__":
    # HERE you need to specify path to data dir with PE files, e.g.:
    # If your files have a file ending, such as .exe, then use the following tuple
    # ("<DATA-EXAMPLE>/.../windows-files/", "exe"). If no file ending, use something
    # like below!
    check_paths = ("<DATA-TODO>/MLSEC_2019_samples_and_variants/", "")


    predScanners: typing.List[PredatorInterface] = []
    predScanners.append(DuplicateSimpleScanner(min_number_matches=2, verbose=False))
    predScanners.append(SlackSpaceScanner(min_number_absolute_matches=3,
                                          min_number_relative_matches=0.333334, verbose=False))
    predScanners.append(OverlayScanner(min_ratio=0.75, verbose=False))

    verbose = True

    matched_count = 0
    variants_path = check_paths[0]
    file_suffix = check_paths[1]
    time_meas = []

    onlyfiles = np.sort(
            [f for f in os.listdir(variants_path) if os.path.isfile(os.path.join(variants_path, f)) and f.endswith(file_suffix)])
    for i, curfile in enumerate(onlyfiles):
        file_data = open(os.path.join(variants_path, curfile), "rb").read()

        binary = lief.parse(file_data)

        preds = []
        for predScanner in predScanners:
            start = time.time()
            if predScanner.check_file(bytez=file_data, lief_binary=binary) is True:
                preds.append(1)
            else:
                preds.append(0)
            end = time.time()
            time_meas.append(end - start)

        if np.sum(preds) > 0:
            if verbose:
                print(curfile, "has matched", preds)
            matched_count += 1

    print("Matches:", matched_count)
    print("Time Measure:", np.mean(time_meas), np.max(time_meas), np.min(time_meas), np.median(time_meas))