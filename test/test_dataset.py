import os
import unittest
import time
import numpy as np
import requests
import json
import argparse

URL = 'http://127.0.0.1:8080'
HEADERS = {
    'Content-Type': 'application/octet-stream'
}


MAX_BYTES = 2097152


def parse_args():
    """ Parse command-line arguments """
    prog = "Test of Defense"
    descr = "Test Defense on the MLSec Dataset"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("-d", "--datasetpath", type=str, required=True,
                        help="Full path to MLSEC_2019_samples_and_variants dir")
    parser.add_argument('--verbose', action='store_true',
                        help='Output some more information')
    args = parser.parse_args()
    return args


def test_dataset(dataset_dir: str, verbose: bool):
    binaries = np.sort(
        [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if
         os.path.isfile(os.path.join(dataset_dir, f))])

    results = []
    for binary_path in binaries:
        if not os.path.exists(binary_path):
            print("{} does not exist".format(binary_path), file=sys.stderr)
            continue

        with open(os.path.join(binary_path), 'rb') as f:
            bytez = f.read()

        response_server = requests.post(URL, data=bytez, headers=HEADERS)
        content = json.loads(response_server.text)
        response = content['result']

        if verbose:
            print(binary_path, response)
        results.append(response)
    print("Overall Classification Accuracy: ", dataset_dir, np.mean(results))


if __name__ == '__main__':
    args = parse_args()
    args_dataset_dir: str = args.datasetpath
    args_verbose: bool = args.verbose
    test_dataset(dataset_dir=args_dataset_dir, verbose=args_verbose)
