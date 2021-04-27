import os
import numpy as np
import json
import typing


def extract_for(cur_path: str) -> typing.List[typing.Dict[str, int]]:
    """
    Extracts peid or yaru features from path.
    Use extract_expert.sh to get json files.
    """
    allparseditems = []

    onlyfiles = np.sort([f for f in os.listdir(cur_path) if os.path.isfile(os.path.join(cur_path, f)) and f.endswith(".txt")])
    assert len(onlyfiles) > 0

    for i, curfile in enumerate(onlyfiles):

        curreport = open(os.path.join(cur_path, curfile), "r").read()
        assert (type(curreport) == str)

        curreport_parsed = json.loads(curreport)
        assert type(curreport_parsed) is list

        curreport_dict = {}
        for x in curreport_parsed:
            curreport_dict[x] = 1

        allparseditems.append(curreport_dict)

    return allparseditems


def vectorize_data(benign_path: str, malware_path: str):
    """
    Extract and vectorize features.
    Benign_path and malware_path are output directories from extract_expert.sh script.
    """
    benign_features_raw = extract_for(cur_path=benign_path)
    malware_features_raw = extract_for(cur_path=malware_path)

    from sklearn.feature_extraction import DictVectorizer
    dictv = DictVectorizer(sparse=False)

    final_features_raw = benign_features_raw + malware_features_raw

    dictv.fit(final_features_raw, None)

    learning_features = dictv.transform(final_features_raw)
    labels = np.concatenate((np.repeat(0, len(benign_features_raw)), np.repeat(1, len(malware_features_raw))))

    return dictv, learning_features, labels


def filter_malware_indices_features(X_data, y_data):
    """
    Keep only features that are present in malware.
    """
    malwindices = np.where((X_data[y_data == 1].sum(axis=0) > 0) & (X_data[y_data == 0].sum(axis=0) == 0))[0]
    return malwindices
