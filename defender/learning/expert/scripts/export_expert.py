import os
import numpy as np
import learning.expert.lib.utilsExpertFeatures
import pickle
import argparse
import sys

prog = "export-yaru-peid"
descr = "Export peid or yaru features"
parser = argparse.ArgumentParser(prog=prog, description=descr)
parser.add_argument("benign_dir", metavar="BENIGNDIR", type=str, help="Directory with benign features")
parser.add_argument("malware_dir", metavar="MALWDIR", type=str, help="Directory with malware features")
parser.add_argument('features', metavar='<features>', help='feature set: peid, yaru')
parser.add_argument("output_dir", metavar="OUTDIR", type=str, help="Output directory")
args = parser.parse_args()

if args.features not in ["peid", "yaru"]:
    print("Error: Unknown feature set {}".format(args.features))
    sys.exit(-1)

feature_type = args.features
output_path = args.output_dir
benign_path = args.benign_dir
malware_path = args.malware_dir


dictv, learning_features, labels = learning.expert.lib.utilsExpertFeatures.vectorize_data(
    benign_path=benign_path, malware_path=malware_path)
feature_names = dictv.get_feature_names()

#############################################################
malwindices = learning.expert.lib.utilsExpertFeatures.filter_malware_indices_features(X_data=learning_features, y_data=labels)


learning_features = learning_features[:, malwindices]
feature_names = np.array(feature_names)[malwindices]

#############################################################

pickle.dump(dictv, open(os.path.join(output_path, "dict-vectorizer-" + feature_type + ".pck"), "wb"))
pickle.dump(malwindices, open(os.path.join(output_path, "feature-indices-" + feature_type + ".pck"), "wb"))

print("Successfully extracted...\n")
print("{} Features are: {}".format(len(feature_names), feature_names))
