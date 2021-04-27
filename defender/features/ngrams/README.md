# Byte NGram Features

## Extraction

1. run `scan_dirs.sh` on the directories containing the features
   - example: `./scan_dirs.sh scan.conf files.txt labels.txt`
   - this will scan the directories in `scan.conf` and write the complete file paths to files.txt and the corresponding label to labels.txt
   - `scan.cof` is expected to contain a directory path and the label (0 or 1), which is assigned to all samples in that directory, separated by a tab character 

2. run `extract_libsvm.sh` with the outputs of the previous step
   -   arg1: "files" file (i.e. `files.txt`)
   -   arg2: label file (i.e. `labels.txt`)
   -   arg3: output file
   -   arg4: number of CPU cores
   -   arg5: command ('ngram' or 'skipgram')
   -   arg6: ngram length
   -   arg7: length of skipping window (only used if command == 'skipgram')
   -   full example: `./extract_libsvm.sh files.txt labels.txt data.libsvm 8 skipgram 3 2`
  
  ### Result
   - `data.libsvm`: libsvm file with ngram counts
   - `files.txt`: file containing the original file names (each line corresponds to a line in the LIBSVM file)
   - `labels.txt`: file containing the label (as with the file names)

## Feature Selection

Select features using statistical tests. We only select features that are relevant for the positive class.

Example call: `python feature_selection.py data.libsvm spearman --threshold 0.1 --n_jobs 8 --json >> selected_dims.json`