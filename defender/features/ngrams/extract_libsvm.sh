#!/bin/bash

if [ "$#" -lt 5 ]; then
    echo "Usage extract.sh <FILES-FILE> <LABEL_FILE> <OUTPUT-FILE> <NO_CORES> [ngram|skipgram] <NLEN> [<SKIPLEN>]"
    echo "<INPUT-FILE> should be the output of scan_dirs.sh"
    exit 1
fi


FILES_FILE=$1
LABEL_FILE=$2
OUTPUT_FILE=$3
NO_CORES=$4 # number of cores
COMMAND=$5
NLEN=$6

if [ "$#" -eq 7 ]; then
    SKIPOPT="$7"
    shift
fi

shift 5

parallel -k -j ${NO_CORES}  --bar --link -a $FILES_FILE -a $LABEL_FILE PYTHONPATH=$(pwd) python ngrams.py --config config.yml --libsvm --label {2} "${COMMAND}" "${NLEN}" "${SKIPOPT}" {1} >> $OUTPUT_FILE
