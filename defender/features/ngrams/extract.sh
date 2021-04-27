#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage extract.sh <INPUT-DIR> <OUTPUT-FILE>"
    exit 1
fi

NO_CORES=4 # number of cores

INPUT_DIR=$1
OUTPUT_FILE=$2

echo $INPUT_DIR $OUTPUT_FILE

ls $INPUT_DIR | grep ".exe" | parallel -k -j ${NO_CORES} PYTHONPATH=$(pwd) python ngrams.py --config config.yml --nocutting ngram 3  "${INPUT_DIR}/{}" >> $OUTPUT_FILE
