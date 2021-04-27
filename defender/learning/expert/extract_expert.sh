#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage extract.sh <INPUT-DIR> <OUTPUT-DIR> <FEATURE-METHOD>"
    exit 1
fi
PYPATH=$(pwd)

if [ ${PYPATH##*/} != "defender" ]; then
  echo "Go to defender directory and start scripts relative to this directory"
  exit 1
fi

NO_CORES=4 # number of cores

INPUT_DIR=$1
OUTPUT_DIR=$2
FEATURE=$3
# ! NO FILTER HERE, as malicious files have no file ending in our dataset

mkdir -p $OUTPUT_DIR
echo $INPUT_DIR $OUTPUT_DIR

ls $INPUT_DIR | parallel -k -j ${NO_CORES} PYTHONPATH=$PYPATH python learning/expert/expert.py --config learning/expert/config.yml $FEATURE  "${INPUT_DIR}/{}" "${OUTPUT_DIR}/{/}.txt"
