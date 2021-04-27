#!/bin/bash


if [ "$#" -ne 3 ]; then
    echo "Usage scan_dirs.sh <INPUT-FILE> <FILES-FILE> <LABEL-FILE>"
    echo "<INPUT-FILE> is assumed to contain a directory to scan and a label per line"
    echo "see the README.md for more info"
    exit 1
fi

INPUT_FILE="$1"
FILES_FILE="$2"
LABEL_FILE="$3"

while IFS=$'\t' read -r INPUT_DIR LABEL
do
    # echo "$INPUT_DIR >> $LABEL"
    echo "scanning $INPUT_DIR"
    # exclude evasion samples from previous MLSEC competition
    find "$INPUT_DIR" -type f ! -name '*_u*' ! -name '*_s*' -exec bash -c "echo {} >> $FILES_FILE; echo $LABEL >> $LABEL_FILE" \;
done < "$INPUT_FILE"