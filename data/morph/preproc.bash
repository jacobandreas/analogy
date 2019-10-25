#!/bin/bash

CONLL_PATH=/home/gridsan/jda/data/conll2018
FULL_ANN_FILE="full_annotations.txt"
SURFACE_FILE="surface.txt"
METADATA_FILE="metadata.txt"

rm -f $FULL_ANN_FILE $SURFACE_FILE $METADATA_FILE

data_path="$CONLL_PATH/task1/all"
data_files=$(find $data_path | grep train-high)
for file in $data_files; do
    lang=$(basename $file | cut -d "-" -f 1)
    paste <(cat $file) <(yes $lang | head - -n $(wc -l $file | cut -d " " -f 1)) >> $FULL_ANN_FILE
done

cat $FULL_ANN_FILE | cut -f 2 > $SURFACE_FILE
cat $FULL_ANN_FILE | cut -f 1,3- > $METADATA_FILE

python encode.py --surface=$SURFACE_FILE --metadata=$METADATA_FILE
