#!/bin/bash

CONLL_PATH=/home/gridsan/jda/data/conll2018
UNIMORPH_PATH=/home/gridsan/jda/data/unimorph

TMP="./tmp"

FULL_ANN_FILE="$TMP/full_annotations.txt"
SURFACE_FILE="$TMP/surface.txt"
METADATA_FILE="$TMP/metadata.txt"

FULL_REF_FILE="$TMP/full_references.txt"
REF_SURFACE_FILE="$TMP/ref_surface.txt"
REF_METADATA_FILE="$TMP/ref_metadata.txt"

# conll data

rm -rf "$TMP"
mkdir "$TMP"

data_path="$CONLL_PATH/task1/all"
data_files=$(find $data_path | grep train-high)
for file in $data_files; do
    lang=$(basename $file | rev | cut -d "-" -f 3- | rev)
    paste <(cat $file) <(yes $lang | head - -n $(wc -l $file | cut -d " " -f 1)) >> $FULL_ANN_FILE
done

cat "$FULL_ANN_FILE" | cut -f 2 > $SURFACE_FILE
cat "$FULL_ANN_FILE" | cut -f 1,3- > $METADATA_FILE

# unimorph data

rm -f "$FULL_REF_FILE" "$REF_SURFACE_FILE" "$REF_METADATA_FILE"

ref_files=$(find $UNIMORPH_PATH | egrep "unimorph/([a-z]{3})/(\\1)$")
for file in $ref_files; do
    lang=$(basename $file)
    paste <(cat $file | cut -f 2) <(yes $lang | head - -n $(wc -l $file | cut -d " " -f 1)) \
        | egrep -v "^\\s" \
        | sort \
	| uniq \
	>> $FULL_REF_FILE
done

cat "$FULL_REF_FILE" | cut -f 1 > "$REF_SURFACE_FILE"
cat "$FULL_REF_FILE" | cut -f 2 > "$REF_METADATA_FILE"

# encode

python encode.py \
    --surface=$SURFACE_FILE \
    --metadata=$METADATA_FILE \
    --ref_surface=$REF_SURFACE_FILE \
    --ref_metadata=$REF_METADATA_FILE \

# clean up

rm -r "$TMP"
