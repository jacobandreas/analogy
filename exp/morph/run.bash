#!/bin/bash

. $HOME/.profile

python3 -u ../../main.py \
  --task="morph" \
  --model="analogy" \
  --n_hidden=1024 \
  --morph_data="/home/gridsan/jda/code/analogy/data/morph" \
  #&> run.log
