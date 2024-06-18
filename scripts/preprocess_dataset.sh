#!/bin/bash

path="src/preprocessing"
num_samples=30000000
num_rows=180000000
splits="train test"

for split in $splits
do
    python $path/preprocess_dataset.py \
        num_samples=$num_samples \
        num_rows=$num_rows \
        split=$split
done
