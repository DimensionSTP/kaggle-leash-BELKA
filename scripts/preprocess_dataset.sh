#!/bin/bash

num_samples=2000000
num_rows=180000000
splits="train test"

for split in $splits
do
    python preprocess_dataset.py \
        num_samples=$num_samples \
        num_rows=$num_rows
done
