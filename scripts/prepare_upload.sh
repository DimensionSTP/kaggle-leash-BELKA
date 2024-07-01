#!/bin/bash

path="src/postprocessing"
is_tuned="untuned"
strategy="ddp"
upload_user="DeepChem"
model_type="ChemBERTa-77M-MLM"
precision=32
batch_size=1024
epoch=10

python $path/prepare_upload.py \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    precision=$precision \
    batch_size=$batch_size \
    epoch=$epoch
