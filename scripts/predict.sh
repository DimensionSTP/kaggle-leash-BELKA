#!/bin/bash

is_tuned="untuned"
strategy="ddp"
upload_user="DeepChem"
model_type="ChemBERTa-10M-MTR"
precision=32
batch_size=1024
epochs="11 12"

for epoch in $epochs
do
    python main.py mode=predict \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        precision=$precision \
        batch_size=$batch_size \
        epoch=$epoch
done
