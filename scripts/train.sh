#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../train_retrieval.py \
    --do_train \
    --prefix strong_dpr_baseline_b150 \
    --predict_batch_size 2000 \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --train_batch_size 75 \
    --learning_rate 2e-5 \
    --fp16 \
    --train_file ../data/dense_train.txt \
    --predict_file ../data/dense_dev.txt \
    --seed 16 \
    --eval_period 300 \
    --max_c_len 300 \
    --max_q_len 30 \
    --warmup_ratio 0.1 \
    --num_train_epochs 20 \
    --dense_only \
    --output_dir /path/to/model/output \
