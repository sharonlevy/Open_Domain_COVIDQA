#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../qa/run_qa.py \
  --model_name_or_path ktrapeznikov/biobert_v1.1_pubmed_squad_v2 \
  --train_file ../data/qa_train.json \
  --validation_file ../data/qa_dev.json \
  --test_file ../data/qa_test.json \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /path/to/model/output \

