#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python ../encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --fp16 \
    --predict_file /path/to/corpus \
    --max_c_len 300 \
    --init_checkpoint /path/to/saved/model/checkpoint_best.pt \
    --save_path /path/to/encoded/corpus

