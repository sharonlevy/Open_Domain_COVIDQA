#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python ../eval_retrieval.py \
    ../data/dense_test.txt \
    /path/to/encoded/corpus \
    /path/to/saved/model/checkpoint_best.pt \
    --batch-size 1000 --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  --topk 100 --dimension 768
