#!/bin/bash

CUDA_VISIBLE_DEVICES=0 streamlit run ../covid_qa_demo.py -- \
  --retriever-model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --retriever-model path/to/saved/retriever_model/checkpoint_best.pt \
  --qa-model-name ktrapeznikov/biobert_v1.1_pubmed_squad_v2 \
  --qa-model /path/to/saved/qa_model \
  --index-path /path/to/encoded/corpus