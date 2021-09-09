# Open-Domain Question-Answering for COVID-19 and Other Emergent Domains

This repository contains the source code for an end-to-end open-domain question answering system. The system is made up of two components: a retriever model and a reading comprehension (question answering) model. We provide the code for these two models in addition to demo code based on Streamlit. A video of the demo can be viewed [here](https://www.youtube.com/watch?v=lk8LeIF4U7U).


## Installation
Our system uses PubMedBERT, a neural language model that is pretrained on PubMed abstracts for the retriever. Download the PyTorch version of PubMedBert [here](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract/tree/main). 
For reading comprehension, we utilize BioBERT fine-tuned on SQuAD V2 . The model can be found [here](https://huggingface.co/ktrapeznikov/biobert_v1.1_pubmed_squad_v2).


## Datasets
We provide the [COVID-QA](https://www.aclweb.org/anthology/2020.nlpcovid19-acl.18.pdf) dataset under the data directory. This is used for both the retriever and reading models. The train/dev/test files for the retriever are named dense_\*.txt and those for reading comprehension are named qa_\*.json.

The CORD-19 dataset is available for download [here](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html). Our system requires download of both the document_parses and metadata files for complete article information. For our system we use the 2021-02-15 download but any other download can also work. This must be combined into a jsonl file where each line contains a json object with:
  * id: article PMC id
  * title: article title
  * text: article text
  * index: text's index in the corpus (also the same as line number in the jsonl file)
  * date: article date 
  * journal: journal published
  * authors: author list

We split each article into multiple json entries based on paragraph text cutoff in the document_parses file. Paragraphs that are longer than 200 tokens are split futher. This can be done with ```splitCORD.py``` where
```
* metdata-file: the metadata downloaded for CORD
* pmc-path: path to the PMC articles downloaded for CORD
* out-path: output jsonl file
```

## Dense Retrieval Model
Once we have our model (PubMedBERT), we can start training. More specifically during training, we use positive and negative paragraphs, positive being paragraphs that contain the answer to a question, and negative ones not. We train on the COVID-QA dataset (see the Datasets section for more information on COVID-QA). We have a unified encoder for both questions and text paragraphs that learns to encode questions and associated texts into similar vectors. Afterwards, we use the model to encode the CORD-19 corpus.
### Training
```scripts/train.sh``` can be used to train our dense retrieval model.
```
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
```

Here are things to keep in mind:
```
1. The output_dir flag is where the model will be saved.
2. You can define the init_checkpoint flag to continue fine-tuning on another dataset.
```
The Dense retrieval model is then combined with BM25 for reranking (see paper for details).

### Corpus
Next, go to ```scripts/encode_covid_corpus.sh``` for the command to encode our corpus.
```
CUDA_VISIBLE_DEVICES=0 python ../encode_corpus.py \
    --do_predict \
    --predict_batch_size 1000 \
    --model_name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
    --fp16 \
    --predict_file /path/to/corpus \
    --max_c_len 300 \
    --init_checkpoint /path/to/saved/model/checkpoint_best.pt \
    --save_path /path/to/encoded/corpus
```

We pass the corpus (CORD-19) to our trained encoder in our dense retrieval model. Corpus embeddings are indexed. 
<br></br>

Here are things to keep in mind:
```
1. The predict_file flag should take in your CORD-19 dataset path. It should be a .jsonl file.
2. Look at your output_dir path when you ran train.sh. After training our model, we should now have a checkpoint in that folder. Copy the exact path onto
the init_checkpoint flag here.
3. As previously mentioned, the result of these commands is the corpus (CORD-19) embeddings become indexed. The embeddings are saved in the save_path flag argument. Create that directory path as you wish.
```

### Evaluation
You can run ```scripts/eval.sh``` to evaluate the document retrieval model.
```
CUDA_VISIBLE_DEVICES=0 python ../eval_retrieval.py \
    ../data/dense_test.txt \
    /path/to/encoded/corpus \
    /path/to/saved/model/checkpoint_best.pt \
    --batch-size 1000 --model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext  --topk 100 --dimension 768
```
We evaluate retrieval on a test set from COVID-QA. We determine the percentage of questions that have retrieved paragraphs with the correct answer across different top-k settings.

We do that in the following 3 ways:
1. exact answer matches in top-k retrievals
2. matching articles in top-k retrievals
3. F1 and Siamese BERT fuzzy matching
<br></br>

Here are things to think about:
```
1. The first, second, and third arguments are our COVID-QA test set, corpus indexed embeddings, and retrieval model respectively.
2. The other flag that is important is the topk one. This flag determines the quantity of retrieved CORD19 paragraphs.
```

## Reading Comprehension
We utilize the HuggingFace's question answering scripts to train and evaluate our reading comprehension model. This can be done with ```scripts/qa.sh```. The scripts are modified to allow for the extraction of multiple answer spans per document. We use a BioBERT model fine-tuned on SQuAD V2 as our pre-trained model.
```
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
```


## Demo
We combine the retrieval model and reading model for an end-to-end open-domain question answering demo with Streamlit. This can be run with ```scripts/demo.sh```.
```
CUDA_VISIBLE_DEVICES=0 streamlit run ../covid_qa_demo.py -- \
  --retriever-model-name microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --retriever-model path/to/saved/retriever_model/checkpoint_best.pt \
  --qa-model-name ktrapeznikov/biobert_v1.1_pubmed_squad_v2 \
  --qa-model /path/to/saved/qa_model \
  --index-path /path/to/encoded/corpus
```
Here are things to keep in mind:
```
1. retriever-model is the checkpoint file of your trained retriever model.
2. qa-model is the trained reading comprehension model.
3. index-path is the path to the encoded corpus embeddings.
```

## Requirements
See requirements.txt
