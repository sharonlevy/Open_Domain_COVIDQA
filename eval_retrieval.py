

import numpy as np
import json
import argparse
import logging
import torch
from tqdm import tqdm
import os

from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import defaultdict

from utils.torch_utils import load_saved, move_to_cuda
from utils.eval_utils import para_has_answer
from utils.basic_tokenizer import SimpleTokenizer

from transformers import AutoConfig, AutoTokenizer
from models.bert_retriever import BERTEncoder



from indexes import Extract_Index
from scipy.sparse import csr_matrix, load_npz

from coqa_process.span_heuristic import find_closest_span_match



from collections import Counter


from sentence_transformers import SentenceTransformer, util
import torch
from coqa_process.evaluate_qa import normalize_answer

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


PROCESS_TOK = None
def init():
    global PROCESS_TOK
    PROCESS_TOK = SimpleTokenizer()
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
modelTransf = SentenceTransformer('bert-base-nli-mean-tokens')  # NEW LINE

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def miniTest(question, answer, paragraph):
    strWord = paragraph
    sentences = strWord.split(".")
    stripSentences = [stri.strip() for stri in sentences]
    corpusEmbeddings = modelTransf.encode(stripSentences,show_progress_bar=False)
    queries = [answer]
    queryEmbeddings = modelTransf.encode(queries,show_progress_bar=False)
    topRank = min(2, len(stripSentences))
    cosScore = util.pytorch_cos_sim(queryEmbeddings, corpusEmbeddings)[0]
    bestResults = torch.topk(cosScore, k=topRank)
    for idx, score in zip(bestResults[1], bestResults[0]):
        num = f1_score(stripSentences[idx], queries[0])
        cosineScore = score
        if 0.291 <= num and 0.783 <= cosineScore:
            return True
        if 0.82 <= cosineScore:
            return True
    return False



def get_F1(answer_doc):
    answer = example["Answer"]

    gold_paras = []
    for p in passages:
        best_span, best_score = find_closest_span_match(
            p['contents'], answer)
        if best_score >= 0.8:
            gold_paras.append((p, best_score))

    example['answer_paras'] = gold_paras
    return example

def get_score(answer_doc, topk=20):
    """Search through all the top docs to see if they have the answer."""
    question, answer, docs, id = answer_doc
    global PROCESS_TOK
    topkpara_covered = []
    topkrecall = []
    topkF1 = []
    real_answer = []
    for p in docs:
        topkpara_covered.append(
            int(para_has_answer(answer, p["text"], PROCESS_TOK)))
        topkrecall.append(
            int(p["id"] == id))
        real_answer.append(para_has_answer(answer, p['text'], PROCESS_TOK) and p["id"] == id)
        topScore = False
        # for a in answer:
        #     best_span, best_score = find_closest_span_match(p["text"], a)
        #     if best_score >= 0.8:
        #         topScore = True
        #         break
        for a in answer:
            booVar = False
            if len(a.split()) <= 3:
                best_span, best_score = find_closest_span_match(p["text"], a)
                if best_score >= 0.8:
                    booVar = True
            else:
                booVar = miniTest(question, a, p["text"])
            if booVar:
                topScore = True
                break
        topkF1.append(int(topScore))
    return {
        "1": int(np.sum(topkpara_covered[:1]) > 0),
        "5": int(np.sum(topkpara_covered[:5]) > 0),
        "20": int(np.sum(topkpara_covered[:20]) > 0),
        "50": int(np.sum(topkpara_covered[:50]) > 0),
        "100": int(np.sum(topkpara_covered[:100]) > 0),
        #"500": int(np.sum(topkpara_covered[:500]) > 0),
        "recall@1": int(np.sum(topkrecall[:1]) > 0),
        "recall@5": int(np.sum(topkrecall[:5]) > 0),
        "recall@20": int(np.sum(topkrecall[:20]) > 0),
        "recall@50": int(np.sum(topkrecall[:50]) > 0),
        "recall@100": int(np.sum(topkrecall[:100]) > 0),
        #"recall@500": int(np.sum(topkrecall[:500]) > 0),
        "real_answer@1": int(np.sum(real_answer[:1]) > 0),
        "real_answer@5": int(np.sum(real_answer[:5]) > 0),
        "real_answer@20": int(np.sum(real_answer[:20]) > 0),
        "real_answer@50": int(np.sum(real_answer[:50]) > 0),
        "real_answer@100": int(np.sum(real_answer[:100]) > 0),
        "F1@1": int(np.sum(topkF1[:1]) > 0),
        "F1@5": int(np.sum(topkF1[:5]) > 0),
        "F1@20": int(np.sum(topkF1[:20]) > 0),
        "F1@50": int(np.sum(topkF1[:50]) > 0),
        "F1@100": int(np.sum(topkF1[:100]) > 0),
        #"F1@500": int(np.sum(topkF1[:500]) > 0)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data', type=str, default=None, help='query data')
    parser.add_argument('encode_corpus_path', type=str, default='./encoded/bart_aug')
    parser.add_argument('model_path', type=str, default=None, help="pretrained retriever checjpoint")
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--max-q-len', type=int, default=50)
    parser.add_argument('--model-name', type=str, default='facebook/bart-base')
    parser.add_argument("--save-pred", default="", type=str)
    parser.add_argument("--dimension", default=768, type=int)
    parser.add_argument('--index-type', type=str, default='exact')
    
    args = parser.parse_args()

    logger.info(f"Loading questions")
    qas = [json.loads(line) for line in open(args.raw_data).readlines()]
    questions = [_["question"][:-1]
                 if _["question"].endswith("?") else _["question"] for _ in qas]
    answers = [item.get("answer", item['answers']) for item in qas]
    ids = [_["id"] for _ in qas]

    logger.info("Loading trained model...")
    model_config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = BERTEncoder(args.model_name)
    model = load_saved(model, args.model_path)
    cuda = torch.device('cuda')
    model.to(cuda)
    modelTransf.to(cuda)

    from apex import amp
    model = amp.initialize(model, opt_level='O1')
    model.eval()

    logger.info(f"Loading index")
    if args.index_type == "exact":
        index_path = os.path.join(args.encode_corpus_path, "embeds.npy")
        embeds = np.load(index_path).astype('float32')
        index = Extract_Index(embeds, dimension=args.dimension)

    logger.info(f"Loading corpus")
    id2doc = json.load(open(os.path.join(args.encode_corpus_path, "id2doc.json")))
    logger.info(f"Corpus size {len(id2doc)}")

    retrieved_results = []
    for b_start in tqdm(range(0, len(questions), args.batch_size)):
        with torch.no_grad():
            batch_q = questions[b_start:b_start + args.batch_size]
            batch_ans = answers[b_start:b_start + args.batch_size]
            batch_q_encodes = tokenizer(
                batch_q, max_length=args.max_q_len, return_tensors="pt", truncation=True, padding=True)

            batch_q_encodes = move_to_cuda(dict(batch_q_encodes))
            q_embeds = model(
                batch_q_encodes["input_ids"], batch_q_encodes["attention_mask"], batch_q_encodes.get("token_type_ids", None))
            q_embeds_numpy = q_embeds.cpu().contiguous().numpy()
            distances, indices = index.search(args.topk, q_embeds_numpy)

            for b_idx in range(len(batch_q)):
                top_doc_ids = indices[b_idx]
                topk_docs = [{"title": id2doc[str(doc_id)][0], "text": id2doc[str(
                    doc_id)][1],"id": id2doc[str(doc_id)][2]} for doc_id in top_doc_ids]
                retrieved_results.append(topk_docs)

                new_rank = {}
                for place, doc_index in enumerate(topk_docs):
                    new_rank[place] = place




    answers_docs = list(zip(questions, answers, retrieved_results, ids))
    init()
    results = []
    for answer_doc in answers_docs:
        results.append(get_score(answer_doc, topk=args.topk))
    # results = get_score(qas_docs,topk=args.topk)

    error_cases = []
    for instance, res in zip(answers_docs, results):
        if res['20'] == 0:
            error_cases.append(instance)
    json.dump(error_cases, open(
        "nq-symmetric-dpr-dev-errors.json", 'w'), indent=1)

    aggregate = defaultdict(list)
    for r in results:
        for k, v in r.items():
            aggregate[k].append(v)

    for k in aggregate:
        results = aggregate[k]
        print('{}: {} ...'.format(
            k, np.mean(results)))
