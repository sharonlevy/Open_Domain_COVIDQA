from torch.utils.data import Dataset
import json
import random
import attr
import os
import ipdb

def collate_tokens(values, pad_idx, eos_idx=None, left_pad=False, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    if len(values[0].size()) > 1:
        values = [v.view(-1) for v in values]
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res

def update_batch_dict(
    prefix, 
    instances,
    batch_dict, 
    pad_token_id
    ):
    batch_dict[f"{prefix}_input_ids"] = collate_tokens([s[prefix]["input_ids"].view(-1) for s in instances], pad_token_id)

    # use the padded tokens as inputs

    batch_dict[f"{prefix}_mask"] = collate_tokens([s[prefix]["attention_mask"].view(-1) for s in instances], 0)


    batch_dict[f"{prefix}_type_ids"] = collate_tokens([s[prefix]["token_type_ids"].view(-1) for s in instances], 0)


class RetrievalDataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 args,
                 train=False
                 ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_q_len = args.max_q_len
        self.max_c_len = args.max_c_len
        self.max_aug_len = args.max_aug_len
        self.aug_option = args.aug_option

        self.train = args.do_train
        self.test = "test" in data_path
        print(f"Loading data from {data_path}")
        self.data = [json.loads(line) for line in open(data_path).readlines()]

    def __getitem__(self, index):
        sample = self.data[index]
        question = sample['question']
        if question.endswith("?"):
            question = question[:-1]
        
        if self.test:
            return {
                "q": self.tokenizer(
                    question, max_length=self.max_q_len, return_tensors="pt", truncation=True),
                "question_raw": question
            }

        if isinstance(sample["pos_paras"], list):
            if self.train:
                pos_para = random.choice(sample["pos_paras"])
            else:
                pos_para = sample["pos_paras"][0]
            sample["pos_para"] = pos_para

        pos_title = sample['pos_para']['title'].strip()
        paragraph = sample['pos_para']['text'].strip()


        if self.train:
            random.shuffle(sample["neg_paras"])
        if len(sample["neg_paras"]) == 0:
            if self.train:
                neg_item = random.choice(self.data)

                if "pos_paras" in neg_item:
                    neg_item["pos_para"] = neg_item["pos_paras"][0]

                neg_title = neg_item["pos_para"]["title"].strip()
                neg_paragraph = neg_item["pos_para"]["text"].strip()
            else:
                neg_title = "dummy"
                neg_paragraph = "dummy"
        else:
            neg = random.choice(sample["neg_paras"])
            neg_title = neg['title'].strip()
            neg_paragraph = neg['text'].strip()
    

        neg_codes = self.tokenizer(
            neg_paragraph, max_length=self.max_c_len, return_tensors="pt", truncation=True)

        q_codes = self.tokenizer(
            question, max_length=self.max_q_len, return_tensors="pt", truncation=True)
        

        pos_codes = self.tokenizer(paragraph, max_length=self.max_c_len, return_tensors="pt", truncation=True)

        return {
            "q": q_codes,
            "c": pos_codes,
            "neg": neg_codes,
            "question_raw": question
        }

    def __len__(self):
        return len(self.data)

def retrieval_collate(samples, pad_id=0):

    batch = {}
    if len(samples) == 0:
        return batch
    for field in samples[0].keys():
        if "raw" not in field:
            update_batch_dict(field, samples, batch, pad_id)
        else:
            batch[field] = [s[field] for s in samples]

    return batch

import unicodedata
import csv

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)

@attr.s
class EncodingDataset(object):

    tokenizer = attr.ib()
    max_c_len = attr.ib()
    save_path = attr.ib()
    data_path = attr.ib()
    data = attr.ib(default=None)

    def load_data(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_path = os.path.join(self.save_path, "id2doc.json")  # ID to doc mapping
        reverse_path = os.path.join(self.save_path, "docIndex2id.json")

        print(f"Loading data from {self.data_path}")
        self.data = []

        if self.data_path.endswith("jsonl"):
            self.data = [json.loads(l) for l in open(self.data_path).readlines()]
        else:
            with open(self.data_path) as tsvfile:
                reader = csv.reader(tsvfile, delimiter='\t', )
                for row in reader:
                    if row[0] != 'id':
                        id_, text, title, index, date, journal, authors = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
                        self.data.append(
                            {"id": id_, "text": text, "title": title, "index": index, "date": date, "journal": journal, "authors": authors})

        print(f"Loaded {len(self.data)} documents...")
        id2doc = {}
        if 'authors' in self.data[0]:
            for idx, doc in enumerate(self.data):
                id2doc[idx] = (doc["title"], doc["text"], doc['id'], doc["index"], doc["date"], doc["journal"], doc["authors"])
        else:
            for idx, doc in enumerate(self.data):
                id2doc[idx] = (doc["title"], doc["text"], doc['id'])
        with open(save_path, "w") as g:
            json.dump(id2doc, g)
        print(f"Max sequence length: {self.max_c_len}")


    def __getitem__(self, index):
        sample = self.data[index]

        sent_codes = self.tokenizer(sample['text'].strip(), max_length=self.max_c_len, return_tensors="pt", truncation=True)

        return sent_codes

    def __len__(self):
        return len(self.data)


def encoding_collate(samples, pad_id=0):
    if len(samples) == 0:
        return {}

    batch = {
        'input_ids': collate_tokens([s['input_ids'].view(-1) for s in samples], pad_id),
        'input_mask': collate_tokens([s['attention_mask'].view(-1) for s in samples], 0),
    }

    if "token_type_ids" in samples[0]:
        batch["input_type_ids"] = collate_tokens(
            [s['token_type_ids'].view(-1) for s in samples], 0)

    return batch
