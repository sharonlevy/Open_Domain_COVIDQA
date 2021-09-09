from transformers import AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import ipdb

import math



class BERTRetriever(nn.Module):

    def __init__(self,
                 model_name,
                 ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)


    def encode_seq(self, input_ids, mask, type_ids):

        all_token_rep = self.encoder(input_ids, mask, type_ids)[0]

        vector = all_token_rep[:,0,:]

        return vector, None



    def forward(self, batch):
        c_cls, c_orig = self.encode_seq(
            batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'])
        neg_c_cls, neg_c_orig = self.encode_seq(
            batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'])
        q_cls, q_orig = self.encode_seq(
                batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'])

        

        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}


class BERTEncoder(BERTRetriever):

    def forward(self, input_ids, input_mask, type_ids):
        return self.encode_seq(input_ids, input_mask, type_ids)[0]
