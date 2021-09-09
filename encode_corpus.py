import logging
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from datetime import date
from torch.utils.data import DataLoader
import json

from transformers import AutoConfig, AutoTokenizer
from models.bert_retriever import BERTEncoder
from data_classes.dr_datasets import EncodingDataset, encoding_collate
from utils.torch_utils import move_to_cuda, AverageMeter, load_saved
from config import encode_args
from functools import partial
import apex

def main():
    args = encode_args()
    
    if args.sparse:
        from scipy.sparse import csr_matrix, vstack, save_npz

    if args.fp16:
        apex.amp.register_half_function(torch, 'einsum')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.StreamHandler()])
    logger = logging.getLogger(__name__)
    logger.info(args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r",
                device, n_gpu, bool(args.local_rank != -1))


    model_config = AutoConfig.from_pretrained(args.model_name)
    model = BERTEncoder(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(encoding_collate, pad_id=tokenizer.pad_token_id)

    if args.do_train and args.max_c_len > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, model_config.max_position_embeddings))

    eval_dataset = EncodingDataset(
        tokenizer=tokenizer, 
        max_c_len=args.max_c_len, 
        save_path=args.save_path,
        data_path=args.predict_file
        )
    eval_dataset.load_data()
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        logger.info("Loading best checkpoint")
        model = load_saved(model, args.init_checkpoint)

    model.to(device)

    if args.fp16:
        model = apex.amp.initialize(model, opt_level=args.fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    model.eval()
    embed_array = []
    for batch in tqdm(eval_dataloader):
        batch = move_to_cuda(batch)
        with torch.no_grad():
            embeds = model(batch["input_ids"], batch["input_mask"], batch.get("token_type_ids", None)).cpu().numpy()
            if args.sparse:
                embeds = csr_matrix(embeds)
            embed_array.append(embeds)

    embed_save_path = os.path.join(args.save_path, "embeds")
    if args.sparse:
        embed_array = vstack(embed_array)
        save_npz(embed_save_path + ".npz", embed_array)
    else:
        embed_array = np.concatenate(embed_array, axis=0)
        np.save(embed_save_path, embed_array.astype("float16"))
    logger.info(f'corpus embedding size {embed_array.shape}')
    

if __name__ == "__main__":
    main()
