import logging
import os
import random
from tqdm import tqdm
import numpy as np
import torch
from datetime import date
from torch.utils.data import DataLoader
import json

from transformers import AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup
import transformers

from models.bert_retriever import BERTRetriever

from torch.utils.tensorboard import SummaryWriter
from data_classes.dr_datasets import RetrievalDataset, retrieval_collate
from utils.torch_utils import move_to_cuda, AverageMeter, load_saved
from config import train_args
from criterions import  loss_dense
from torch.optim import Adam
from functools import partial
from utils.eval_utils import f1_score, metric_max_over_ground_truths, exact_match_score
import apex

def main():
    args = train_args()

    if args.fp16:
        apex.amp.register_half_function(torch, 'einsum')

    date_curr = date.today().strftime("%m-%d-%Y")
    model_name = f"{args.prefix}-seed{args.seed}-bsz{args.train_batch_size}-fp16{args.fp16}-lr{args.learning_rate}-decay{args.weight_decay}-warm{args.warmup_ratio}-{args.model_name}"
    args.output_dir = os.path.join(args.output_dir, date_curr, model_name)
    tb_logger = SummaryWriter(os.path.join(
        args.output_dir.replace("logs", "tflogs")))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        print(
            f"output directory {args.output_dir} already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.output_dir, "log.txt")),
                                  logging.StreamHandler()])
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

    args.train_batch_size = int(
        args.train_batch_size / args.accumulate_gradients)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    model_config = AutoConfig.from_pretrained(args.model_name)

    model = BERTRetriever(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    collate_fc = partial(retrieval_collate, pad_id=tokenizer.pad_token_id)

    if args.do_train and args.max_c_len > model_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (args.max_c_len, model_config.max_position_embeddings))

    eval_dataset = RetrievalDataset(
            tokenizer, args.predict_file, args)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.predict_batch_size, collate_fn=collate_fc, pin_memory=True, num_workers=args.num_workers)
    logger.info(f"Num of dev batches: {len(eval_dataloader)}")

    if args.init_checkpoint != "":
        logger.info("Loading best checkpoint")
        model = load_saved(model, args.init_checkpoint)

    model.to(device)
    logger.info(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.do_train:
        sparse_params = [p for n, p in model.named_parameters()
                         if "qz_loga" in n]
        optimizer_parameters = [
            {'params': sparse_params, 'lr': args.learning_rate},
            {'params': [p for n, p in model.named_parameters() if 'qz_loga' not in n]},
        ]
        optimizer = Adam(optimizer_parameters,
                         lr=args.learning_rate, eps=args.adam_epsilon)

        if args.fp16:
            model, optimizer = apex.amp.initialize(
                model, optimizer, opt_level=args.fp16_opt_level)
    else:
        if args.fp16:
            model = apex.amp.initialize(model, opt_level=args.fp16_opt_level)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.do_train:
        global_step = 0  # gradient update step
        batch_step = 0  # forward batch count
        best_mrr = 0
        train_loss_meter = AverageMeter()
        model.train()

        train_dataset = RetrievalDataset(
                tokenizer, args.train_file, args)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, pin_memory=True, collate_fn=collate_fc, num_workers=args.num_workers, shuffle=True)

        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info('Start training....')
        for epoch in range(int(args.num_train_epochs)):

            for batch in tqdm(train_dataloader):

                batch_step += 1
                batch = move_to_cuda(batch)


                losses = loss_dense(model, batch)
                loss = losses["retr_loss"]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                train_loss_meter.update(loss.item())

                if (batch_step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(
                            apex.amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    #scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    for k, v in losses.items():
                        tb_logger.add_scalar(f'batch {k}', v.item(), global_step)

                    tb_logger.add_scalar('batch_train_loss',
                                         loss.item(), global_step)
                    tb_logger.add_scalar('smoothed_train_loss',
                                         train_loss_meter.avg, global_step)

                    if args.eval_period != -1 and global_step % args.eval_period == 0:
                        mrr = predict(args, model, eval_dataloader,
                                      device, logger, tokenizer)
                        logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                            global_step, train_loss_meter.avg, mrr*100, epoch))

                        if best_mrr < mrr:
                            logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                                        (best_mrr*100, mrr*100, epoch))
                            torch.save(model.state_dict(), os.path.join(
                                args.output_dir, f"checkpoint_best.pt"))
                            model = model.to(device)
                            best_mrr = mrr

            mrr = predict(args, model, eval_dataloader, device, logger, tokenizer)
            logger.info("Step %d Train loss %.2f MRR %.2f on epoch=%d" % (
                global_step, train_loss_meter.avg, mrr*100, epoch))
            tb_logger.add_scalar('dev_mrr', mrr*100, epoch)
            if best_mrr < mrr:
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_last.pt"))
                logger.info("Saving model with best MRR %.2f -> MRR %.2f on epoch=%d" %
                            (best_mrr*100, mrr*100, epoch))
                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, f"checkpoint_best.pt"))
                model = model.to(device)
                best_mrr = mrr

        logger.info("Training finished!")

    elif args.do_predict:
        mrr = predict(args, model, eval_dataloader, device, logger, tokenizer)
        logger.info(f"test performance {mrr}")


def predict(args, model, eval_dataloader, device, logger, tokenizer):

    model.eval()
    num_correct, num_total, rrs = 0, 0, []  # reciprocal rank
    f1s, ems = [], [] # augmentation accuracy
    sparse_ratio_q = []
    sparse_ratio_c = []

    if args.quantization:
        num_correct_quant, num_total_quant, rrs_quant = 0, 0, []

    def cal_metric(q, c, neg_c):
        product_in_batch = torch.mm(q, c.t())
        product_neg = (q * neg_c).sum(-1).unsqueeze(1)
        product = torch.cat([product_in_batch, product_neg], dim=-1)

        target = torch.arange(product.size(0)).to(product.device)
        ranked = product.argsort(dim=1, descending=True)
        prediction = product.argmax(-1)

        # MRR
        batch_rrs = []
        idx2rank = ranked.argsort(dim=1)
        for idx, t in enumerate(target.tolist()):
            batch_rrs.append(1 / (idx2rank[idx][t].item() + 1))
        pred_res = prediction == target

        batch_total = pred_res.size(0)
        batch_correct = pred_res.sum(0)

        return {
            'batch_rrs': batch_rrs,
            'batch_total': batch_total,
            'batch_correct': batch_correct
        }



    for batch in tqdm(eval_dataloader):
        batch = move_to_cuda(batch)
        with torch.no_grad():
            outputs = model(batch)
            q, c, neg_c = outputs["q"], outputs["c"], outputs["neg_c"]

            # calculate the sparsity
            sparse_ratio_q += (torch.count_nonzero(q, dim=1) / q.size(1)).tolist()
            sparse_ratio_c += (torch.count_nonzero(c, dim=1) / c.size(1)).tolist()

            batch_metrics = cal_metric(q, c, neg_c)
            rrs += batch_metrics['batch_rrs']
            num_correct += batch_metrics['batch_correct']
            num_total += batch_metrics['batch_total']


    acc = num_correct / num_total
    mrr = np.mean(rrs)
    logger.info(f"evaluated {num_total} examples...")
    logger.info(f"avg. Acc: {acc:.3f}")
    logger.info(f'avg. MRR: {mrr:.3f}')
    
    logger.info(f'avg sparsity question: {np.mean(sparse_ratio_q)}, {len(sparse_ratio_q)}')
    logger.info(f'avg sparsity context: {np.mean(sparse_ratio_c)}, {len(sparse_ratio_c)}')

    model.train()
    return mrr


if __name__ == "__main__":
    main()
