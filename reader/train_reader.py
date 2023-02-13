# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import time
import sys
import torch
import transformers
import wandb
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from src.options import Options

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model

def train(model, optimizer, scheduler, step, train_dataset, eval_dataset, opt, collator, best_dev_em, checkpoint_path):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir)/opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.global_rank + opt.seed) #different seed for different sampling depending on global_rank
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_train_batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=collator
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()

    while step < opt.total_steps:
        epoch += 1
        for i, batch in tqdm(enumerate(train_dataloader), desc="Training "):
            step += 1
            (idx, labels, _, context_ids, context_mask, graph_ids, graph_masks, rel_ids) = batch
            if not opt.cpu:
                context_ids = context_ids.cuda()
                context_mask = context_mask.cuda()
                graph_ids = graph_ids.cuda()
                graph_masks = graph_masks.cuda()
                labels = labels.cuda()
            train_loss = model(
                input_ids=context_ids,
                graph_ids=graph_ids,
                attention_mask=context_mask,
                graph_mask=graph_masks,
                labels=labels,
            )[0]

            train_loss.backward()
            if opt.local_rank in [-1, 0] and not opt.no_wandb:
                wandb.log({'train_loss': train_loss.item(), 'iteration': step})
                wandb.log({'lr': scheduler.get_last_lr()[0], 'iteration': step})
            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                optimizer.step()
                scheduler.step()

                model.zero_grad()

            train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()


            if step % opt.eval_freq == 0:
                dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                model.train()
                if opt.is_main:
                    if dev_em > best_dev_em:
                        best_dev_em = dev_em
                        src.util.save(model, optimizer, scheduler, step, best_dev_em,
                                  opt, checkpoint_path, 'best_dev')
                    log = f"{step} / {opt.total_steps} |"
                    log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                    log += f"evaluation: {100*dev_em:.2f}EM |"
                    log += f"lr: {scheduler.get_last_lr()[0]:.5f}"
                    logger.info(log)
                    if opt.local_rank in [-1, 0] and not opt.no_wandb:
                        wandb.log({'dev_em': 100 * dev_em, 'iteration': step})
                        wandb.log({'stable_train_loss': curr_loss / opt.eval_freq, 'iteration': step})
                    if tb_logger is not None:
                        tb_logger.add_scalar("Evaluation", dev_em, step)
                        tb_logger.add_scalar("Training", curr_loss / (opt.eval_freq), step)
                    curr_loss = 0.

            if opt.is_main and step % opt.save_freq == 0:
                src.util.save(model, optimizer, scheduler, step, best_dev_em,
                          opt, checkpoint_path, f"step-{step}")
            if step > opt.total_steps:
                break

def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_eval_batch_size,
        drop_last=False,
        num_workers=0,
        collate_fn=collator
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc="Evaluating "):
            (idx, labels, _, context_ids, context_mask, graph_ids, graph_masks, rel_ids) = batch

            if not opt.cpu:
                context_ids = context_ids.cuda()
                context_mask = context_mask.cuda()
                graph_ids = graph_ids.cuda()
                graph_masks = graph_masks.cuda()

            outputs = model.generate(
                input_ids=context_ids,
                graph_ids=graph_ids,
                attention_mask=context_mask,
                graph_mask=graph_masks,
                max_length=30
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                #print('{}\t{}\t{}'.format(k, ans, gold))
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch


def get_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if 'dim_reduction' in name:
            params.append(param)
    return params

if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()
    #opt = options.get_options(use_reader=True, use_optim=True)
    if opt.local_rank in [-1, 0] and not opt.no_wandb:
        wandb.login(key="64d4aba41acda3b77b50e25d595fb18fdf327590")
        wandb.init(project="FiD_struc", entity="seanwang", notes=opt.name)

    print('name = {}'.format(opt.name))
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir)/opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    #if not checkpoint_exists and opt.is_main:
    #    options.print_options(opt)
    #checkpoint_path, checkpoint_exists = util.get_checkpoint_path(opt)

    logger = src.util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )

    model_name = 't5-' + opt.model_size
    model_class = src.model_1.FiDT5

    #load data
    tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)

    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    train_dataset = src.data.AMRDataset(train_examples, opt=opt, tokenizer=tokenizer)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.AMRDataset(eval_examples, opt=opt, tokenizer=tokenizer)

    if opt.model_size is not None and opt.model_path == "none":
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model_1.FiDT5(t5.config, opt)
        model.load_t5(t5.state_dict())
        model = model.to(opt.local_rank if not opt.cpu else 'cpu')
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        load_path = checkpoint_path / 'checkpoint' / 'latest'
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, load_path, opt, reset_params=False)
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = \
            src.util.load(model_class, opt.model_path, opt, reset_params=True)
        opt1 = copy.deepcopy(opt)
        logger.info(f"Model loaded from {opt.model_path}")
    model.set_checkpoint(opt.use_checkpoint)
    words_dict = src.data.get_words_dict(tokenizer, train_examples + eval_examples, opt)
    collator = src.data.AMRCollator(opt, tokenizer, words_dict=words_dict)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path
    )