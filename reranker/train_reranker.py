import argparse
import random
from tqdm import tqdm
import os
import json

import torch
import torch.distributed as dist

from torch.optim import Adam
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers


from data_utils import (
    get_rerank_dataloader,
    get_rerank_examples_odqa,
    get_words_dict,
    get_rerank_examples_odqa_amr,
    cal_metrics,
    load_data)
from reranker import candidate_models, Reranker


def get_model_saving_path():
    _saved_model_name = 'reranker_'
    _saved_model_name += f'num_neg_psg_{args.num_negative_psg}_'
    _saved_model_name += f'bsz_{args.train_bsz * args.gradient_accumulation_step}_'
    _saved_model_name += f'model_{args.model}_'
    _saved_model_name += f'loss_{args.loss}_'
    # _saved_model_name += f'max_query_len_{args.max_query_length}_'
    # _saved_model_name += f'max_combined_len_{args.max_combined_length}_'
    _saved_model_name += f'{args.note}'
    if not os.path.exists('reranker_checkpoints/{}'.format(_saved_model_name)):
        os.mkdir('reranker_checkpoints/{}'.format(_saved_model_name))
    return os.path.join('reranker_checkpoints/{}'.format(_saved_model_name), _saved_model_name)


def train(rank, train_examples, eval_examples, model, args):
    # print('Setting up')
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    # dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    # print('Get model')
    transformers.logging.set_verbosity_error()

    # model = DDP(model, device_ids=[rank], output_device=rank)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0)

    print('get dataloader')
    train_dl = get_rerank_dataloader(
        train_examples,
        args,
        device,
        args.train_bsz,
        tokenizer=tokenizer,
        words_dict=words_dict,
        pid_to_psg=pid_to_psg,
        shuffle=True,
        is_train=True,
        is_amr=args.is_amr)
    step_per_epoch = len(train_dl)
    step = 0

    # print('Start iterating')
    # for epoch in range(args.epoch):
    #     model.train()
    #     bar = tqdm(train_dl)
    #     for i, batch in enumerate(bar):
    #         step += 1
    # return

    print('Start training')
    f_result = open(f'{args.model_saving_path}_results.txt', 'w', encoding='utf8')

    for epoch in range(args.epoch):
        model.train()
        bar = tqdm(train_dl)
        loss_ave = 0
        for i, batch in enumerate(bar):
            step += 1
            # inputs = batch['inputs'].to(rank)
            inputs = batch['inputs'].to(device)
            labels = batch['labels']
            loss = model(inputs, labels, True)
            loss.backward()
            loss_ave += loss.item()
            bar.set_description(f'loss: {round(loss.item(), 4)}')
            if step % args.gradient_accumulation_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                model.zero_grad()
            if step % args.save_step == 0 and args.save_step > 0:
                ckpt_name = f'{args.model_saving_path}_step_{step+1}'
                print(f'Save model {ckpt_name}.ckpt')
                torch.save(model.state_dict(), ckpt_name+'.ckpt')
                evaluate(device, args, eval_examples, model, ckpt_name)

        loss_ave = loss_ave/len(train_dl)
        print(f'Average loss in epoch {epoch} = {round(loss_ave, 4)}')
        f_result.write(f'Average loss in epoch {epoch} = {round(loss_ave, 4)}\n')
        ckpt_name = f'{args.model_saving_path}_epoch_{epoch}'
        print(f'Save model {ckpt_name}.ckpt')
        torch.save(model.state_dict(), ckpt_name+'.ckpt')
        evaluate(device, args, eval_examples, model, ckpt_name, f_result)
    # dist.destroy_process_group()

def evaluate(rank, args, examples, model, ckpt_name, f_result=None):
    print('Setting up')
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    # dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    dataloader = get_rerank_dataloader(
        examples,
        args,
        device,
        1,
        tokenizer=tokenizer,
        words_dict=words_dict,
        pid_to_psg=pid_to_psg,
        shuffle=False,
        is_train=False,
        is_amr=args.is_amr)
    transformers.logging.set_verbosity_error()
    if model is not None:
        model = model.to(device)
    else:
        model = Reranker(args=args).to(device)
        print(f'Load model {ckpt_name}.ckpt')
        model.load_state_dict(torch.load(ckpt_name + '.ckpt'), strict=False)  # , map_location=map_location))
    # model = DDP(model, device_ids=[rank], output_device=rank)
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}

    # print(f'Get model {ckpt_name}.ckpt')
    # model.load_state_dict(torch.load(ckpt_name+'.ckpt'), strict=False) # , map_location=map_location))
    model.eval()

    val_predictions = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            scores = model(batch['inputs'].to(device), False).cpu().data.numpy().tolist()
            score_pid_list = list(zip(scores, batch['retrieved_pids'][0]))
            score_pid_list = sorted(score_pid_list, key=lambda x: x[0], reverse=True)
            val_predictions.append({
                'positive_pids': batch['positive_pids'][0],
                'retrieved_pids': batch['retrieved_pids'][0],
                'sorted_score_pid_list': score_pid_list,
            })
            # continue
            # for i in range(args.eval_bsz):
            #     if i >= len(batch['retrieved_pids']):
            #         break
            #     scores_i = scores[i * 100: (i+1) * 100]
            #     score_pid_list = list(zip(scores_i, batch['retrieved_pids'][i]))
            #     score_pid_list = sorted(score_pid_list, key=lambda x: x[0], reverse=True)
            #     val_predictions.append({
            #         'positive_pids': batch['positive_pids'][i],
            #         'retrieved_pids': batch['retrieved_pids'][i],
            #         'sorted_score_pid_list': score_pid_list,
            #     })

    split = args.eval_data.split('/')[-1].split('.')[0]
    json.dump(val_predictions, open(f'{ckpt_name}_{split}.json', 'w', ), indent=2)

    # dist.destroy_process_group()
    mrrs = 0
    mhits = 0
    top5 = 0
    top10 = 0
    top20 = 0
    for pred in val_predictions:
        topk_pids = [x[1] for x in pred['sorted_score_pid_list']]
        ranks = [topk_pids.index(positive_id) for positive_id in pred['positive_pids']]
        mrr, mhit, top5_tmp, top10_tmp, top20_tmp = cal_metrics(ranks, args.mhits_bar)
        mrrs += mrr
        top5 += top5_tmp
        top10 += top10_tmp
        top20 += top20_tmp
        if len(ranks) > 0:
            mhits += (mhit / len(ranks))
    print(f'Validation:')
    print(f'Num Examples {len(val_predictions)}')
    print(f'MRR: {round(100 * mrrs / len(val_predictions), 1)}')
    print(f'MHits@{args.mhits_bar}: {round(100 * mhits / len(val_predictions), 1)}')
    print(f'TOP5: {round(100 * top5 / len(val_predictions), 1)}')
    print(f'TOP10: {round(100 * top10 / len(val_predictions), 1)}')
    print(f'TOP20: {round(100 * top20 / len(val_predictions), 1)}')
    if f_result is not None:
        f_result.write(f'MRR: {round(100 * mrrs / len(val_predictions), 1)}\n')
        f_result.write(f'MHits@{args.mhits_bar}: {round(100 * mhits / len(val_predictions), 1)}\n')
        f_result.write(f'TOP5: {round(100 * top5 / len(val_predictions), 1)}\n')
        f_result.write(f'TOP10: {round(100 * top10 / len(val_predictions), 1)}\n')
        f_result.write(f'TOP20: {round(100 * top20 / len(val_predictions), 1)}\n')

def inference(rank, args, examples, ckpt_name):
    print('Setting up')
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    # dist.init_process_group("nccl", rank=rank, world_size=args.world_size)

    dataloader = get_rerank_dataloader(
        examples,
        args,
        device,
        1,
        tokenizer=tokenizer,
        words_dict=words_dict,
        pid_to_psg=pid_to_psg,
        shuffle=False,
        is_train=False,
        is_inference=True,
        is_amr=args.is_amr)

    transformers.logging.set_verbosity_error()
    model = Reranker(args).to(device)
    # model = DDP(model, device_ids=[rank], output_device=rank)
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    print(f'Get model {ckpt_name}.ckpt')
    model.load_state_dict(torch.load(ckpt_name + '.ckpt'))  # , map_location=map_location))

    model.eval()
    predictions = []
    for batch in tqdm(dataloader):
        with torch.no_grad():
            scores = model(batch['inputs'].to(device), False).cpu().data.numpy().tolist()
            score_pid_list = list(zip(scores, batch['retrieved_pids'][0]))
            score_pid_list = sorted(score_pid_list, key=lambda x: x[0], reverse=True)
            predictions.append({
                'sorted_score_pid_list': score_pid_list,
            })
            # for i in range(args.eval_bsz):
            #     if i >= len(batch['retrieved_pids']):
            #         break
            #     scores_i = scores[i * 100: (i+1) * 100]
            #     score_pid_list = list(zip(scores_i, batch['retrieved_pids'][i]))
            #     score_pid_list = sorted(score_pid_list, key=lambda x: x[0], reverse=True)
            #     predictions.append({
            #         'sorted_score_pid_list': score_pid_list,
            #     })

    split = args.eval_data.split('/')[-1].split('.')[0]
    # json.dump(predictions, open(f'{args.model_saving_path}_result_epoch_{args.ckpt}_localrank_{rank}_{split}.json', 'w'))
    # print(f'Scores writing in {ckpt_name}_{split}_scores.json')
    # json.dump(predictions, open(f'{ckpt_name}_{split}_scores.json', 'w', ), indent=2)
    if args.write_new_file:
        # ckpt_name = ckpt_name.strip('reranker_checkpoints').strip('/').strip('reranker').strip('_')
        print(ckpt_name)
        ckpt_name = ckpt_name.split('/')[2].strip('reranker').strip('_')
        print(f'New {split} split writing in ../rerank/{ckpt_name}/{split}.json')
        if not os.path.exists('../rerank/{}/'.format(ckpt_name)):
            os.mkdir('../rerank/{}/'.format(ckpt_name))
        ori_data = load_data(args.eval_data)
        new_file = open('../rerank/{}/{}.json'.format(ckpt_name, split),'w', encoding='utf8')
        new_cases = []
        for i, case in tqdm(enumerate(ori_data), desc='Writing'):
            new_case = {}
            new_case['question'] = case['question']
            new_case['answers'] = case['answers']
            # ids = [ctx['id'] for ctx in case['ctxs']]
            # ids_set = set(ids)
            new_ctxs = []
            psg_dict = {}
            for ctx in case['ctxs']:
                psg_dict[ctx['id']] = ctx
            for score in predictions[i]['sorted_score_pid_list'][:args.inference_ctx_num]:
                id = score[1]
                psg = psg_dict[id]
                psg['score'] = score[0]
                new_ctxs.append(psg)
                # if id in ids_set:
                #     ids_set.remove(id)
            # for id in ids_set:
            #     idx = min(ids.index(id), len(predictions[i]['sorted_score_pid_list']))
            #     psg = psg_dict[id]
            #     psg['score'] = predictions[i]['sorted_score_pid_list'][idx][0]
            #     new_ctxs.insert(idx, psg)
            # if len(new_ctxs) != 100:
            #     continue
            new_case['ctxs'] = new_ctxs
            new_cases.append(new_case)
            # json.dump(new_case, new_file)
            # new_file.write('\n')
        json.dump(new_cases, new_file, indent=1)
    # dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--is_amr', action='store_true')
    parser.add_argument('--write_new_file', action='store_true')
    parser.add_argument('--token_type', action='store_true')
    parser.add_argument('--graph_embs_visible_for_decoder', action='store_true')
    parser.add_argument('--graph_token_attention', action='store_true')
    parser.add_argument('--split_node_edge_tokens', action='store_true')
    parser.add_argument('--all4train', action='store_true')
    parser.add_argument('--only_nodes', action='store_true', help='not use edges but only use nodes as tokens')
    parser.add_argument('--only_edges', action='store_true', help='not use nodes but only use edges as tokens')
    parser.add_argument('--num_negative_psg',
                        type=int,
                        default=7)
    parser.add_argument('--model',
                        type=str,
                        default='electra',
                        choices=['bert', 'roberta', 'electra', 'bart'])
    parser.add_argument('--psgs',
                        type=str,
                        default='')
    parser.add_argument('--loss',
                        type=str,
                        default='cross_entropy',
                        choices=['cross_entropy', 'double_softmax'])
    parser.add_argument('--train_data',
                        type=str,
                        default='')
    parser.add_argument('--eval_data',
                        type=str,
                        default='')
    '''
    variance = a.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    '''
    parser.add_argument('--nodes_edges_file',
                        type=str, default='none', help='path of nodes and edges data')
    parser.add_argument('--epoch',
                        type=int,
                        default=1)
    parser.add_argument('--save_step',
                        type=int,
                        default=-1)
    parser.add_argument('--gradient_accumulation_step',
                        type=int,
                        default=1)
    parser.add_argument('--train_bsz',
                        type=int,
                        default=1)
    parser.add_argument('--lr',
                        type=float,
                        default=1e-5)
    parser.add_argument('--max_query_length', type=int, default=50)
    parser.add_argument('--max_combined_length', type=int, default=200)
    parser.add_argument('--node_length', type=int, default=95,
                             help='maximum number of tokens in amr nodes')
    parser.add_argument('--edge_length', type=int, default=125,
                        help='maximum number of tokens in amr edges')
    parser.add_argument('--word_length', type=int, default=7,
                             help='maximum number of tokens in one tokenized amr node/relation')
    parser.add_argument('--mhits_bar', type=int, default=10)
    parser.add_argument('--inference_ctx_num', type=int, default=100)
    parser.add_argument('--note',
                        type=str,
                        default='')
    parser.add_argument('--ckpt', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print('note = {}'.format(args.note))

    n_gpus = torch.cuda.device_count()
    args.world_size = n_gpus

    passages = json.load(open(args.psgs, 'r', encoding='utf8'))
    pid_to_psg = dict()
    for p in passages:
        pid_to_psg[p['id']] = p

    _, tokenizer_cls, model_name = candidate_models[args.model]

    tokenizer = tokenizer_cls.from_pretrained(model_name)
    # tokenizer.add_tokens(['||', '//'])

    # if args.fq_marker:
    #     tokenizer.add_tokens(args.fq_marker)

    if args.is_amr:
        words_dict = get_words_dict(tokenizer, args)
        args.model = 'graph_' + args.model
    else:
        words_dict = None
    args.model_saving_path = get_model_saving_path()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    get_rerank_examples = get_rerank_examples_odqa_amr if args.is_amr else get_rerank_examples_odqa
    if args.train:
        model = Reranker(args)
        model = model.to(device)
        train_examples = get_rerank_examples(args, args.train_data, 'train')
        val_examples = None
        if args.eval_data != '':
            val_examples = get_rerank_examples(args, args.eval_data, 'eval')
        train(0, train_examples, val_examples, model, args)
        # mp.spawn(train,
        #          args=(train_examples, args),
        #          nprocs=n_gpus,
        #          join=True)
    if args.val:
        val_examples = get_rerank_examples(args, args.eval_data, 'eval')
        if args.ckpt >= 0:
            ckpt_name = f'{args.model_saving_path}_epoch_{args.ckpt}'
        elif args.save_step > 0:
            ckpt_name = f'{args.model_saving_path}_step_{args.save_step}'
        else:
            raise ValueError('no checkpoint is specified')
        evaluate(0, args, val_examples, None, ckpt_name)
        # mp.spawn(evaluate,
        #          args=(args, val_examples, 'dev'),
        #          nprocs=n_gpus,
        #          join=True)
        # val_predictions = []
        # for local_rank in range(n_gpus):
        #     fname = f'{args.model_saving_path}_result_epoch_{args.ckpt}_localrank_{local_rank}_dev.json'
        #     if os.path.exists(fname):
        #         val_predictions += json.load(open(fname))
        #         os.remove(fname)
        # split = args.data_path.split('/')[-1].split('.')[0]
        # fname = f'{args.model_saving_path}_result_epoch_{args.ckpt}_{split}.json'
        # val_predictions = json.load(open(fname))
        # json.dump(val_predictions, open(f'{args.model_saving_path}_result_epoch_{args.ckpt}_dev.json', 'w'), indent=4)
    if args.inference:
        test_examples = get_rerank_examples(args, args.eval_data, 'inference')
        if args.ckpt >= 0:
            ckpt_name = f'{args.model_saving_path}_epoch_{args.ckpt}'
        elif args.save_step > 0:
            ckpt_name = f'{args.model_saving_path}_step_{args.save_step}'
        else:
            raise ValueError('no checkpoint is specified')
        inference(0, args, test_examples, ckpt_name)
