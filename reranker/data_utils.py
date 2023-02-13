import json
import random
import regex, string, re
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, DistributedSampler
from torch.nn import ZeroPad2d

from check_answers import has_answer1, has_answer2, SimpleTokenizer

check_answers = has_answer2

def load_data(data_path=None):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in tqdm(enumerate(data), desc='Loading: '):
        if data_path is not None and data_path.endswith('.jsonl'):
            example = json.loads(example)
        if not 'id' in example:
            example['id'] = k
        for c in example['ctxs']:
            if not 'score' in c:
                c['score'] = 1.0 / (k + 1)
        examples.append(example)
    ## egrave: is this needed?
    if data_path is not None and data_path.endswith('.jsonl'):
        data.close()

    return examples

def cal_metrics(ranks, mhits_bar=10):
    mrr = 0
    mhit = 0
    top5_tmp = 0
    top10_tmp = 0.
    top20_tmp = 0
    for rank in ranks:
        mrr += 1 / (rank + 1) / len(ranks)
        if rank < mhits_bar:
            mhit += 1
        if rank < 5:
            top5_tmp = 1
        if rank < 10:
            top10_tmp = 1
        if rank < 20:
            top20_tmp = 1
    return mrr, mhit, top5_tmp, top10_tmp, top20_tmp

def write_pos_ids():
    data_path = "../{}/{}.json"
    datasets = ['TQ', 'NQ']
    splits = ['dev', 'test']

    for dataset in datasets:
        for split in splits:
            examples = []
            data = load_data(data_path.format(dataset, split))
            tmp_tknzir = SimpleTokenizer()
            for d in tqdm(data, desc='Processing: '):
                ex = dict()
                ex['question'] = d['question']
                ex['answers'] = d['answers']
                # retrieved_ids = [ctx['id'] for ctx in d['ctxs']]
                golden_ids = [ctx['id'] for ctx in d['ctxs'] if
                              check_answers(d['answers'], ctx['title'] + '.' + ctx['text'], tmp_tknzir)]
                ex['positive_ids'] = golden_ids
                ex['positive_ids'] = golden_ids
                examples.append(ex)
            fout = open('positives/{}_{}_positives.json'.format(dataset, split), 'w', encoding='utf8')
            json.dump(examples, fout, indent=1)

def get_rerank_examples_odqa(args, data_path, status):
    data = load_data(data_path)
    examples = []
    mrrs = 0.
    mhits = 0.
    top5 = 0.
    top10 = 0.
    top20 = 0.
    tmp_tknzir = SimpleTokenizer()
    for d in tqdm(data, desc='Processing: '):
        ex = dict()
        ex['question'] = d['question']
        ex['answers'] = d['answers']
        retrieved_ids = [ctx['id'] for ctx in d['ctxs']]

        if status == 'eval':
            golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + '.' + ctx['text'], tmp_tknzir)]
            # if len(golden_ids) == 0 or len(golden_ids) >= 50:
            #     continue
            ex['positive_ids'] = golden_ids
            ranks = [retrieved_ids.index(positive_id) for positive_id in golden_ids]
            mrr, mhit, top5_tmp, top10_tmp, top20_tmp = cal_metrics(ranks, args.mhits_bar)
            top5 += top5_tmp
            top10 += top10_tmp
            top20 += top20_tmp
            mrrs += mrr
            if len(ranks) > 0:
                mhits += (mhit / len(ranks))

        # get negative passages
        if status == 'train':
            golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + '.' + ctx['text'], tmp_tknzir)]
            if len(golden_ids) == 0:
                continue
            ex['positive_ids'] = golden_ids
            non_golden_ids = [ctx['id'] for ctx in d['ctxs'] if ctx['id'] not in golden_ids]
            if len(non_golden_ids) < args.num_negative_psg:
                    continue
            ex['negative_ids'] = non_golden_ids
        else:
            ex['retrieved_ids'] = retrieved_ids
        examples.append(ex)

    if status == 'eval':
        print('DPR results:')
        print(f'Num Examples {len(examples)}')
        print(f'MRR: {round(100 * mrrs / len(examples), 1)}')
        print(f'MHits@{args.mhits_bar}: {round(100 * mhits / len(examples), 1)}')
        print(f'TOP5: {round(100 * top5 / len(examples), 1)}')
        print(f'TOP10: {round(100 * top10 / len(examples), 1)}')
        print(f'TOP20: {round(100 * top20 / len(examples), 1)}')
    return examples

def get_rerank_examples_odqa_amr(args, data_path, status):
    data = load_data(data_path)
    examples = []
    mrrs = 0.
    mhits = 0.
    top5 = 0.
    top10 = 0.
    top20 = 0.
    tmp_tknzir = SimpleTokenizer()
    node_lens = []
    edge_lens = []
    for it, d in tqdm(enumerate(data), desc='Processing: '):
        ex = dict()
        ex['question'] = d['question']
        ex['answers'] = d['answers']
        retrieved_ids = [ctx['id'] for ctx in d['ctxs']]

        if status == 'train' or status == 'eval':
            golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + ctx['text'], tmp_tknzir)]
            if status == 'train' and (len(golden_ids) == 0 or len(golden_ids)>50):
                continue
            non_golden_ids = [ctx['id'] for ctx in d['ctxs'] if ctx['id'] not in golden_ids]
            if status == 'train' and len(non_golden_ids) <= args.num_negative_psg:
                continue
            if status == 'eval':
                ranks = [retrieved_ids.index(positive_id) for positive_id in golden_ids]
                mrr, mhit, top5_tmp, top10_tmp, top20_tmp = cal_metrics(ranks, args.mhits_bar)
                mrrs += mrr
                if len(ranks) > 0:
                    mhits += (mhit / len(ranks))
                top5 += top5_tmp
                top10 += top10_tmp
                top20 += top20_tmp

        # nodess = [ctx['nodes'].strip('|$').strip().split('|$') for ctx in d['ctxs']]
        # edgess = [ctx['edges'].strip('|$').strip().split('|$') for ctx in d['ctxs']]
        id2node_edge = {}
        for i, ctx in enumerate(d['ctxs']):
            # node_lens.append(len(nodess[i]))
            # edge_lens.append(len(edgess[i]))
            nodes = ctx['nodes'][:args.node_length]
            edges = ctx['edges'][:args.edge_length]
            nodes_tokens = nodes[:]
            nodes_tokens_len = len(nodes_tokens)
            edges_tokens = [[nodes_tokens[int(edge[0])], edge[1], nodes_tokens[int(edge[2])]] for edge in edges if int(edge[0]) < nodes_tokens_len and int(edge[2]) < nodes_tokens_len]
            # nodes = [node.split('\t') for node in nodess_]
            # edges = [edge.split('\t') for edge in edgess_]
            # nodes_id2token = {node[0].strip(): re.sub(re.compile(r"-([0-9]([0-9]))"), "", node[1].strip()).strip() for
            #                   i, node in enumerate(nodes)}
            # nodes_tokens = [re.sub(re.compile(r"-([0-9]([0-9]))"), "", node[1].strip()).strip() for node in nodes]
            # edges_ = [edge for edge in edges if edge[0] in nodes_tokens and edge[2] in nodes_tokens]
            # edges_tokens = [node if i==1 else nodes_tokens[node] for edge in edges_ for i, node in enumerate(edge)]
            # edges_tokens = [[edges_tokens[j], edges_tokens[j + 1], edges_tokens[j + 2]] for j in
            #                 range(0, len(edges_tokens), 3)]
            id2node_edge[ctx['id']] = [nodes_tokens, edges_tokens]

        if status == 'train' or status == 'eval':
            ex['positive_ids'] = golden_ids
            ex['negative_ids'] = non_golden_ids
            ex['pos_nodes_edges'] = [id2node_edge[id] for id in golden_ids]
            ex['neg_nodes_edges'] = [id2node_edge[id] for id in non_golden_ids]
        if status != 'train':
            ex['retrieved_ids'] = retrieved_ids
        ex['ret_nodes_edges'] = [id2node_edge[id] for id in retrieved_ids]
        # # get negative passages
        # if status == 'train':
        #     golden_ids = [ctx['id'] for ctx in d['ctxs'] if check_answers(d['answers'], ctx['title'] + ctx['text'])]
        #     if len(golden_ids) == 0:
        #         continue
        #     positive_ids = random.sample(golden_ids, k=1)[0]
        #     ex['positive_ids'] = positive_ids
        #     non_golden_ids = [ctx['id'] for ctx in d['ctxs'] if ctx['id'] not in golden_ids]
        #     if len(non_golden_ids) < args.num_negative_psg:
        #         continue
        #     negative_ids = random.sample(non_golden_ids[:50], k=args.num_negative_psg)
        #     ex['negative_ids'] = negative_ids
        examples.append(ex)
    if status == 'eval':
        print('DPR results:')
        print(f'Num Examples {len(examples)}')
        print(f'MRR: {round(100 * mrrs / len(examples), 1)}')
        print(f'MHits@{args.mhits_bar}: {round(100 * mhits / len(examples), 1)}')
        print(f'TOP5: {round(100 * top5 / len(examples), 1)}')
        print(f'TOP10: {round(100 * top10 / len(examples), 1)}')
        print(f'TOP20: {round(100 * top20 / len(examples), 1)}')
    return examples

def encode_nodes_edges(nodes, edges, graph_max_length, words_dict=None):
    node_ids, node_masks = [], []
    for j in range(len(nodes)):
        nodes_ = nodes[j][:graph_max_length-1]
        edges_ = edges[j][:max(graph_max_length-len(nodes_),1)]
        if len(nodes_) == 0:
            nodes_ = ["", ]
        tmp_node_ids = torch.cat([words_dict[node]['input_ids'] for node in nodes_]).unsqueeze(0)
        tmp_node_ids = tmp_node_ids.unsqueeze(-2)
        tmp_node_ids = torch.cat([tmp_node_ids, tmp_node_ids, tmp_node_ids], dim=-2)

        tmp_node0_ids = torch.cat([words_dict[edge[0]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_rel_ids = torch.cat([words_dict[edge[1]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_node1_ids = torch.cat([words_dict[edge[2]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_edge_ids = torch.cat(
            [tmp_node0_ids.unsqueeze(-2), tmp_rel_ids.unsqueeze(-2), tmp_node1_ids.unsqueeze(-2)], dim=-2)
        tmp_node_ids = torch.cat([tmp_node_ids, tmp_edge_ids], dim=1)
        padding_num = graph_max_length - tmp_node_ids.shape[1]
        zero_pad = ZeroPad2d(padding=(0, 0, 0, 0, 0, padding_num))
        tmp_node_ids = zero_pad(tmp_node_ids)

        tmp_node_attention_mask = torch.LongTensor([2 for t in range(len(nodes_))]).unsqueeze(0)
        tmp_edge_attention_mask = torch.LongTensor([3 for t in range(len(edges_))]).unsqueeze(0)
        tmp_pad_attention_mask = torch.LongTensor([0 for t in range(padding_num)]).unsqueeze(0)
        # tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_pad_attention_mask], dim=-1)
        tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_edge_attention_mask, tmp_pad_attention_mask],dim=-1)
        node_masks.append(tmp_attention_mask)
        node_ids.append(tmp_node_ids)
    node_ids = torch.cat(node_ids, dim=0)
    node_masks = torch.cat(node_masks, dim=0)
    return node_ids, node_masks

def encode_nodes(nodes, node_max_length, words_dict=None):
    node_ids, node_masks = [], []
    for j in range(len(nodes)):
        nodes_ = nodes[j][:node_max_length]
        if len(nodes_) == 0:
            nodes_ = ["", ]
        tmp_node_ids = torch.cat([words_dict[node]['input_ids'] for node in nodes_]).unsqueeze(0)
        tmp_node_ids = tmp_node_ids.unsqueeze(-2)
        tmp_node_ids = torch.cat([tmp_node_ids, tmp_node_ids, tmp_node_ids], dim=-2)

        padding_num = node_max_length - tmp_node_ids.shape[1]
        zero_pad = ZeroPad2d(padding=(0, 0, 0, 0, 0, padding_num))
        tmp_node_ids = zero_pad(tmp_node_ids)

        tmp_node_attention_mask = torch.LongTensor([2 for t in range(len(nodes_))]).unsqueeze(0)
        tmp_pad_attention_mask = torch.LongTensor([0 for t in range(padding_num)]).unsqueeze(0)
        tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_pad_attention_mask],dim=-1)
        node_masks.append(tmp_attention_mask)
        node_ids.append(tmp_node_ids)
    node_ids = torch.cat(node_ids, dim=0)
    node_masks = torch.cat(node_masks, dim=0)
    return node_ids, node_masks

def encode_edges(edges, edge_max_length, words_dict=None):
    edge_ids, edge_masks = [], []
    for j in range(len(edges)):
        edges_ = edges[j][:max(edge_max_length,1)]

        tmp_node0_ids = torch.cat([words_dict[edge[0]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_rel_ids = torch.cat([words_dict[edge[1]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_node1_ids = torch.cat([words_dict[edge[2]]['input_ids'] for edge in edges_]).unsqueeze(0)
        tmp_edge_ids = torch.cat(
            [tmp_node0_ids.unsqueeze(-2), tmp_rel_ids.unsqueeze(-2), tmp_node1_ids.unsqueeze(-2)], dim=-2)
        padding_num = edge_max_length - tmp_edge_ids.shape[1]
        zero_pad = ZeroPad2d(padding=(0, 0, 0, 0, 0, padding_num))
        tmp_edge_ids = zero_pad(tmp_edge_ids)

        tmp_edge_attention_mask = torch.LongTensor([3 for t in range(len(edges_))]).unsqueeze(0)
        tmp_pad_attention_mask = torch.LongTensor([0 for t in range(padding_num)]).unsqueeze(0)
        tmp_attention_mask = torch.cat([tmp_edge_attention_mask, tmp_pad_attention_mask],dim=-1)
        edge_masks.append(tmp_attention_mask)
        edge_ids.append(tmp_edge_ids)
    edge_ids = torch.cat(edge_ids, dim=0)
    edge_masks = torch.cat(edge_masks, dim=0)
    return edge_ids, edge_masks

def get_rerank_dataloader(
        examples,
        args,
        rank,
        bsz,
        shuffle,
        is_train,
        words_dict,
        pid_to_psg,
        tokenizer,
        is_amr=False,
        is_inference=False):

    def _collate_fn(batch):
        random.seed(args.seed)
        ret_ex = {}
        ret_ex['retrieved_pids'] = []
        ret_ex['positive_pids'] = []
        labels = []
        query_psgs = []
        for i, ex in enumerate(batch):
            truncated_query = tokenizer.tokenize(ex['question'])[:args.max_query_length]
            truncated_query = tokenizer.convert_tokens_to_string(truncated_query)
            if is_train:
                if args.all4train:
                    positive_pids = ex['positive_ids']
                    negative_pids = ex['negative_ids']
                    label = torch.cat([torch.ones(len(positive_pids)), torch.zeros(len(negative_pids))], dim=0).type(torch.long)
                    pos_psgs = [pid_to_psg[positive_pid]['text'] for positive_pid in positive_pids]
                    neg_psgs = [pid_to_psg[negative_pid]['text'] for negative_pid in negative_pids]
                    psgs = pos_psgs + neg_psgs
                else:
                    positive_id = random.randint(0, len(ex['positive_ids'])-1)
                    neg_num = [i for i in range(len(ex['negative_ids'][:50]))]
                    negative_ids = random.sample(neg_num, k=args.num_negative_psg)
                    positive_pid = ex['positive_ids'][positive_id]
                    psgs = [pid_to_psg[positive_pid]['text']]
                    for id in negative_ids:
                        psgs.append(pid_to_psg[ex['negative_ids'][id]]['text'])
                    label = torch.cat([torch.ones(1), torch.zeros(args.num_negative_psg)], dim=0).type(torch.long)
                labels.append(label)
            elif is_inference:
                psgs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['text'])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])
            else:
                psgs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['text'])
                ret_ex['positive_pids'].append(ex['positive_ids'])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])

            for psg in psgs:
                query_psgs.append(truncated_query + ' ? ' + psg)

        inputs = tokenizer(
            query_psgs,
            max_length=args.max_combined_length,
            truncation=True,
            return_tensors='pt',
            padding=True)
        ret_ex['inputs'] = inputs
        ret_ex['labels'] = torch.cat(labels, dim=0) if len(labels) > 0 else None
        return ret_ex

    def _collate_fn_amr(batch):
        random.seed(args.seed)
        ret_ex = {}
        ret_ex['retrieved_pids'] = []
        ret_ex['positive_pids'] = []
        labels = []
        query_psgs = []
        nodes = []
        edges = []
        for i, ex in enumerate(batch):
            truncated_query = tokenizer.tokenize(ex['question'])[:args.max_query_length]
            truncated_query = tokenizer.convert_tokens_to_string(truncated_query)
            if is_train:
                positive_id = random.randint(0, len(ex['positive_ids'])-1)
                neg_num = [i for i in range(len(ex['negative_ids'][:50]))]
                negative_ids = random.sample(neg_num, k=args.num_negative_psg)
                positive_pid = ex['positive_ids'][positive_id]
                psgs = [pid_to_psg[positive_pid]['title'] + ' . ' + pid_to_psg[positive_pid]['text']]
                amrs = [ex['ret_nodes_edges'][positive_id]]
                for id in negative_ids:
                    psgs.append(pid_to_psg[ex['negative_ids'][id]]['title']+' . '+pid_to_psg[ex['negative_ids'][id]]['text'])
                    amrs.append(ex['ret_nodes_edges'][id])
                label = torch.cat([torch.ones(1), torch.zeros(args.num_negative_psg)], dim=0).type(torch.long)
                labels.append(label)
            elif is_inference:
                psgs = []
                amrs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['title'] + ' . ' + pid_to_psg[pid]['text'])
                    amrs.append(ex['ret_nodes_edges'][id])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])
            else:
                psgs = []
                amrs = []
                for id, pid in enumerate(ex['retrieved_ids']):
                    psgs.append(pid_to_psg[pid]['title'] + ' . ' + pid_to_psg[pid]['text'])
                    amrs.append(ex['ret_nodes_edges'][id])
                ret_ex['positive_pids'].append(ex['positive_ids'])
                ret_ex['retrieved_pids'].append(ex['retrieved_ids'])

            for psg in psgs:
                query_psgs.append(truncated_query + ' ? '  + psg)
            for amr in amrs:
                nodes.append(amr[0])
                edges.append(amr[1])

        inputs = tokenizer(
            query_psgs,
            max_length=args.max_combined_length,
            truncation=True,
            return_tensors='pt',
            padding=True)

        if args.only_nodes:
            graph_ids, graph_mask = encode_nodes(nodes, args.node_length, words_dict)
        elif args.only_edges:
            graph_ids, graph_mask = encode_edges(edges, args.edge_length, words_dict)
        else:
            graph_ids, graph_mask = encode_nodes_edges(nodes, edges, args.node_length+args.edge_length, words_dict)

        inputs.data.update({'graph_ids':graph_ids, 'graph_mask':graph_mask})
        ret_ex['inputs'] = inputs
        ret_ex['labels'] = torch.cat(labels, dim=0) if len(labels) > 0 else None
        return ret_ex

    # sampler = DistributedSampler(examples, num_replicas=args.world_size, rank=rank, shuffle=shuffle)
    # dl = DataLoader(examples, batch_size=1, collate_fn=_collate_fn, sampler=sampler)
    if is_amr:
        dl = DataLoader(examples, batch_size=bsz, collate_fn=_collate_fn_amr, shuffle=shuffle)
    else:
        dl = DataLoader(examples, batch_size=bsz, collate_fn=_collate_fn, shuffle=shuffle)
    return dl

def get_words_dict(tokenizer, args):
    words_set = set(['Self', ':same'])
    nodes_edges_file = open(args.nodes_edges_file, 'r', encoding='utf8')
    nodes_edges = [n_e.strip() for n_e in nodes_edges_file.readlines()]
    words_set.update(nodes_edges)
    dict = {
        word: tokenizer.encode_plus(word, max_length=args.word_length, padding='max_length', return_tensors='pt', truncation=True)
        for word in tqdm(words_set)}
    dict['Padding-None'] = {'input_ids':torch.LongTensor([[0 for i in range(args.word_length)]])}
    return dict