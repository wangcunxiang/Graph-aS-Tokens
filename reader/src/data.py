# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import copy
import re

import torch
import random
import json
from tqdm import tqdm
from torch.nn import ZeroPad2d

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 amr_prefix='amr:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.amr_prefix = amr_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target'] + ' </s>'
            return target
        elif 'answers' in example:
            return random.choice(example['answers']) + ' </s>'
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}" + self.amr_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [float(c['score']) for c in contexts]
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

class AMRDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 opt,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:',
                 amr_prefix='amr:',
                 tokenizer=None):
        self.data = data
        self.n_context = opt.n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.amr_prefix = amr_prefix
        self.tokenizer = tokenizer
        self.graph_maxlength = opt.graph_maxlength
        self.graph_as_token = opt.graph_as_token
        self.no_relation = opt.no_relation
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        elif 'answers' in example:
            return random.choice(example['answers'])
        else:
            return None

    def get_symmetric_graph(self, graph):
        edge_len = len(graph[0])
        for i in range(edge_len):
            for j in range(i, edge_len):
                if graph[i][j] != 'Self' and graph[i][j] != "None":
                    graph[j][i] = graph[i][j].replace(':', '+:')
        return graph

    def update_matrix(self, psg_nodes, psg_nodes_dict, psg_graph, question_nodes, ques_nodes_dict, question_graph):
        nodes_all = question_nodes + psg_nodes
        nodes_length = min(len(nodes_all), self.graph_maxlength)
        ques_nodes_length = len(question_nodes)
        # matrix = [['Padding-None' if i != j else 'Self' for i in range(nodes_length)] for j in
        #                    range(nodes_length)]
        matrix = [['Padding-None' for i in range(nodes_length)] for j in range(nodes_length)]
        for i, node in enumerate(nodes_all):
            if i >= nodes_length:
                break
            if i < ques_nodes_length:
                graph = question_graph
                nodes_dict = ques_nodes_dict
            else:
                graph = psg_graph
                nodes_dict = psg_nodes_dict
            if node not in graph.keys():
                continue
            row = i
            if row >= self.graph_maxlength:
                break
            for target in graph[node]:
                if type(target[1]) == list:
                    continue
                else:
                    col = nodes_dict[target[1]]
                    if col >= self.graph_maxlength:
                        continue
                    matrix[row][col] = target[0]#.replace(':', '+:')
                    #matrix[col][row] = target[0]#.replace(':', '-:')
        return matrix

    def get_matrix_from_nodes_edges(self, nodes, edges, nodes_id2pos):
        nodes_length = min(len(nodes), self.graph_maxlength)
        matrix = [['Padding-None' if i != j else 'Self' for i in range(nodes_length)] for j in
                  range(nodes_length)]
        for edge in edges:
            if edge[0] in nodes_id2pos and edge[2] in nodes_id2pos:
                start = nodes_id2pos[edge[0]]
                end = nodes_id2pos[edge[2]]
                matrix[start][end] = edge[1]
        return matrix

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            if self.n_context > len(example['ctxs']):
                diff =  self.n_context - len(example['ctxs'])
                for ii in range(diff):
                    example['ctxs'].append(example['ctxs'][ii])
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            relations = []
            # nodess_combined = [question_nodes+nodess[i] for i in range(self.n_context)]
            # nodess = [c['nodes'].strip('|$').strip().split('|$') for c in contexts]
            # edgess = [c['edges'].strip('|$').strip().split('|$') for c in contexts]
            nodes_ids = []
            nodess_tokens = []
            edgess_tokens = []

            for i in range(self.n_context):
                # nodes = [node.split('\t') for node in nodess[i]][:self.graph_maxlength]
                # edges = [edge.split('\t') for edge in edgess[i]]
                # nodes_id2pos = {node[0].strip(): i for i, node in enumerate(nodes)}
                # nodes_id2token = {node[0].strip(): re.sub(re.compile(r"-([0-9]([0-9]))"), "", node[1].strip()).strip() for
                #                   i, node in enumerate(nodes)}
                # nodes_tokens = [re.sub(re.compile(r"-([0-9]([0-9]))"), "", node[1].strip()).strip() for node in nodes]
                # nodess_tokens.append(nodes_tokens)
                # nodes_ids.append([node[0].strip() for i, node in enumerate(nodes)])
                # edges_ = [edge for edge in edges if edge[0] in nodes_id2token and edge[2] in nodes_id2token]
                # edges_tokens = [node if i==1 else nodes_id2token[node] for edge in edges_ for i, node in enumerate(edge)]
                # edges_tokens = [[edges_tokens[j], edges_tokens[j+1], edges_tokens[j+2]] for j in range(0, len(edges_tokens), 3)]
                # edgess_tokens.append(edges_tokens)
                nodes = contexts[i]['nodes'][:self.graph_maxlength]
                edges = contexts[i]['edges'][:self.graph_maxlength - len(nodes)]
                nodes_tokens = nodes[:]
                nodes_tokens_len = len(nodes_tokens)
                edges_tokens = [[nodes_tokens[edge[0]], edge[1], nodes_tokens[edge[2]]] for edge in edges if
                                edge[0] < nodes_tokens_len and edge[2] < nodes_tokens_len]
                nodess_tokens.append(nodes_tokens)
                edgess_tokens.append(edges_tokens)
                if self.graph_as_token or self.no_relation:
                    relations.append(None)
                    continue
                else:
                    matrix = self.get_matrix_from_nodes_edges(nodes, edges, nodes_id2pos)
                    relations.append(matrix)
            # scores = [float(c['score']) for c in contexts]
            # scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores, nodes, nodes_ids, nodess_tokens, edgess_tokens, relations = None, None, None, None, None, None, None
        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            # 'nodes' : nodes_ids,
            'nodes_token' : nodess_tokens,
            'edges_token' : edgess_tokens,
            'relations' : relations
            #'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()

def encode_graphs(batch_nodes, batch_rels, graph_max_length, word_maxlength, words_dict=None):
    batch_graph_ids, batch_graph_masks = [], []
    batch_rel_ids = []
    for i in range(len(batch_nodes)):
        nodes = batch_nodes[i]
        rels = batch_rels[i]
        graph_ids, graph_masks = [], []
        rel_ids = []
        for j in range(len(nodes)):
            nodes_ = nodes[j][:graph_max_length]
            if j >= len(rels):
                print(j)
                print(rels)
                print(nodes)
            rels_ = rels[j]

            if len(nodes_) == 0:
                print()
                nodes_ = ["",]
            tmp_graph_ids = torch.cat([words_dict[node]['input_ids'] for node in nodes_]).unsqueeze(0)
            zero_pad = ZeroPad2d(padding=(0,0,0, graph_max_length-tmp_graph_ids.shape[1]))
            tmp_graph_ids = zero_pad(tmp_graph_ids)
            graph_ids.append(tmp_graph_ids)
            tmp_attention_mask = torch.LongTensor([1 if t.bool().any() else 0 for t in tmp_graph_ids[0]]).unsqueeze(0)
            graph_masks.append(tmp_attention_mask)

            tmp_rels_ids = []
            for k in range(len(rels_)):
                tmp_rel_ids = torch.cat([words_dict[rel]['input_ids'] for rel in rels_[k]]).unsqueeze(0)
                zero_pad = ZeroPad2d(padding=(0, 0, 0, graph_max_length - tmp_rel_ids.shape[1]))
                tmp_rel_ids = zero_pad(tmp_rel_ids)
                tmp_rels_ids.append(tmp_rel_ids)
            zeros_rel_ids = torch.zeros((max(graph_max_length-len(tmp_rels_ids),0), graph_max_length, word_maxlength), dtype=torch.long)
            tmp_rels_ids.append(zeros_rel_ids)
            tmp_rels_ids = torch.cat(tmp_rels_ids, dim=0)
            tmp_rels_ids = tmp_rels_ids.unsqueeze(dim=0)
            rel_ids.append(tmp_rels_ids)
        graph_ids = torch.cat(graph_ids, dim=0).unsqueeze(dim=0)
        graph_masks = torch.cat(graph_masks, dim=0).unsqueeze(dim=0)
        batch_graph_ids.append(graph_ids)
        batch_graph_masks.append(graph_masks)
        rel_ids = torch.cat(rel_ids, dim=0).unsqueeze(dim=0)
        batch_rel_ids.append(rel_ids)

    batch_graph_ids = torch.cat(batch_graph_ids, dim=0)
    batch_graph_masks = torch.cat(batch_graph_masks, dim=0)
    batch_rel_ids = torch.cat(batch_rel_ids, dim=0)

    return batch_graph_ids, batch_graph_masks.bool(), batch_rel_ids

def encode_nodes(batch_nodes, graph_max_length, words_dict=None):
    batch_graph_ids, batch_graph_masks = [], []
    for i in range(len(batch_nodes)):
        nodes = batch_nodes[i]
        graph_ids, graph_masks = [], []
        for j in range(len(nodes)):
            nodes_ = nodes[j][:graph_max_length]
            if len(nodes_) == 0:
                print()
                nodes_ = ["", ]
            tmp_graph_ids = torch.cat([words_dict[node]['input_ids'] for node in nodes_]).unsqueeze(0)
            zero_pad = ZeroPad2d(padding=(0, 0, 0, graph_max_length - tmp_graph_ids.shape[1]))
            tmp_graph_ids = zero_pad(tmp_graph_ids)
            graph_ids.append(tmp_graph_ids)
            tmp_attention_mask = torch.LongTensor([1 if t.bool().any() else 0 for t in tmp_graph_ids[0]]).unsqueeze(0)
            graph_masks.append(tmp_attention_mask)
        graph_ids = torch.cat(graph_ids, dim=0).unsqueeze(dim=0)
        graph_masks = torch.cat(graph_masks, dim=0).unsqueeze(dim=0)
        batch_graph_ids.append(graph_ids)
        batch_graph_masks.append(graph_masks)
    batch_graph_ids = torch.cat(batch_graph_ids, dim=0)
    batch_graph_masks = torch.cat(batch_graph_masks, dim=0)

    return batch_graph_ids, batch_graph_masks.bool()

def encode_nodes_edges(batch_nodes, batch_edges, graph_max_length, words_dict=None):
    batch_graph_ids, batch_graph_masks = [], []
    for i in range(len(batch_nodes)):
        nodes = batch_nodes[i]
        edges = batch_edges[i]
        graph_ids, graph_masks = [], []
        for j in range(len(nodes)):
            nodes_ = nodes[j][:graph_max_length-1]
            edges_ = edges[j][:max(graph_max_length-len(nodes_),1)]
            if len(nodes_) == 0:
                nodes_ = ["", ]
            tmp_graph_ids = torch.cat([words_dict[node]['input_ids'] for node in nodes_]).unsqueeze(0)
            tmp_graph_ids = tmp_graph_ids.unsqueeze(-2)
            tmp_graph_ids = torch.cat([tmp_graph_ids, tmp_graph_ids, tmp_graph_ids], dim=-2)

            tmp_node0_ids = torch.cat([words_dict[edge[0]]['input_ids'] for edge in edges_]).unsqueeze(0)
            tmp_rel_ids = torch.cat([words_dict[edge[1]]['input_ids'] for edge in edges_]).unsqueeze(0)
            tmp_node1_ids = torch.cat([words_dict[edge[2]]['input_ids'] for edge in edges_]).unsqueeze(0)
            tmp_edge_ids = torch.cat(
                [tmp_node0_ids.unsqueeze(-2), tmp_rel_ids.unsqueeze(-2), tmp_node1_ids.unsqueeze(-2)], dim=-2)
            tmp_graph_ids = torch.cat([tmp_graph_ids, tmp_edge_ids], dim=1)
            padding_num = graph_max_length - tmp_graph_ids.shape[1]
            zero_pad = ZeroPad2d(padding=(0, 0, 0, 0, 0, padding_num))
            tmp_graph_ids = zero_pad(tmp_graph_ids)

            tmp_node_attention_mask = torch.LongTensor([2 for t in range(len(nodes_))]).unsqueeze(0)
            tmp_edge_attention_mask = torch.LongTensor([3 for t in range(len(edges_))]).unsqueeze(0)
            tmp_pad_attention_mask = torch.LongTensor([0 for t in range(padding_num)]).unsqueeze(0)
            # tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_pad_attention_mask], dim=-1)
            tmp_attention_mask = torch.cat([tmp_node_attention_mask, tmp_edge_attention_mask, tmp_pad_attention_mask],dim=-1)
            graph_masks.append(tmp_attention_mask)
            graph_ids.append(tmp_graph_ids)
        graph_ids = torch.cat(graph_ids, dim=0).unsqueeze(dim=0)
        graph_masks = torch.cat(graph_masks, dim=0).unsqueeze(dim=0)
        batch_graph_ids.append(graph_ids)
        batch_graph_masks.append(graph_masks)
    batch_graph_ids = torch.cat(batch_graph_ids, dim=0)
    batch_graph_masks = torch.cat(batch_graph_masks, dim=0)

    return batch_graph_ids, batch_graph_masks


class AMRCollator(object):
    def __init__(self, opt, tokenizer, words_dict):
        self.tokenizer = tokenizer
        self.text_maxlength = opt.text_maxlength
        self.graph_maxlength = opt.graph_maxlength
        self.answer_maxlength = opt.answer_maxlength
        self.word_maxlength = opt.word_maxlength
        self.no_relation = opt.no_relation
        self.graph_as_token = opt.graph_as_token
        self.words_dict = words_dict

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        nodes = [[example['nodes_token'][j] for j, t in enumerate(example['nodes_token'])] for i, example in
                 enumerate(batch)]
        edges = [[example['edges_token'][j] for j, t in enumerate(example['edges_token'])] for i, example in
                 enumerate(batch)]
        rels = [[t for t in example['relations']] for example in batch]
        if self.graph_as_token == True:
            graph_ids, graph_masks = encode_nodes_edges(nodes, edges,
                                                          self.graph_maxlength,
                                                          words_dict=self.words_dict)
            rel_ids = None
        elif self.no_relation:
            graph_ids, graph_masks = encode_nodes(nodes,
                                                      self.graph_maxlength,
                                                      words_dict=self.words_dict)
            rel_ids = None
        else:
            graph_ids, graph_masks, rel_ids = encode_graphs(nodes, rels,
                                                     self.graph_maxlength,
                                                     word_maxlength=self.word_maxlength,
                                                     words_dict=self.words_dict)

        return (index, target_ids, target_mask, passage_ids, passage_masks, graph_ids, graph_masks, rel_ids)

class Collator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)

def load_data(data_path=None, global_rank=-1, world_size=-1):
    assert data_path
    if data_path.endswith('.jsonl'):
        data = open(data_path, 'r')
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    examples = []
    for k, example in enumerate(data):
        # if global_rank > -1 and not k%world_size==global_rank:
        #     continue
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

class RetrieverCollator(object):
    def __init__(self, tokenizer, passage_maxlength=200, question_maxlength=40):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength

    def __call__(self, batch):
        index = torch.tensor([ex['index'] for ex in batch])

        question = [ex['question'] for ex in batch]
        question = self.tokenizer.batch_encode_plus(
            question,
            padding='max_length',
            return_tensors="pt",
            max_length=self.question_maxlength,
            truncation=True
        )
        question_ids = question['input_ids']
        question_mask = question['attention_mask'].bool()

        if batch[0]['scores'] is None or batch[0]['passages'] is None:
            return index, question_ids, question_mask, None, None, None

        scores = [ex['scores'] for ex in batch]
        scores = torch.stack(scores, dim=0)

        passages = [ex['passages'] for ex in batch]
        passage_ids, passage_masks = encode_passages(
            passages,
            self.tokenizer,
            self.passage_maxlength
        )

        return (index, question_ids, question_mask, passage_ids, passage_masks, scores)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data,
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        text = self.title_prefix + " " + example[2] + " " + \
            self.passage_prefix + " " + example[1]
        return example[0], text

class TextCollator(object):
    def __init__(self, tokenizer, maxlength=200):
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __call__(self, batch):
        index = [x[0] for x in batch]
        encoded_batch = self.tokenizer.batch_encode_plus(
            [x[1] for x in batch],
            padding='max_length',
            return_tensors="pt",
            max_length=self.maxlength,
            truncation=True
        )
        text_ids = encoded_batch['input_ids']
        text_mask = encoded_batch['attention_mask'].bool()

        return index, text_ids, text_mask

def get_words_dict(tokenizer, examples, opt):
    words_set = set(['Self', ':same'])
    # [words_set.update(set(re.sub(re.compile(r"_([0-9]|([0-9]))"), " ", example['question_amr']).replace("(", " ( ").replace(")", " ) ").replace("\"", " ").split())) \
    #  for example in tqdm(examples)]
    # [words_set.update(set(re.sub(re.compile(r"_([0-9]|([0-9]))"), " ", ctx['amr']).replace("(", " ( ").replace(")", " ) ").replace("\"", " ").split())) \
    #  for example in tqdm(examples) for ctx in example['ctxs']]
    nodes_edges_file = open(opt.nodes_edges_file, 'r', encoding='utf8')
    nodes_edges = [n_e.strip() for n_e in nodes_edges_file.readlines()]
    words_set.update(nodes_edges)
    dict = {
        word: tokenizer.encode_plus(word, max_length=opt.word_maxlength, padding='max_length', return_tensors='pt', truncation=True)
        for word in tqdm(words_set)}
    dict['Padding-None'] = {'input_ids':torch.LongTensor([[0 for i in range(opt.word_maxlength)]])}
    return dict