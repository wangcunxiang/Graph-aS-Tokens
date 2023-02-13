# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import types
import torch
import transformers
import re
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from .modeling_t5 import T5ForConditionalGeneration_graph_in_encoder
from collections import OrderedDict
from copy import deepcopy
import numpy as np

class FiDT5(T5ForConditionalGeneration_graph_in_encoder):
    def __init__(self, config, opt):
        super().__init__(config, opt)
        self.wrap_encoder()
        self.graph_maxlength = opt.graph_maxlength
        self.word_length = opt.word_maxlength

    def forward_(self, **kwargs):
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = kwargs['input_ids'].view(kwargs['input_ids'].size(0), -1)
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = kwargs['attention_mask'].view(kwargs['attention_mask'].size(0), -1)

        return super(FiDT5, self).forward(
            **kwargs
        )

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self,
                input_ids=None,
                graph_ids=None,
                attention_mask=None,
                graph_mask=None,
                **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if graph_ids != None:
            if graph_ids.dim() == 5:
                graph_ids = graph_ids.view(graph_ids.size(0), -1, 3, graph_ids.size(-1))
        # if attention_mask != None:
        #     attention_mask = attention_mask.view(attention_mask.size(0), -1)
        # if graph_mask != None:
        #     graph_mask = graph_mask.view(graph_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids,
            graph_ids=graph_ids,
            attention_mask=attention_mask,
            graph_mask=graph_mask,
            **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self,
                 input_ids=None,
                 graph_ids=None,
                 attention_mask=None,
                 graph_mask=None,
                 max_length=20,
                 **kwargs):
        self.encoder.n_passages = input_ids.size(1)
        if graph_ids != None:
            if graph_ids.dim() == 5:
                graph_ids = graph_ids.view(graph_ids.size(0), -1, 3, graph_ids.size(-1))
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            graph_ids=graph_ids,
            attention_mask=attention_mask,
            graph_mask=graph_mask,
            max_length=max_length,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict.update({k:v})
        self.load_state_dict(new_state_dict, strict=False)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.
        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores/ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
        self.base_model_prefix = ""
        self.main_input_name = "input_ids"


    def forward(self,
                input_ids=None,
                graph_ids=None,
                attention_mask=None,
                graph_mask=None,
                **kwargs,):

        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)

        bsz, total_graph_length, _, word_length = graph_ids.shape
        assert _ == 3
        graph_length = total_graph_length // self.n_passages
        graph_ids = graph_ids.view(bsz * self.n_passages, graph_length, 3, word_length)
        graph_mask = graph_mask.view(bsz * self.n_passages, graph_length)

        outputs = self.encoder(input_ids, graph_ids, attention_mask, graph_mask, **kwargs)
        outputs["last_hidden_state"] = outputs[0].view(bsz, self.n_passages*(passage_length+graph_length), -1)
        return outputs

class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output

def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block
