import torch
import torch.nn.functional as F

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    ElectraForSequenceClassification,
    ElectraTokenizer,
    BartForSequenceClassification,
    BartTokenizer,
    # Trainer
)
from models.bart.modeling_bart import GraphBartForSequenceClassification
from models.bert.modeling_bert import GraphBertForSequenceClassification
from torch.nn import CrossEntropyLoss

candidate_models = {
    'bert': (BertForSequenceClassification, BertTokenizer, 'bert-large-uncased'),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, 'roberta-large'),
    'electra': (ElectraForSequenceClassification, ElectraTokenizer, 'deepset/electra-base-squad2'),
    'bart': (BartForSequenceClassification, BartTokenizer, 'facebook/bart-large'),
    'graph_bart': (GraphBartForSequenceClassification, BartTokenizer, 'facebook/bart-large'),
    'graph_bert': (GraphBertForSequenceClassification, BertTokenizer, 'bert-large-uncased'),
}


class Reranker(torch.nn.Module):
    def __init__(self, args):
        super(Reranker, self).__init__()
        model_class, _, model_name = candidate_models[args.model]
        if 'graph' in args.model:
            self.encoder = model_class.from_pretrained(model_name, args=args)
        else:
            self.encoder = model_class.from_pretrained(model_name)
        self.num_neg_psg = args.num_negative_psg
        self.loss = args.loss

    def forward(self, inputs, labels=None, is_training=False):
        # input_ids = inputs.input_ids
        # attention_mask = inputs.attention_mask
        # token_type_ids = inputs.token_type_ids
        #
        # num_psg = input_ids.size()[0]
        # max_num_psg = 15 if is_training else 120
        # if num_psg > max_num_psg:
        #     num_split = math.ceil(num_psg / max_num_psg)
        #     logits_list = []
        #     for i in range(num_split):
        #         start = i * max_num_psg
        #         if i == num_split - 1:
        #             end = num_psg
        #         else:
        #             end = (i+1) * max_num_psg
        #         print(i, start, end)
        #         logits_list.append(self.encoder(
        #             input_ids=input_ids[start: end, :],
        #             attention_mask=attention_mask[start:end, :],
        #             token_type_ids=token_type_ids[start:end, :]
        #         ).logits)
        #     logits = torch.cat(logits_list, 0)
        # else:
        #     logits = self.encoder(**inputs).logits

        logits = self.encoder(**inputs).logits
        shape = logits.shape
        if is_training:
            loss = 0
            num = self.num_neg_psg+1
            if self.loss == 'cross_entropy':
                if labels is None:
                    labels = torch.zeros(shape[0]).to(logits.device).type(torch.long)
                    for i in range(int(shape[0] / num)):
                        labels[i * num] = 1
                labels = labels.to(logits.device)
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, shape[1]), labels.view(-1))
            elif self.loss == 'double_softmax':
                logprobs = F.log_softmax(logits, -1)[:, 1]
                logprobs = F.log_softmax(logprobs, 0)
                for i in range(int(shape[0]/num)):
                    loss -= logprobs[i * num]
            return loss
        else:
            return logits[:, 1]
