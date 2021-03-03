

import torch
from transformers import BertTokenizer, BertModel
from transformers import BertForTokenClassification, BertPreTrainedModel, AdamW

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from tqdm import tqdm, trange

class BertSimpleEntityMaskTyping(BertPreTrainedModel):

    def __init__(self, config):
        super(BertSimpleEntityMaskTyping, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                ent_start=None):

        # Get the embedding for paraphrase
        # outputs = self.bert(input_ids[:, 0, :], attention_mask=attention_mask[:, 0, :])
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # outputs = self.bert(input_ids_new, attention_mask=attention_mask_new)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Take the representation for entity start mask token
        ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, ent_start)])

        #         bag_output = self.dropout(pooled_output)
        bag_output = self.dropout(ent_output)
        #         print("bag_output.size()", bag_output.size())

        logits = self.classifier(bag_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)