from transformers import BertModel 
import torch
from torch import nn

import sys 
import os 

sys.path.append(os.getcwd())

from src.bert_model import parameters as p

class Classifier(nn.Module):
  def __init__(self, n_classes):
    super(Classifier, self).__init__()
    self.bert = BertModel.from_pretrained(p.PRE_TRAINED_MODEL_NAME)

    self.drop = nn.Dropout(p=p.DROPOUT_RATE)
    self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask):

    # using temp to work around the issue described here: 
    # https://github.com/huggingface/transformers/issues/8968

    temp = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    pooler_output = temp[1]

    output = self.drop(pooler_output)

    return self.out(output)