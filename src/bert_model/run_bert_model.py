
import numpy as np
import pandas as pd
from collections import defaultdict

import torch 
from torch import nn 

import re

import transformers
from transformers import BertTokenizer

from bert_model import Classifier

from sklearn.metrics import confusion_matrix

from copy import deepcopy

import sys 
import os 

sys.path.append(os.getcwd())

from src import data_sources as ds
from src.bert_model import parameters as p


def train_epoch(
  model,
  encoded_plus_list,
  target_list,
  loss_fn,
  optimizer,
  device,
  n_examples,
  batch_size=16
):

  model = model.train()
  losses = []
  correct_predictions = 0
  total_pos_preds = 0
  correct_pos_preds = 0
  actual_pos = 0


  for i in range(len(encoded_plus_list) // batch_size + 1):

    optimizer.zero_grad()

    this_batch = encoded_plus_list[i * batch_size : (i + 1) * batch_size]

    step_1 = [ele["input_ids"] for ele in this_batch]

    if len(step_1) == 0:
        continue

    input_ids = torch.stack(step_1).squeeze(dim=1)
    input_ids = input_ids.to(device)

    step_1 = [ele["attention_mask"] for ele in this_batch]
    attention_mask = torch.stack(step_1).squeeze(dim=1)
    attention_mask = attention_mask.to(device)


    targets = target_list[i * batch_size : (i+1) * batch_size]

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)

    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds == targets).double()
    total_pos_preds += torch.sum(preds).double()
    correct_pos_preds += torch.sum(preds[preds == 1] == targets[preds == 1]).double()
    actual_pos += torch.sum(targets == 1).sum().double()
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

  accuracy = (correct_predictions / n_examples).item()
  precision = correct_pos_preds / total_pos_preds
  recall = correct_pos_preds / total_pos_preds

  f1_score = (precision * recall * 2) / (precision + recall).item()

  return accuracy, f1_score, np.mean(losses)


def eval_model(
    model,
    encoded_plus_list,
    target_list,
    loss_fn,
    device,
    n_examples,
    batch_size=128
    ):

  model = model.eval()
  losses = []
  correct_predictions = 0
  total_pos_preds = 0
  correct_pos_preds = 0
  actual_pos = 0
  predictions_list = []

  with torch.no_grad():

    for i in range(len(encoded_plus_list) // batch_size + 1):

      this_batch = encoded_plus_list[i * batch_size : (i + 1) * batch_size]
      step_1 = [ele["input_ids"] for ele in this_batch]

      if len(step_1) == 0:
          continue

      input_ids = torch.stack(step_1).to(device).squeeze()

      step_1 = [ele["attention_mask"] for ele in this_batch]
      attention_mask = torch.stack(step_1).to(device).squeeze()

      this_batch_targets = target_list[i * batch_size : (i+1) * batch_size]


      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )

      _, this_batch_preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, this_batch_targets)
      correct_predictions += torch.sum(this_batch_preds == this_batch_targets).double()
      total_pos_preds += torch.sum(this_batch_preds).double()
      correct_pos_preds += torch.sum(this_batch_preds[this_batch_preds == 1] == this_batch_targets[this_batch_preds == 1]).double()
      actual_pos += torch.sum(this_batch_targets == 1).sum().double()
      losses.append(loss.item())
      predictions_list.append(this_batch_preds)


    predictions = torch.cat(predictions_list)

    cf_matrix = confusion_matrix(target_list.cpu(), predictions.cpu())

  accuracy = (correct_predictions / n_examples).item()
  precision = correct_pos_preds / total_pos_preds
  recall = correct_pos_preds / total_pos_preds

  f1_score = (precision * recall * 2).item() / (precision + recall).item()

  return accuracy, f1_score, np.mean(losses).item(), cf_matrix



def build_model_and_get_results(encoded_plus_list, 
                                labels_torch,
                                part,
                                n_parts,
                                device,
                                n_classes,
                                learning_rate,
                                epochs
                                ):

    val_len = len(encoded_plus_list) // n_parts
    
    start = val_len * part 
    end = val_len * (part + 1)

    encoded_plus_val = encoded_plus_list[start:end]
    labels_val = labels_torch[start:end]

    encoded_plus_train = encoded_plus_list[:start] + encoded_plus_list[end:]
    labels_train = torch.cat([labels_torch[:start], labels_torch[end:]])

    model = Classifier(n_classes=n_classes).to(device)

    optimizer = transformers.AdamW(model.parameters(),
                                   lr=learning_rate,
                                   correct_bias=False)

    loss_fn = nn.CrossEntropyLoss().to(device)
    best_f1_score = -1

    for epoch in range(epochs):


        print(f"On epoch {epoch + 1} of {epochs}")

        train_acc, train_f1_score, train_loss = train_epoch(
            model,
            encoded_plus_train,
            labels_train,
            loss_fn,
            optimizer,
            device,
            len(encoded_plus_train)
        )

        print(f'Train loss {train_loss} Train f1-score {train_f1_score} accuracy {train_acc}')

        val_acc, val_f1_score, val_loss, cf_matrix = eval_model(
            model,
            encoded_plus_val,
            labels_val,
            loss_fn,
            device,
            len(encoded_plus_val)
        )

        print(f'Val   loss {val_loss} f1-score {val_f1_score} accuracy {val_acc}')
        print(f"confusion matrix: {cf_matrix}")


        if val_f1_score > best_f1_score:

            best_model_dict = deepcopy(model.state_dict())
            best_cf_matrix = cf_matrix
            best_f1_score = val_f1_score

    torch.save(best_model_dict, p.NEW_MODEL_NAME.format(i))

    return best_f1_score, best_cf_matrix


if __name__ == "__main__":

    np.random.seed(p.SEED)

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    print("running with bert uncased...")
    print(f"using N_SUBSET of: {p.N_SUBSET}")

    df = pd.read_csv(ds.TRAIN_FILE_PATH)[:p.N_SUBSET]
    df = df.sample(frac=1) #shuffles the data

    pattern_hashtag = re.compile(r"#\w+")
    df[p.TEXT_COLUMN] = df[p.TEXT_COLUMN].apply(lambda x: pattern_hashtag.sub("", x))
    pattern_username = re.compile(r"@\w+")
    df[p.TEXT_COLUMN] = df[p.TEXT_COLUMN].apply(lambda x: pattern_username.sub("", x))
    pattern_url = re.compile(r'http\S+')  
    df[p.TEXT_COLUMN] = df[p.TEXT_COLUMN].apply(lambda x: pattern_url.sub("", x))

    test_list = df[p.TEXT_COLUMN].tolist()

    tokenizer = BertTokenizer.from_pretrained(p.PRE_TRAINED_MODEL_NAME)

    labels_torch = torch.from_numpy(np.array(df[p.LABEL_COLUMN])).to(device)

    encoded_plus_list = [] 

    for text in test_list:
        encoded_plus = tokenizer.encode_plus(
            text,
            max_length=p.MAX_LENGTH,
            truncation=p.TRUNCATION,
            add_special_tokens=p.ADD_SPECIAL_TOKENS,
            return_token_type_ids=p.RETURN_TOKEN_TYPE_IDS,
            pad_to_max_length=p.PAD_TO_MAX_LENGTH,
            return_attention_mask=p.RETURN_ATTENTION_MASK,
            return_tensors=p.RETURN_TENSORS,
        )

        encoded_plus_list.append(encoded_plus)

    temp = sum(ele["attention_mask"][0][-1].item() for ele in encoded_plus_list)
    print(f"Number of fully masked : {temp} / {len(encoded_plus_list)}")

    # accuracy_scores = []
    f1_scores = []
    cf_matrixes = []

    for i in range(p.N_PARTS):

        f1_score, cf_matrix = build_model_and_get_results(encoded_plus_list, 
                                                                labels_torch,
                                                                i,
                                                                p.N_PARTS,
                                                                device,
                                                                p.N_CLASSES,
                                                                p.LEARNING_RATE,
                                                                p.EPOCHS
                                                                )

        f1_scores.append(f1_score)
        cf_matrixes.append(cf_matrix)
    
    print("f1_scores:")
    print(f1_scores)
    print(np.mean(f1_scores))
    print("confusion matrixes")
    print(cf_matrixes)