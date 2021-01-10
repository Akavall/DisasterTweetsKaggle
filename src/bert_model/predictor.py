

import torch
import pandas as pd
import numpy as np

import re

from transformers import BertTokenizer
from bert_model import Classifier

import sys 
import os 

sys.path.append(os.getcwd())

from src.bert_model import parameters as p
from src import data_sources as ds

def predict(model, tokenizer, loss_desc_list, df_part, device, batch_size):

    encoded_plus_list = []

    for loss_desc in loss_desc_list:
        encoded_plus = tokenizer.encode_plus(
            loss_desc,
            max_length=p.MAX_LENGTH,
            truncation=p.TRUNCATION,
            add_special_tokens=p.ADD_SPECIAL_TOKENS,
            return_token_type_ids=p.RETURN_TOKEN_TYPE_IDS,
            pad_to_max_length=p.PAD_TO_MAX_LENGTH,
            return_attention_mask=p.RETURN_ATTENTION_MASK,
            return_tensors=p.RETURN_TENSORS,
    )

        encoded_plus_list.append(encoded_plus)

    step_1 = [ele["input_ids"] for ele in encoded_plus_list]

    input_ids = torch.stack(step_1).squeeze()
    input_ids = input_ids.to(device)

    step_1 = [ele["attention_mask"] for ele in encoded_plus_list]
    attention_mask = torch.stack(step_1).squeeze()
    attention_mask = attention_mask.to(device)

    with torch.no_grad():

        probs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        _, preds_bert = torch.max(probs, dim=1)

    return preds_bert


if __name__ == "__main__":

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")

    tokenizer = BertTokenizer.from_pretrained(p.PRE_TRAINED_MODEL_NAME)

    df = pd.read_csv(ds.TEST_FILE_PATH)[:p.N_PREDICTIONS]

    pattern_hashtag = re.compile(r"#\w+")
    df[p.TEXT_COLUMN] = df[p.TEXT_COLUMN].apply(lambda x: pattern_hashtag.sub("", x))
    pattern_username = re.compile(r"@\w+")
    df[p.TEXT_COLUMN] = df[p.TEXT_COLUMN].apply(lambda x: pattern_username.sub("", x))
    pattern_url = re.compile(r'http\S+')  
    df[p.TEXT_COLUMN] = df[p.TEXT_COLUMN].apply(lambda x: pattern_url.sub("", x))

    df["keyword"] = df["keyword"].fillna("unknown")

    predictions = []
    predictions_bert = []

    for j in range(p.N_PARTS):

        model = Classifier(n_classes=p.N_CLASSES).to(device)
        model.load_state_dict(torch.load(p.NEW_MODEL_NAME.format(j)))

        print(f"processing model: {j}")

        this_model_preds_bert = []

        for i in range(len(df) // p.PRED_BATCH_SIZE + 1):

            df_part = df[i * p.PRED_BATCH_SIZE : (i + 1) * p.PRED_BATCH_SIZE]
            text_list = df["text"][i * p.PRED_BATCH_SIZE : (i + 1) * p.PRED_BATCH_SIZE].to_list()

            if len(text_list) == 0:
                continue

            this_batch_preds_bert = []

            preds_bert = predict(model, tokenizer, text_list, df_part, device, p.PRED_BATCH_SIZE)
            preds_bert_cpu = preds_bert.cpu().detach().numpy()
            this_model_preds_bert.append(preds_bert_cpu)

            if i == 0:
                print(f"model: {j}")
                print(f"preds_bert: {preds_bert_cpu}")

        del model 
        del preds_bert
        torch.cuda.empty_cache()

        this_model_preds_array_bert = np.concatenate(this_model_preds_bert)

        predictions_bert.append(this_model_preds_array_bert)

    bert_preds_t = np.vstack(predictions_bert).T
    final_preds_bert = []

    for i in range(len(bert_preds_t)):
        final_preds_bert.append(np.argmax(np.bincount(bert_preds_t[i])))

    result_df_bert = pd.DataFrame({"id": df["id"], "target": final_preds_bert})
    result_df_bert.to_csv(ds.OUTPUT_FILE_PATH, index=False)
