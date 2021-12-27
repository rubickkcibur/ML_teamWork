import os
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset, load_metric, load_from_disk
import pandas as pd
import re

DATA_PATH = "./data"
DIV_PROP = 0.9
TOKEN_LEN = 256
SHUFFLE_SEED = 42
BATCH_SIZE = 128
LR = 1e-5

def process(s):
    re = s.replace("\n", " ")
    re = res.strip().split(" ")
    re = [word.lower() for word in res]
    re = " ".join(res)
    return re

def tokenize(data):
    texts = []  
    for text in data['text']:
        texts.append(text)
    re = tokenizer(texts, padding='max_length', truncation=True, max_length=TOKEN_LEN)
    return re

def tokenize_pred(data):
    texts = []
    for example in data["text"]:
        texts.append(example)
    tokenized = tokenizer(texts, padding='max_length', truncation=True, max_length=TOKEN_LEN)
    return tokenized

data = pd.read_csv(DATA_PATH + "/train.csv")
data["comment_text"] = data["comment_text"].apply(process)
train_len = int(len(data) * DIV_PROP)
data_train = data[["target", "comment_text"]][:train_len]
data_test = data[["target", "comment_text"]][train_len:]
data_train.to_csv(DATA_PATH + "/my_train.csv", index = False, header = ["labels", 'text'])
data_test.to_csv(DATA_PATH + "/my_val.csv", index = False, header = ["labels", 'text'])

pretrained_model = "./albert/"
dataset = load_dataset("csv", data_files={ 'train': DATA_PATH + "/my_train.csv", 'val': DATA_PATH + "/my_val.csv"})
tokenizer = AutoTokenizer.from_pretrained(pretrained_model, cache_dir=DATA_PATH+ '/pretrained', use_fast=True)

tokenized_datasets = dataset.map(tokenize, batched=True)
train_dataset = tokenized_datasets["train"].shuffle(seed=SHUFFLE_SEED)
eval_dataset = tokenized_datasets["val"].shuffle(seed=SHUFFLE_SEED)

model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=1, cache_dir=DATA_PATH+ '/pretrained')

training_args = TrainingArguments(
                    output_dir=DATA_PATH+'/checkpoints',
                    evaluation_strategy = "epoch",
                    save_strategy="epoch",
                    learning_rate=LR,
                    per_device_train_batch_size=BATCH_SIZE,
                    per_device_eval_batch_size=BATCH_SIZE,
                    load_best_model_at_end=True,
                    num_train_epochs=4,
                    dataloader_num_workers=4
                )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)
print("i'm going to train")
trainer.train()
print(trainer.evaluate())

test_data = pd.read_csv(DATA_PATH + "/test.csv")
test_data["comment_text"] = test_data["comment_text"].apply(process)

my_pred = test_data["comment_text"]
my_pred.to_csv(DATA_PATH + "/my_pred.csv", index = False, header = ["text"])

pred_dataset = load_dataset("csv", data_files={ 'pred': DATA_PATH + "/my_pred.csv" })
pred_dataset = pred_dataset.map(tokenize_pred, batched=True)
pred_dataset = pred_dataset["pred"]

pred = trainer.predict(test_dataset = pred_dataset)
pred = np.array(pred[0]).squeeze()
np.save(DATA_PATH + "/res_roberta.npy", pred)
print(pred)