# 11 Agosto. Treino e avaliação separado. Primeiro avaliação
import load_model

import numpy as np
import torch.nn
from transformers import LukeTokenizer, LukeForEntityClassification, LukeConfig, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from tqdm import trange
from datasets import ClassLabel, load_dataset
import json
from datasets import Dataset, load_dataset
import logging
from tqdm import trange
from pynvml import *

torch.cuda.empty_cache()

def accuracy(out, l):
    cnt = 0
    y1 = []
    y2 = []
    for x1, x2 in zip(out, l):
        yy1 = []
        yy2 = []
        top = max(x1)
        for i in range(len(x1)):
            if x1[i] > 0 or x1[i] == top: # adiionar como predicted o que tem prob > 0 ou prob igual ao topo
                yy1.append(i)
            if x2[i] > 0: # adicionar as labels reais para se poder comparar
                yy2.append(i)
        y1.append(yy1)
        y2.append(yy2)
        cnt += set(yy1) == set(yy2)
    return cnt, y1, y2

def gold_labels_to_numbers(labels):
    #label_list defeituosa com o _
    label_list = ["entity", "event", "group", "location", "object", "organization", "person", "place", "time"]
    c2l = ClassLabel(num_classes=9, names=label_list)
    labels_ids = [c2l.str2int(label) for label in labels]
    base = [-1]*9
# final tera dimensao n e cada dimensao sera o vetor com 0s e 1s se a label esta designated
    final = [base for i in range(len(labels))] #len test_examples
    pos = 0
    while pos < (len(final)):
        for aux in range(len(labels_ids)):
            to_modify = [0 for i in range(9)]
            indexes = labels_ids[aux]
            replacements = [1]*len(labels_ids[aux])
            for (index, replacement) in zip(indexes, replacements):
                to_modify[index] = replacement
            final[pos] = to_modify
            pos+=1
    return final


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

logging.basicConfig(level=logging.INFO)
torch.cuda.empty_cache()


def load_examples(dataset_file):
    with open(dataset_file, "r") as f:
        data = json.load(f)
    examples = []
    for item in data:
        examples.append(dict(
            text=item["sent"],
            entity_spans=[(item["start"], item["end"])],
            label=item["labels"]
        ))
    return examples


test_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/new_convertLUKE/OpenEntity/test.json")

max_pos = load_model.max_pos
logging.info("Data Memory before Loading Models")
#print_gpu_utilization()
logging.info("############### LOAD MODEL ###################")
#my_config = LukeConfig.from_json_file("./ET_for_FIGER_model_v2/config.json")
#print(my_config)
# model LongLuke NoGlobal - OpenEntity
model = load_model.model
#model.config = my_config
#model.luke.config = my_config
model.eval()
# Load the tokenizer
tokenizer = load_model.tokenizer
logging.info("Data Memory after Loading Models")
#print_gpu_utilization()

logging.info("CHOOSE GPU")
########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0  # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
model = model.to(device)
logging.info(f"The device is {device}")
batch_size = 8

num_predicted = 0
num_gold = 0
num_correct = 0

all_predictions = []
all_labels = []
pred = []
true = []
eval_loss, eval_accuracy = 0,0
nb_eval_steps, nb_eval_examples = 0,0

model.to(device)
#model.cuda()
model.eval()
for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding = "max_length", max_length = max_pos)
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    num_gold += sum(len(l) for l in gold_labels)
    for logits, labels in zip(outputs.logits, gold_labels):
        for index, logit in enumerate(logits):
            if logit > 0:
                num_predicted += 1
                predicted_label = model.config.id2label[index]
                if predicted_label in labels:
                    num_correct += 1

    gold_labels_ids = gold_labels_to_numbers(gold_labels)
    #print(model.config)
    tmp_eval_accuracy, tmp_pred, tmp_true = accuracy(outputs.logits, gold_labels_ids)
    pred.extend(tmp_pred)
    true.extend(tmp_true)
    #eval_accuracy += tmp_eval_accuracy
model.cpu()
torch.cuda.empty_cache()
precision = num_correct / num_predicted
recall = num_correct / num_gold
#f1 = 2 * precision * recall / float(precision + recall)



def f1(p, r):
    if r == 0.:
        return 0.
    return 2 * p * r / float(p + r)


def loose_macro(true, pred):
    num_entities = len(true)
    p = 0.
    r = 0.
    for true_labels, predicted_labels in zip(true, pred):
        if len(predicted_labels) > 0:
            p += len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
        if len(true_labels):
            r += len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
    precision = p / num_entities
    recall = r / num_entities
    return precision, recall, f1(precision, recall)


def loose_micro(true, pred):
    num_predicted_labels = 0.
    num_true_labels = 0.
    num_correct_labels = 0.
    for true_labels, predicted_labels in zip(true, pred):
        num_predicted_labels += len(predicted_labels)
        num_true_labels += len(true_labels)
        num_correct_labels += len(set(predicted_labels).intersection(set(true_labels)))
    if num_predicted_labels > 0:
        precision = num_correct_labels / num_predicted_labels
    else:
        precision = 0.
    recall = num_correct_labels / num_true_labels
    return precision, recall, f1(precision, recall)



micro_f1 = loose_micro(true,pred)
macro_f1 = loose_macro(true,pred)
f1_score = f1(recall, precision)


print(f"\n\nprecision: {precision} recall: {recall} f1: {f1_score}")

with open("LongLukeOpenEntity_16Agosto.txt", "w") as text_file:
    text_file.write(f"RESULTS \n precision: {precision} \n Recall: {recall} \n Micro-f1: {micro_f1} \n Macro-f1: {macro_f1}\n F1: {f1_score}")

