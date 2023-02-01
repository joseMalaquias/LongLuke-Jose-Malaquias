import logging

from datasets import load_dataset
from datasets import ClassLabel
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification, TrainingArguments, Trainer
import torch
from tqdm import trange
# construir função que converta spans de relativos a frase para globais
import load_model
import random
import os
import json
#from pynvml import *


#def print_gpu_utilization():
#    nvmlInit()
#    handle = nvmlDeviceGetHandleByIndex(0)
#    info = nvmlDeviceGetMemoryInfo(handle)
#    print(f"GPU memory occupied: {info.used//1024**2} MB.")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def convert_spans(item):
    sents = []
    sent_map = []

    entities = item["vertexSet"]
    entity_start, entity_end = [], []
    mention_types = []
    entity_spans = []
    for entity in entities:
        for mention in entity:
            if mention["sent_id"] != 0:
                current_id = mention["sent_id"]
                mention["pos"] = [sum(len(s) for s in item["sents"][:current_id])+mention["pos"][0],
                    sum(len(s) for s in item["sents"][:current_id])+mention["pos"][1]]
                mention["sent_id"] = 0

            pos = mention["pos"]
            mention_types.append(mention['type'])
            entity_spans.append(pos)
    item["vertexSet"] = entities
    return item, entity_spans




def extendDataset():
    examples = []
    num_positive = 0
    num_negative = 0

    f = open('test_revised.json')
    data = json.load(f)


    for doc in data:
        ix = 0
        head = [0] * len(doc["labels"])
        tail = [0] * len(doc["labels"])
        relation_id = [0] * len(doc["labels"])
        evidence = [0] * len(doc["labels"])
        for label in doc["labels"]:
            head[ix] = label["h"]
            tail[ix] = label["t"]
            relation_id[ix] = label["r"]
            evidence[ix] = label["evidence"]
            ix +=1

        doc["labels"] = dict(
            head = head,
            tail = tail,
            r = relation_id,
            evidence = evidence
        )

    for i, item in enumerate(data):
        negatives_in_document = 0
        concat_tokens = []
        counter = 0
        converted_item, entity_spans = convert_spans(item)
        tokens = item["sents"]
        for j in range(len(tokens)):
            concat_tokens += tokens[j]
        del j
        tokens = concat_tokens
        del concat_tokens

        # new
        text = ""
        cur = 0
        new_char_spans = [0] * len(entity_spans)
        entity_spans.sort(key=lambda y: y[0])
        for target_entity in entity_spans:
            tamanho_texto = len(text)
            text += " ".join(tokens[cur: target_entity[0]])
            if text:
                text += " "
            char_start = len(text)
            text += " ".join(tokens[target_entity[0]: target_entity[1]])
            char_end = len(text)
            new_char_spans[counter] = (char_start, char_end)
            text += " "
            cur = target_entity[1]
            counter += 1
        text += " ".join(tokens[cur:])
        text = text.rstrip()
        aux_head = 0
        aux_tail = 0

        labels_pairs = []
        # get true labels

        relation_id = "Na"
        for head_id in range(len(item["vertexSet"])):
            for tail_id in range(len(item["vertexSet"])):
                if (head_id != tail_id):
                    for x in range(len(item["labels"]["head"])):
                        if (item["labels"]["head"][x]==head_id) and (item["labels"]["tail"][x] ==tail_id):
                            relation_id = item["labels"]["r"][x]
                            num_positive+=1
                            break
                        else:
                            relation_id = "Na"
                            num_negative+=1
                    labels_pair = tuple([head_id, tail_id, relation_id])
                    labels_pairs.append(labels_pair)

        entity_spans = [tuple(l) for l in entity_spans]
        oldToNewPos = dict(zip(entity_spans, new_char_spans))
        entities = item["vertexSet"]
        correlations = []
        for pair in labels_pairs:
            for head in entities[pair[0]]:
                for tail in entities[pair[1]]:
                    entity_head_id = pair[0]
                    entity_tail_id = pair[1]
                    rel = pair[2]

                    if tuple(head["pos"]) in oldToNewPos:
                        head["pos"] = oldToNewPos[tuple(head["pos"])]
                    if tuple(tail["pos"]) in oldToNewPos:
                        tail["pos"] = oldToNewPos[tuple(tail["pos"])]
                    pack = tuple(
                        (head["pos"], tail["pos"], pair[2], tuple([entity_head_id, entity_tail_id]), item["title"]))

                    item["vertexSet"] = entities
                    examples.append(dict(
                        text=text,
                        entity_spans=pack[:2],
                        labels=pack[2],
                        idxs_entity_pair=pack[3],
                        title=pack[4]
                    ))
    return examples

dataset = extendDataset()
random.shuffle(dataset)
max_value = 0


# FAZER LOAD DO MODEL FINETUNED DE 3 EPOCHS
model = load_model.model
tokenizer = load_model.tokenizer
maximum = 0
max_seq = 0

c2l = ClassLabel(num_classes = 97, names = model.config.relations_code_list)
label_list_ids = [c2l.str2int(label) for label in model.config.relations_code_list]



logging.info("Memory before choosing GPU")
#torch.cuda.empty_cache()

########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0 # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
#model = model.to(device)
#model.eval()

logging.info("Beginning of evaluation batching")
output_dir = "evaluation_LongLUKE_reDOCRED"

batch_size = 64


num_predicted = 0
num_gold = 0
num_correct = 0
this_pair = []
all_pairs = []
list_of_dicts = []

torch.cuda.empty_cache()
#print_gpu_utilization()

model.eval()
model.to(device)
evidence = []
for batch_start_idx in trange(0, len(dataset), batch_size):
    batch_examples = dataset[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    idxs_entity_pair = [example["idxs_entity_pair"] for example in batch_examples]
    titles = [example["title"] for example in batch_examples]
    gold_labels = [example["labels"] for example in batch_examples]
    gold_labels_ids = [c2l.str2int(label) for label in gold_labels]

    for i in range(len(entity_spans)):
        entity_spans[i] = list(entity_spans[i])


    inputs = tokenizer(text = texts, entity_spans=entity_spans, return_tensors="pt", padding="max_length", max_length = 1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_indices = outputs.logits.argmax(-1)
    predicted_labels = [c2l.int2str(pred) for pred in predicted_indices.tolist()]
    predicted_relation = [model.config.rel2word.get(rel) for rel in predicted_labels]

    num_predicted += len(predicted_relation)

    comparison = list(zip(predicted_relation, gold_labels))
    for pack in comparison:
        if pack[0] == pack[1]:
            num_correct += 1

    for i in range(len(predicted_relation)):
        list_of_dicts.append(dict(
            title=titles[i],
            h_idx=idxs_entity_pair[i][0],
            t_idx = idxs_entity_pair[i][1],
            r = predicted_relation[i]
        ))
    torch.cuda.empty_cache()
    
num_gold = len(dataset)
precision = num_correct / num_predicted
recall = num_correct / num_gold
f1 = 2 * precision * recall / (precision + recall)
with open("results_reDocRED_2epochs.txt", "w") as text_file:
    text_file.write(f"RESULTS \n correct = {num_correct}, gold = {num_gold}, predicted = {num_predicted}, precision: {precision} \n Recall: {recall} \n F1: {f1}")

json_object = json.dumps(list_of_dicts, indent = 4)
with open("results_reDOCRED_2epochs.json", "w") as outfile:
    outfile.write(json_object)



