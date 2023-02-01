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





def load_examples_test(dataset):
    examples = []


    for i, item in enumerate(dataset["validation"]):
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
        new_char_spans = [0]*len(entity_spans)
        entity_spans.sort(key=lambda y:y[0])
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
            counter+=1
        text += " ".join(tokens[cur:])
        text = text.rstrip()
        # get true labels
        labels_pairs = tuple(zip(item["labels"]["head"], item["labels"]["tail"], item["labels"]["relation_id"]))
        entity_spans = [tuple(l) for l in entity_spans]
        oldToNewPos =  dict(zip(entity_spans, new_char_spans))
        entities = item["vertexSet"]
        correlations = []
        for pair in labels_pairs:
            for head in entities[pair[0]]:
                if tuple(head["pos"]) in oldToNewPos:
                    head["pos"]=oldToNewPos[tuple(head["pos"])]
                for tail in entities[pair[1]]:
                    if tuple(tail["pos"]) in oldToNewPos:
                        tail["pos"] = oldToNewPos[tuple(tail["pos"])]
                    pack = tuple((head["pos"], tail["pos"], pair[2]))
                    correlations += (pack),
            item["vertexSet"] = entities
            examples.append(dict(
                text=text,
                entity_spans= [d[:][:-1] for d in correlations],
                labels = [d[:][-1] for d in correlations]
            ))
        return examples


def load_examples_competition(dataset):
    examples = []


    for i, item in enumerate(dataset["test"]):
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
        new_char_spans = [0]*len(entity_spans)
        entity_spans.sort(key=lambda y:y[0])
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
            counter+=1
        text += " ".join(tokens[cur:])
        text = text.rstrip()
        aux_head = 0
        aux_tail = 0

        labels_pairs = []
        # get true labels
        for head_id in range(len(item["vertexSet"])):
            for tail_id in range(len(item["vertexSet"])):
                if (head_id!=tail_id):
                    labels_pair = tuple([head_id, tail_id , "Na"])
                    labels_pairs.append(labels_pair)

        entity_spans = [tuple(l) for l in entity_spans]
        oldToNewPos =  dict(zip(entity_spans, new_char_spans))
        entities = item["vertexSet"]
        correlations = []
        for pair in labels_pairs:
            head = random.choice(entities[pair[0]])
            tail = random.choice(entities[pair[1]])
            entity_head_id = pair[0]
            entity_tail_id = pair[1]
            rel = pair[2]

            if tuple(head["pos"]) in oldToNewPos:
                head["pos"]=oldToNewPos[tuple(head["pos"])]
            if tuple(tail["pos"]) in oldToNewPos:
                tail["pos"] = oldToNewPos[tuple(tail["pos"])]
            pack = tuple((head["pos"], tail["pos"], pair[2], tuple([entity_head_id, entity_tail_id]), item["title"]))

            item["vertexSet"] = entities
            examples.append(dict(
                text=text,
                entity_spans= pack[:2],
                labels = pack[2],
                idxs_entity_pair = pack[3],
                title = pack[4]
            ))
    return examples

dataset = load_dataset("docred")
max_value = 0
#for i, item in enumerate(dataset["train_annotated"]):
#    total_text_len = 0
#    tokens = item["sents"]
#    num_relations = len(item["labels"]["head"])


# FAZER LOAD DO MODEL FINETUNED DE 3 EPOCHS
model = load_model.model
tokenizer = load_model.tokenizer
test_examples = load_examples_competition(dataset)
maximum = 0
max_seq = 0


"""
for i in range(len(train_examples)):
    this_value = len(train_examples[i]["labels"])
    this_seq = len(train_examples[i]["text"])
    if maximum <= this_value:
        maximum = this_value
        i_max_pairs = i
    if max_seq <= this_seq:
        max_seq = this_seq
        i_max_seq = i
print(maximum)
"""


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


# Convert to inputs
for batch_start_idx in trange(0, len(test_examples), len(test_examples)):
    batch_examples = test_examples[batch_start_idx:batch_start_idx+len(test_examples)]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    #gold_labels = [example["labels"] for example in batch_examples]
    idxs_entity_pair = [example["idxs_entity_pair"] for example in batch_examples]
    titles = [example["title"] for example in batch_examples]


for i in range(len(entity_spans)):
    entity_spans[i] = list(entity_spans[i])

del batch_examples


c2l = ClassLabel(num_classes = 97, names = model.config.relations_code_list)
label_list_ids = [c2l.str2int(label) for label in model.config.relations_code_list]
#gold_labels_ids = [c2l.str2int(label) for label in gold_labels]
#aa = [c2l.int2str(label) for label in gold_labels_ids] # convert ints to CODE of label!! USE IN EVAL


#inputs = tokenizer(text=texts[0], entity_spans = entity_spans[0], padding = "max_length", max_length = 1024, task = "entity_pair_classification", return_tensors = "pt")
#torch.save(inputs, 'inputs_eval.pt')
#test_dataset = MyDataset(inputs, gold_labels_ids)

logging.info("Beginning of evaluation batching")
output_dir = "evaluation_LongLUKE_17Out"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.join(output_dir, 'results_myModel.json')
output_file = open(output_filename, 'w')


batch_size = 16


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
for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    idxs_entity_pair = [example["idxs_entity_pair"] for example in batch_examples]
    titles = [example["title"] for example in batch_examples]
    #gold_labels = [example["labels"] for example in batch_examples]

    #gold_labels_ids = [c2l.str2int(label) for label in gold_labels]

    for i in range(len(entity_spans)):
        entity_spans[i] = list(entity_spans[i])

    inputs = tokenizer(text = texts, entity_spans=entity_spans, return_tensors="pt", padding="max_length", max_length = 1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_indices = outputs.logits.topk(k = 2, largest=True)[1].tolist()
    clean_indices = []
    [clean_indices.append(el[1]) for el in predicted_indices]
    del predicted_indices
    predicted_labels = [c2l.int2str(pred) for pred in clean_indices]
    predicted_relation = [model.config.rel2word.get(rel) for rel in predicted_labels]
    for i in range(len(predicted_relation)):
        list_of_dicts.append(dict(
            title=titles[i],
            h_idx=idxs_entity_pair[i][0],
            t_idx = idxs_entity_pair[i][1],
            r = predicted_relation[i]
        ))
    torch.cuda.empty_cache()

json_object = json.dumps(list_of_dicts, indent = 4)
with open("results_low_loss_myModel_try2.json", "w") as outfile:
    outfile.write(json_object)


