import logging

from datasets import load_dataset
from datasets import ClassLabel
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification, TrainingArguments, Trainer
import torch
from tqdm import trange
import random
# construir função que converta spans de relativos a frase para globais
import load_model
import json
from pynvml import *

torch.cuda.empty_cache()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")




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

    f = open('train_revised.json')
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


def load_examples_train(dataset):
    examples = []


    for i, item in enumerate(dataset["train_annotated"]):
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

dataset = load_dataset("docred")

max_value = 0


model = load_model.model
tokenizer = load_model.tokenizer


maximum = 0
max_seq = 0


########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0 # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
model = model.to(device)
model.train()




#### CONVERT HERE!!

extended_train_examples = extendDataset()
NA_list = []
positives_list = []
for example in extended_train_examples:
    if example["labels"] == "Na":
        NA_list.append(example)
    else:
        positives_list.append(example)
a = 0
#remove 50% of "Nas"
NA_list = random.sample(NA_list, int(len(NA_list)*(1-0.5)))
extended_train_examples = positives_list+NA_list
random.shuffle(extended_train_examples)


# Convert to inputs
for batch_start_idx in trange(0, len(extended_train_examples), len(extended_train_examples)):
    batch_examples = extended_train_examples[batch_start_idx:batch_start_idx+len(extended_train_examples)]
#    texts = [example["text"] for example in batch_examples]
#    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["labels"] for example in batch_examples]

#for i in range(len(entity_spans)):
#    entity_spans[i] = list(entity_spans[i])


c2l = ClassLabel(num_classes = 97, names = model.config.relations_code_list)
label_list_ids = [c2l.str2int(label) for label in model.config.relations_code_list]
gold_labels_ids = [c2l.str2int(label) for label in gold_labels]
#aa = [c2l.int2str(label) for label in gold_labels_ids] # convert ints to CODE of label!! USE IN EVAL
# ADD NEGATIVE EXAMPLES

model.to("cpu")

torch.cuda.empty_cache()
logging.info("Memory after deleting")
torch.cuda.empty_cache()
print_gpu_utilization()
#inputs = tokenizer(text=texts, entity_spans = entity_spans, padding = "max_length", max_length = 1024, task = "entity_pair_classification", return_tensors = "pt").to(device)
del batch_examples,  extended_train_examples, positives_list, NA_list #texts, entity_spans

#torch.save(inputs, "inputs_balanced_half_negatives.pt")



inputs = torch.load("inputs_balanced_half_negatives.pt")
train_dataset = MyDataset(inputs, gold_labels_ids)

logging.info("Beginning of trainer batching")

logging.info("Creating training arguments \n")
torch.cuda.empty_cache()
print_gpu_utilization()

training_args = TrainingArguments(
    do_train = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 4,
    gradient_checkpointing = True,
    fp16 = True,
    optim = "adafactor", # com ou sem adafactor??? 
    output_dir = "checkpoints_50_Negatives",
    num_train_epochs = 2,
    report_to = "none",
    dataloader_pin_memory = False,
    logging_strategy =  "steps",
    logging_steps = 1000,
    learning_rate = 0.01
)
model.train()
model.to(device)
logging.info("Creating Trainer class \n")
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset
)

logging.info("Asking Trainer class to begin trainning!! \n")
print_gpu_utilization()
trainer.train("/mnt/shared/home/jose.luis.malaquias.ext/reDOCRED/Revisiting_DOCRED/checkpoints_50_Negatives/checkpoint-15000")

logging.info("Save finetuned model")
trainer.save_model("model_finetuned_balanced_50_negatives")

logging.info("Training is over. Trained model is saved in model_finetuned_balanced!! \n")
print("END of TRAINING ! !")





