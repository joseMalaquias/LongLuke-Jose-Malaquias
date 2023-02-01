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
import load_model_ET
from configurations import my_configuration
#from pynvml import *



#def print_gpu_utilization():
#    nvmlInit()
#    handle = nvmlDeviceGetHandleByIndex(0)
#    info = nvmlDeviceGetMemoryInfo(handle)
#    print(f"GPU memory occupied: {info.used//1024**2} MB.")


#def print_summary(result):
#    print(f"Time: {result.metrics['train_runtime']:.2f}")
#    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
#    print_gpu_utilization()

logging.basicConfig(level=logging.INFO)
torch.cuda.empty_cache()

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
    return item, entity_spans, mention_types

def load_examples_test():
    examples = []
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
        concat_tokens = []
        counter = 0
        converted_item, entity_spans, ET_labels = convert_spans(item)
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
        labels_pairs = tuple(zip(item["labels"]["head"], item["labels"]["tail"], item["labels"]["r"]))
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
            entity_spans= [d for d in new_char_spans],
            labels = [d for d in ET_labels]
        ))
    return examples

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

torch.cuda.empty_cache()

test_examples = load_examples_test()

logging.info("Data Memory before Loading Models")
#print_gpu_utilization()

logging.info("############### LOAD MODEL ###################")
model = load_model_ET.model
model.classifier= torch.nn.Linear(in_features=model.classifier.in_features, out_features=7, bias = True)
model.config.id2label = {0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2', 3: 'LABEL_3', 4: 'LABEL_4', 5: 'LABEL_5', 6: 'LABEL_6'}

tokenizer = load_model_ET.tokenizer

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

batch_size = 16

num_predicted = 0
num_gold = 0
num_correct = 0

all_predictions = []
all_labels = []

torch.cuda.empty_cache()
logging.info("EVALUATION WILL BEGIN!!")

final_test_examples = []
for an_example in test_examples:
    for ix in range(len(an_example["entity_spans"])):
        final_test_examples.append(dict(
            text = an_example["text"],
            entity_span = an_example["entity_spans"][ix],
            label = an_example["labels"][ix]
        ))

test_examples = final_test_examples
del final_test_examples
for d in test_examples:
    d["entity_span"] = list([d["entity_span"]])

label2word = {"LABEL_0": "BLANK","LABEL_1": "ORG","LABEL_2": "LOC","LABEL_3": "TIME",
    "LABEL_4": "PER","LABEL_5": "MISC","LABEL_6": "NUM"}
first_time = 0




types_list = ["BLANK", "ORG", "LOC", "TIME", "PER", "MISC", "NUM"]
c2l = ClassLabel(num_classes = 7, names = types_list)
label_list_ids = [c2l.str2int(label) for label in types_list]



model.eval()

#print_gpu_utilization()
logging.info("Let's clear cache and send model to memory!")
torch.cuda.empty_cache()
#model.to(device)
#print_gpu_utilization()


for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_span"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]
    gold_labels_ids = [c2l.str2int(label) for label in gold_labels]

    clean_texts = []
    clean_entity_spans = []
    clean_gold_labels = []
    for i in trange(len(texts)):
        analyser_input = tokenizer(text=texts[i], entity_spans=entity_spans[i])
        if len(analyser_input.data['input_ids']) < 500:
            clean_texts.append(texts[i])
            clean_entity_spans.append(entity_spans[i])
            clean_gold_labels.append(gold_labels[i])
    texts = clean_texts
    entity_spans = clean_entity_spans
    gold_labels = clean_gold_labels
    del clean_texts, clean_entity_spans, clean_gold_labels


    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding="max_length", max_length=512).to(device)
    del batch_examples, texts, entity_spans
 #   print_gpu_utilization()
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    num_gold += len(gold_labels)
    for logits, labels in zip(outputs.logits, gold_labels):
        first_time = 1
        for index, logit in enumerate(logits):
            if logit > 0:
                if first_time == 1:
                    num_predicted += 1
                    first_time = 0
                predicted_label = model.config.id2label[index]
                predicted_label = label2word[predicted_label]
                if predicted_label in labels:
                    num_correct += 1
    torch.cuda.empty_cache()
precision = num_correct / num_predicted
recall = num_correct / num_gold
f1 = 2 * precision * recall / (precision + recall)


print(f"\n\nprecision: {precision} recall: {recall} f1: {f1}")

with open("results_ET_reDocRED_7epochs_baseline.txt", "w") as text_file:
    text_file.write(f"RESULTS \n num_correct = {num_correct}, num_gold = {num_gold}, num_predicted = {num_predicted} \n"
                    f"precision: {precision} \n Recall: {recall} \n F1: {f1}")


