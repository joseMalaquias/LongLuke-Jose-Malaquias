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


test_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/data/FIGER/OG/OG/test.json")


logging.info("Data Memory before Loading Models")
print_gpu_utilization()

logging.info("############### LOAD MODEL ###################")
model = LukeForEntityClassification.from_pretrained("ET_for_FIGER_model")
model.eval()
# Load the tokenizer
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

logging.info("Data Memory after Loading Models")
print_gpu_utilization()

logging.info("CHOOSE GPU")
########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0  # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
model = model.to(device)

batch_size = 128

num_predicted = 0
num_gold = 0
num_correct = 0

all_predictions = []
all_labels = []

logging.info("EVALUATION WILL BEGIN!!")

for batch_start_idx in trange(0, len(test_examples), batch_size):
    batch_examples = test_examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]

    inputs = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
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

precision = num_correct / num_predicted
recall = num_correct / num_gold
f1 = 2 * precision * recall / (precision + recall)

print(f"\n\nprecision: {precision} recall: {recall} f1: {f1}")

"""
text = "Octopus live in the Atlantic and are captured by fishermen."
entity_spans = [(0, 7)]  # character-based entity span corresponding to "Beyoncé"

inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
inputs.to("cuda")
outputs = model(**inputs)

predicted_indices = [index for index, logit in enumerate(outputs.logits[0]) if logit > 0]
print("Predicted entity type for Entity:", [model.config.id2label[index] for index in predicted_indices])

entity_spans = [(20, 28)]  # character-based entity span corresponding to "Beyoncé"
inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
inputs.to("cuda")
outputs = model(**inputs)

predicted_indices = [index for index, logit in enumerate(outputs.logits[0]) if logit > 0]
print("Predicted entity type for Entity2:", [model.config.id2label[index] for index in predicted_indices])



entity_spans = [(49, 57)]  # character-based entity span corresponding to "Beyoncé"
inputs = tokenizer(text, entity_spans=entity_spans, return_tensors="pt")
inputs.to("cuda")
outputs = model(**inputs)

predicted_indices = [index for index, logit in enumerate(outputs.logits[0]) if logit > 0]
print("Predicted entity type for Entity3:", [model.config.id2label[index] for index in predicted_indices])

"""
