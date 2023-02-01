from tqdm import trange 
import json
import torch
import torch.cuda
import datasets
from transformers import LukeTokenizer, LukeForEntityClassification, LukeConfig
from transformers import Trainer, TrainingArguments, AdamW, DataCollatorWithPadding
from torch.utils.data import DataLoader
import numpy as np
from datasets import load_metric

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

class FIGERdataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
       # self.entity_spans = entity_spans
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)

batch_size=128
#train_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/data/FIGER/OG/train.json")
train_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/ET/small_train.json")
texts = [example["text"] for example in train_examples]
print(texts)
entity_spans = [example["entity_spans"] for example in train_examples]
print(entity_spans)
gold_labels = [example["label"] for example in train_examples]
print(gold_labels)
########################## LOAD TOKENIZER #################################
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
inputs = tokenizer(texts, entity_spans=entity_spans)

train_dataset = FIGERdataset(inputs, gold_labels)
########################## LOAD DATASETS ##################################
#train_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/data/FIGER/OG/train.json")
#processed_dev = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/data/FIGER/OG/dev.json")

#data_collator = DataCollatorWithPadding(tokenizer = tokenizer
############## LOAD MODEL WITH NEW CONFIG AND NUM_LABELS=113 ##############
my_config = LukeConfig.from_json_file("/mnt/shared/home/jose.luis.malaquias.ext/ET/configurations/FIGERconfig.json")
num_labels=113
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features = num_labels, bias = True)
model.config = my_config
model.num_labels=num_labels

########################## Choose GPU ########################
# set the GPU device to use
cuda_device=0
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
model = model.to(device)


###### TRAINING MODE############3
model.train()





metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references = labels)

training_args = TrainingArguments(
    output_dir="./tmp_trainer",
    num_train_epochs=3,
    evaluation_strategy = "epoch",
    do_train= True,
    do_eval=False,
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = FIGERdataset,
    compute_metrics = compute_metrics,
)
trainer.train()
trainer.save_model()

