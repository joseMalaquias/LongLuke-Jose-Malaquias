import numpy as np
import torch.nn
from transformers import LukeTokenizer, LukeForEntityClassification, LukeConfig, TrainingArguments, Trainer, AutoTokenizer
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from tqdm import trange
from datasets import ClassLabel, load_dataset
import json
from datasets import Dataset, load_dataset
import logging
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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

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

def preprocess_data(examples, num_examples):
    labels_matrix = np.zeros((num_examples, 112))
    #fill numpy_array
    for idx, label in enumerate(examples.labels):
        for item in range(len(label)):
            labels_matrix[idx,label[item]] +=1 #posicao a adicionar é label(item)
    examples.labels = labels_matrix.tolist()
    return examples

def tokenize_function(examples):
    return tokenizer(examples["texts"], entity_spans=examples["entity_spans"], padding="max_length")
train_examples = load_examples("/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/data/FIGER/OG/OG/train.json")

print("\n\n############ Prepare finetune parameters to feed model ##################")
label_list = ['/art', '/art/film', '/astral_body', '/award', '/biology', '/body_part', '/broadcast', '/broadcast/tv_channel', '/broadcast_network', '/broadcast_program', '/building', '/building/airport', '/building/hospital', '/building/library', '/building/restaurant', '/building/sports_facility', '/building/theater', '/chemistry', '/computer', '/computer/programming_language', '/disease', '/education', '/education/department', '/education/educational_degree', '/event', '/event/attack', '/event/election', '/event/military_conflict', '/event/natural_disaster', '/event/protest', '/event/sports_event', '/finance', '/finance/currency', '/food', '/game', '/geography', '/geography/island', '/geography/mountain', '/god', '/government', '/government/government', '/government/political_party', '/government_agency', '/internet', '/internet/website', '/language', '/law', '/livingthing', '/livingthing/animal', '/location', '/location/body_of_water', '/location/bridge', '/location/cemetery', '/location/city', '/location/country', '/location/county', '/location/province', '/medicine', '/medicine/drug', '/medicine/medical_treatment', '/medicine/symptom', '/metropolitan_transit', '/metropolitan_transit/transit_line', '/military', '/music', '/news_agency', '/organization', '/organization/airline', '/organization/company', '/organization/educational_institution', '/organization/fraternity_sorority', '/organization/sports_league', '/organization/sports_team', '/park', '/people', '/people/ethnicity', '/person', '/person/actor', '/person/architect', '/person/artist', '/person/athlete', '/person/author', '/person/coach', '/person/director', '/person/doctor', '/person/engineer', '/person/monarch', '/person/musician', '/person/politician', '/person/religious_leader', '/person/soldier', '/play', '/product', '/product/airplane', '/product/car', '/product/computer', '/product/instrument', '/product/ship', '/product/spacecraft', '/product/weapon', '/rail', '/rail/railway', '/religion', '/religion/religion', '/software', '/time',  '/title', '/train', '/transit', '/transportation', '/transportation/road', '/written_work']
c2l = ClassLabel(num_classes = 112, names = label_list)
label_list_ids = [c2l.str2int(label) for label in label_list]


logging.info("Data Memory before Loading Models")
print_gpu_utilization()


logging.info("LOADING MODEL ..........")
#tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
tokenizer = AutoTokenizer.from_pretrained(
    "studio-ousia/luke-large-finetuned-open-entity",
    task = "entity_classification"
)

my_config = LukeConfig.from_json_file("./configurations/FIGERconfig.json")
num_labels = 112
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
#model = LukeForEntityClassification.from_pretrained("/Users/jose.luis.malaquias.ext/Documents/training/download/luke-large-finetuned-open-entity")
model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features= num_labels) #no bias?
model.config = my_config
model.num_labels = num_labels
model.luke.config = model.config
model.train()


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


logging.info("Extract from raw text to each field")
batch_size = 8 #4
# extract content of file to each one of the fields
for batch_start_idx in trange(0, len(train_examples), len(train_examples)):
    batch_examples = train_examples[batch_start_idx:batch_start_idx+len(train_examples)]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]


# Convert Labels to IDs
gold_labels_ids = [c2l.str2int(label) for label in gold_labels]
logging.info("TOKENIZING, IT MAY TAKE A WHILE......")
inputs = tokenizer(texts, entity_spans=entity_spans, padding=True, truncation=True, return_tensors="pt")
torch.save(inputs, "inputs.pt")
#inputs = torch.load("inputs.pt")
train_dataset = MyDataset(inputs, gold_labels_ids)
train_dataset_ready = preprocess_data(train_dataset, len(texts))

logging.info("Data Memory before training")
print_gpu_utilization()


training_args = TrainingArguments(
    do_train=True,
    per_device_train_batch_size=batch_size,
    output_dir="./test_trainer",
    metric_for_best_model="f1",
    gradient_accumulation_steps=4,
    gradient_checkpointing = True,
    fp16 = True,
    optim = "adafactor"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_ready,
)

torch.cuda.memory_summary(device=None, abbreviated=False)

logging.info("TRAIN WILL BEGIN!!......")
result = trainer.train()
print_summary(result)
logging.info("SAVING MODEL")
model.save_pretrained(save_directory="./saved_model_FIGER")
# to load this saved model we use:
# model = .from_pretrained("path/to/awesome-name-you-picked").
print("END of TRAINING ! !")

"""
outputs = model(**inputs)
logits = outputs.logits
logits = logits.detach().numpy()
predicted_class_id=logits[0].argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_id])
"""






