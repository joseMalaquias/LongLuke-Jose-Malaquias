import pickle
import dill
import os
from tqdm import trange, tqdm 
import json
import torch
import torch.cuda
import datasets
from argparse import Namespace
from transformers import LukeTokenizer, LukeForEntityClassification, LukeConfig
from transformers import Trainer, TrainingArguments, AdamW, DataCollatorWithPadding
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from datasets import load_metric

################ UTILS #################
ENTITY_TOKEN = "[ENTITY]"

def load_examples_from_file(dataset_file):
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

def load_examples(tokenizer, fold = "train"):

    if fold == "train":
        #examples = processor.get_train_examples("/mnt/shared/home/jose.luis.malaquias.ext/ET")
        examples = load_examples_from_file("/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/data/FIGER/OG/train.json")
 #   print(type(examples))
 #   print(examples)
    label_list = ['/art', '/art/film', '/astral_body', '/award', '/biology', '/body_part', '/broadcast', '/broadcast/tv_channel', '/broadcast_network', '/broadcast_program', '/building', '/building/airport', '/building/hospital', '/building/library', '/building/restaurant', '/building/sports_facility', '/building/theater', '/chemistry', '/computer', '/computer/programming_language', '/disease', '/education', '/education/department', '/education/educational_degree', '/event', '/event/attack', '/event/election', '/event/military_conflict', '/event/natural_disaster', '/event/protest', '/event/sports_event', '/finance', '/finance/currency', '/food', '/game', '/geography', '/geography/island', '/geography/mountain', '/god', '/government', '/government/government', '/government/political_party', '/government_agency', '/internet', '/internet/website', '/language', '/law', '/living_thing', '/livingthing', '/livingthing/animal', '/location', '/location/body_of_water', '/location/bridge', '/location/cemetery', '/location/city', '/location/country', '/location/county', '/location/province', '/medicine', '/medicine/drug', '/medicine/medical_treatment', '/medicine/symptom', '/metropolitan_transit', '/metropolitan_transit/transit_line', '/military', '/music', '/news_agency', '/organization', '/organization/airline', '/organization/company', '/organization/educational_institution', '/organization/fraternity_sorority', '/organization/sports_league', '/organization/sports_team', '/park', '/people', '/people/ethnicity', '/person', '/person/actor', '/person/architect', '/person/artist', '/person/athlete', '/person/author', '/person/coach', '/person/director', '/person/doctor', '/person/engineer', '/person/monarch', '/person/musician', '/person/politician', '/person/religious_leader', '/person/soldier', '/play', '/product', '/product/airplane', '/product/car', '/product/computer', '/product/instrument', '/product/ship', '/product/spacecraft', '/product/weapon', '/rail', '/rail/railway', '/religion', '/religion/religion', '/software', '/time',  '/title', '/train', '/transit', '/transportation', '/transportation/road', '/written_work']

    features = convert_examples_to_features(examples, label_list, tokenizer, 512)
    #features = read_list_feats()
    print(type(features))
    print(features)
    def collate_fn(batch):
        def create_padded_sequence(attr_name, padding_value):
            tensors = [torch.tensor(getattr(o, attr_name), dtype=torch.long) for o in batch]
            return torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=padding_value)

        return dict(
            word_ids=create_padded_sequence("word_ids", tokenizer.pad_token_id),
            word_attention_mask=create_padded_sequence("word_attention_mask", 0),
            word_segment_ids=create_padded_sequence("word_segment_ids", 0),
            entity_ids=create_padded_sequence("entity_ids", 0),
            entity_attention_mask=create_padded_sequence("entity_attention_mask", 0),
            entity_position_ids=create_padded_sequence("entity_position_ids", -1),
            entity_segment_ids=create_padded_sequence("entity_segment_ids", 0),
            labels=torch.tensor([o.labels for o in batch], dtype=torch.long),
        )

    if fold in ("dev", "test"):
        dataloader = DataLoader(features, batch_size= 32, shuffle=False, collate_fn=collate_fn)
    else:
        sampler = DistributedSampler(features)
        dataloader = DataLoader(features, sampler=sampler, batch_size=128, collate_fn=collate_fn)

    return dataloader, features

# Read list to memory
def read_list_exs():
    # for reading also binary mode is important
    with open('/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/examples/entity_typing_FIGER/exs.pkl', 'rb') as fp:
        n_list = dill.load(fp)
        return n_list
def read_list_feats():
    # for reading also binary mode is important
    with open('/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/examples/entity_typing_FIGER/featurings.pkl', 'rb') as fp:
        n_list = dill.load(fp)
        return n_list





######################### MAIN CODE ###########################
tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")

############## LOAD MODEL WITH NEW CONFIG AND NUM_LABELS=113 ##############
my_config = LukeConfig.from_json_file("/mnt/shared/home/jose.luis.malaquias.ext/ET/configurations/FIGERconfig.json")
num_labels=113
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features = num_labels, bias = True)
model.config = my_config
model.num_labels=num_labels

########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0 # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
model = model.to(device)

print("CUDA?")
print(torch.cuda.is_available())
print(f"{device}")


##################### Construir um bom dataloader ########
examples = load_examples_from_file("/mnt/shared/home/jose.luis.malaquias.ext/LukeOG/data/FIGER/OG/train.json")
label_list = ['/art', '/art/film', '/astral_body', '/award', '/biology', '/body_part', '/broadcast', '/broadcast/tv_channel', '/broadcast_network', '/broadcast_program', '/building', '/building/airport', '/building/hospital', '/building/library', '/building/restaurant', '/building/sports_facility', '/building/theater', '/chemistry', '/computer', '/computer/programming_language', '/disease', '/education', '/education/department', '/education/educational_degree', '/event', '/event/attack', '/event/election', '/event/military_conflict', '/event/natural_disaster', '/event/protest', '/event/sports_event', '/finance', '/finance/currency', '/food', '/game', '/geography', '/geography/island', '/geography/mountain', '/god', '/government', '/government/government', '/government/political_party', '/government_agency', '/internet', '/internet/website', '/language', '/law', '/living_thing', '/livingthing', '/livingthing/animal', '/location', '/location/body_of_water', '/location/bridge', '/location/cemetery', '/location/city', '/location/country', '/location/county', '/location/province', '/medicine', '/medicine/drug', '/medicine/medical_treatment', '/medicine/symptom', '/metropolitan_transit', '/metropolitan_transit/transit_line', '/military', '/music', '/news_agency', '/organization', '/organization/airline', '/organization/company', '/organization/educational_institution', '/organization/fraternity_sorority', '/organization/sports_league', '/organization/sports_team', '/park', '/people', '/people/ethnicity', '/person', '/person/actor', '/person/architect', '/person/artist', '/person/athlete', '/person/author', '/person/coach', '/person/director', '/person/doctor', '/person/engineer', '/person/monarch', '/person/musician', '/person/politician', '/person/religious_leader', '/person/soldier', '/play', '/product', '/product/airplane', '/product/car', '/product/computer', '/product/instrument', '/product/ship', '/product/spacecraft', '/product/weapon', '/rail', '/rail/railway', '/religion', '/religion/religion', '/software', '/time',  '/title', '/train', '/transit', '/transportation', '/transportation/road', '/written_work']

print("\n\n############# Print Performance Training #############")
batch_size = 32

num_predicted = 0
num_gold = 0
num_correct = 0

all_predictions = []
all_labels = []
model.train()
for batch_start_idx in trange(0, len(examples), batch_size):
    batch_examples = examples[batch_start_idx:batch_start_idx + batch_size]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["label"] for example in batch_examples]

    features = tokenizer(texts, entity_spans=entity_spans, return_tensors="pt", padding=True)
    print(type(features))

    dataloader = DataLoader(features, batch_size = batch_size)

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
        dataloader = dataloader,
        tokenizer = tokenizer
    )

    trainer.train()

#train_dataloader, features= load_examples(tokenizer, fold="train")
#num_labels = len(features[0].labels)
#print(f"Numero de Labels:{num_labels}")
#features = torch.load("/mnt/shared/home/jose.luis.malaquias.ext/ET/DataLoader/train/features/features.pt")
#examples = torch.load('/mnt/shared/home/jose.luis.malaquias.ext/ET/DataLoader/train/examples/examples.pt')
"""



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
"""
