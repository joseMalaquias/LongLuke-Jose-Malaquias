import logging


from datasets import ClassLabel
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification, TrainingArguments, Trainer
import torch
from tqdm import trange
# construir função que converta spans de relativos a frase para globais
import load_model_ET
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
    return item, entity_spans, mention_types





def load_examples_train():
    examples = []
    f = open('/mnt/shared/home/jose.luis.malaquias.ext/reDOCRED/Revisiting_DOCRED/train_revised.json')
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

torch.cuda.empty_cache()
max_value = 0



model = load_model_ET.model
num_labels = 7
model.num_labels = num_labels
model.classifier = torch.nn.Linear(in_features=model.classifier.in_features, out_features=num_labels)
tokenizer = load_model_ET.tokenizer
train_examples = load_examples_train()
maximum = 0
max_seq = 0



for i in range(len(train_examples)):
    this_value = len(train_examples[i]["labels"])
    this_seq = len(train_examples[i]["text"])
    if maximum <= this_value:
        maximum = this_value
        i_max_pairs = i
    if max_seq <= this_seq:
        max_seq = this_seq
        i_max_seq = i
print(max_seq)


new_list = []
for example in train_examples:
    for ix_rel in range(len(example["labels"])):
        new_entry = example.copy()
        new_entry["labels"] = new_entry["labels"][ix_rel]
        new_entry["entity_spans"] = new_entry["entity_spans"][ix_rel]
        new_list.append(new_entry)
        del new_entry
del train_examples
train_examples = []
train_examples = new_list
del new_list

torch.cuda.empty_cache()
########################## Choose GPU ########################
# set the GPU device to use
cuda_device= 0 # mudar para 0 para dar o cuda
if cuda_device < 0:
    device = torch.device("cpu")
else:
    device = torch.device(f"cuda:{cuda_device}")
#model = model.to(device)
#model.train()

batch_size = 4
# Convert to inputs
for batch_start_idx in trange(0, len(train_examples), len(train_examples)):
    batch_examples = train_examples[batch_start_idx:batch_start_idx+len(train_examples)]
    texts = [example["text"] for example in batch_examples]
    entity_spans = [example["entity_spans"] for example in batch_examples]
    gold_labels = [example["labels"] for example in batch_examples]

#list_entity_spans = [list(d) for d in entity_spans]

entity_spans = [[d] for d in entity_spans]

torch.cuda.empty_cache()
#print_gpu_utilization()
logging.info("TOKENIZING WILL BEGIN SHROTLY!!")
clean_texts = []
clean_entity_spans = []
clean_gold_labels = []
for i in trange(len(texts)):
    analyser_input = tokenizer(text=texts[i], entity_spans=entity_spans[i])
    if len(analyser_input.data['input_ids']) < 510:
        clean_texts.append(texts[i])
        clean_entity_spans.append(entity_spans[i])
        clean_gold_labels.append(gold_labels[i])
texts = clean_texts
entity_spans = clean_entity_spans
gold_labels = clean_gold_labels
del clean_texts, clean_entity_spans, clean_gold_labels

types_list = ["BLANK", "ORG", "LOC", "TIME", "PER", "MISC", "NUM"]
c2l = ClassLabel(num_classes = 7, names = types_list)
label_list_ids = [c2l.str2int(label) for label in types_list]
gold_labels_ids = [c2l.str2int(label) for label in gold_labels]

#del texts, entity_spans
inputs = tokenizer(text=texts, entity_spans=entity_spans, truncation=True, padding = "max_length", max_length = 512, task = "entity_classification", return_tensors = "pt").to(device)
#torch.save(inputs, "inputs_ET_baseline_3epochs.pt")

#inputs = torch.load("inputs_ET_baseline_3epochs.pt")
train_dataset = MyDataset(inputs, gold_labels_ids)

logging.info(f"{len(train_dataset.encodings.data['input_ids'])} texts and {len(train_dataset.labels)} labels corresponding!!")
logging.info("Beginning of trainer batching")
torch.cuda.empty_cache()
#print_gpu_utilization()

training_args = TrainingArguments(
    do_train = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 8,
    gradient_checkpointing = True,
    fp16 = True, #??
    optim = "adafactor",
    #optim = "adam"
    output_dir = "checkpoints_reDOCRED_ET_baseline_new",
    num_train_epochs = 4,
    report_to = "none",
    dataloader_pin_memory = False,
    logging_strategy =  "steps",
    logging_steps = 5000, #na aval aumentar x10
    learning_rate= 0.00001
)
torch.cuda.empty_cache()
#print_gpu_utilization()
model.train()
model.to(device)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset
)

logging.info("TRAIN WILL BEGIN. WHAT IS IN THE CLUSTER?")
trainer.train()
trainer.save_model("model_finetuned_baseline_new")
print("END of TRAINING ! !")
