import logging


from datasets import ClassLabel
from transformers import LukeTokenizer, LukeModel, LukeForEntityPairClassification, TrainingArguments, Trainer
import torch
from tqdm import trange
# construir função que converta spans de relativos a frase para globais
import load_model_ET
import json
from pynvml import *

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


#model = LukeModel.from_pretrained("studio-ousia/luke-large")
#tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-tacred")
#model = LukeForEntityPairClassification.from_pretrained("studio-ousia/luke-large-finetuned-tacred")


model = load_model_ET.model
tokenizer = load_model_ET.tokenizer
train_examples = load_examples_train()
#eval_examples = load_examples_eval(dataset) TODOOOO
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


del batch_examples,  example, train_examples, texts, entity_spans

c2l = ClassLabel(num_classes = 7, names = model.config.types_list)
label_list_ids = [c2l.str2int(label) for label in model.config.types_list]
gold_labels_ids = [c2l.str2int(label) for label in gold_labels]

torch.cuda.empty_cache()
print_gpu_utilization()
logging.info("TOKENIZING WILL BEGIN SHROTLY!!")
#inputs = tokenizer(text=texts, entity_spans = entity_spans, padding = "max_length", max_length = 1024, return_tensors = "pt", task="entity_classification").to(device)
#torch.save(inputs, "inputs_ET_reDOCRED_3_2.pt")



inputs = torch.load("inputs_ET_reDOCRED_3_2.pt")

train_dataset = MyDataset(inputs, gold_labels_ids)

logging.info("Beginning of trainer batching")
torch.cuda.empty_cache()
print_gpu_utilization()

training_args = TrainingArguments(
    do_train = True,
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 8,
    gradient_checkpointing = True,
    fp16 = True, #??
    optim = "adafactor",
    #optim = "adam"
    output_dir = "train_reDOCRED_ET_checkpoints_3_plus_2",
    num_train_epochs = 2,
    report_to = "none",
    dataloader_pin_memory = False,
    logging_strategy =  "steps",
    logging_steps = 10000,
    learning_rate= 0.001
)
torch.cuda.empty_cache()
print_gpu_utilization()
model.train()
model.to(device)
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset
)

logging.info("TRAIN WILL BEGIN. WHAT IS IN THE CLUSTER?")
trainer.train()
trainer.save_model("model_finetuned_ET_reDocRED_3_mais_2")
print("END of TRAINING ! !")



