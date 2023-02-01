# coding=utf-8
# Configuration for LongLuke, MSc research IST LISBOA José Malaquias
""" LongLUKE - OpenEntity ET configuration"""

from transformers import PretrainedConfig
#from ...utils import logging


#logger = logging.get_logger(__name__)

LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "studio-ousia/luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/config.json",
    "studio-ousia/luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/config.json",
    "studio-ousia/luke-large-finetuned-open-entity": "https://huggingface.co/studio-ousia/luke-large-finetuned-open-entity/blob/main/config.json"
}


class LukeConfig_LongDOCRED_ET(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LukeModel`]. It is used to instantiate a LUKE
    model according to the specified arguments, defining the model architecture.
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the LUKE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`LukeModel`].
        entity_vocab_size (`int`, *optional*, defaults to 500000):
            Entity vocabulary size of the LUKE model. Defines the number of different entities that can be represented
            by the `entity_ids` passed when calling [`LukeModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        entity_emb_size (`int`, *optional*, defaults to 256):
            The number of dimensions of the entity embedding.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (`int`, *optional*, defaults to 2):
            The vocabulary size of the `token_type_ids` passed when calling [`LukeModel`].
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        use_entity_aware_attention (`bool`, defaults to `True`):
            Whether or not the model should use the entity-aware self-attention mechanism proposed in [LUKE: Deep
            Contextualized Entity Representations with Entity-aware Self-attention (Yamada et
            al.)](https://arxiv.org/abs/2010.01057).
    Examples:
    ```python
    #>>> # Initializing a LUKE configuration
    #>>> configuration = MyConfig()
    #>>> # Initializing a model from the configuration
    #>>> model = LukeModel(configuration)
    #>>> # Accessing the model configuration
    #>>> configuration = model.config
    ```"""
    model_type = "luke"

    def __init__(
        self,
        vocab_size=50267,
        entity_vocab_size=500000,
        hidden_size=1024,
        entity_emb_size=256,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=4098,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_entity_aware_attention=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        num_labels = 7,
        rel2id = {
            "P1376": 79,
            "P607": 27,
            "P136": 73,
            "P137": 63,
            "P131": 2,
            "P527": 11,
            "P1412": 38,
            "P206": 33,
            "P205": 77,
            "P449": 52,
            "P127": 34,
            "P123": 49,
            "P86": 66,
            "P840": 85,
            "P355": 72,
            "P737": 93,
            "P740": 84,
            "P190": 94,
            "P576": 71,
            "P749": 68,
            "P112": 65,
            "P118": 40,
            "P17": 1,
            "P19": 14,
            "P3373": 19,
            "P6": 42,
            "P276": 44,
            "P1001": 24,
            "P580": 62,
            "P582": 83,
            "P585": 64,
            "P463": 18,
            "P676": 87,
            "P674": 46,
            "P264": 10,
            "P108": 43,
            "P102": 17,
            "P25": 81,
            "P27": 3,
            "P26": 26,
            "P20": 37,
            "P22": 30,
            "Na": 0,
            "P807": 95,
            "P800": 51,
            "P279": 78,
            "P1336": 88,
            "P577": 5,
            "P570": 8,
            "P571": 15,
            "P178": 36,
            "P179": 55,
            "P272": 75,
            "P170": 35,
            "P171": 80,
            "P172": 76,
            "P175": 6,
            "P176": 67,
            "P39": 91,
            "P30": 21,
            "P31": 60,
            "P36": 70,
            "P37": 58,
            "P35": 54,
            "P400": 31,
            "P403": 61,
            "P361": 12,
            "P364": 74,
            "P569": 7,
            "P710": 41,
            "P1344": 32,
            "P488": 82,
            "P241": 59,
            "P162": 57,
            "P161": 9,
            "P166": 47,
            "P40": 20,
            "P1441": 23,
            "P156": 45,
            "P155": 39,
            "P150": 4,
            "P551": 90,
            "P706": 56,
            "P159": 29,
            "P495": 13,
            "P58": 53,
            "P194": 48,
            "P54": 16,
            "P57": 28,
            "P50": 22,
            "P1366": 86,
            "P1365": 92,
            "P937": 69,
            "P140": 50,
            "P69": 25,
            "P1198": 96,
            "P1056": 89},
        ner2id = {
            "BLANK": 0,
            "ORG": 1,
            "LOC": 2,
            "TIME": 3,
            "PER": 4,
            "MISC": 5,
            "NUM": 6},
        id2ner={
            0: "BLANK",
            1: "ORG",
            2: "LOC",
            3: "TIME",
            4: "PER",
            5: "MISC",
            6: "NUM"},
        types_list = [
                "BLANK",
                "ORG",
                "LOC",
                "TIME",
                "PER",
                "MISC",
                "NUM"
        ],
        relations_code_list = [
            "P6",
            "P17",
            "P19",
            "P20",
            "P22",
            "P25",
            "P26",
            "P27",
            "P30",
            "P31",
            "P35",
            "P36",
            "P37",
            "P39",
            "P40",
            "P50",
            "P54",
            "P57",
            "P58",
            "P69",
            "P86",
            "P102",
            "P108",
            "P112",
            "P118",
            "P123",
            "P127",
            "P131",
            "P136",
            "P137",
            "P140",
            "P150",
            "P155",
            "P156",
            "P159",
            "P161",
            "P162",
            "P166",
            "P170",
            "P171",
            "P172",
            "P175",
            "P176",
            "P178",
            "P179",
            "P190",
            "P194",
            "P205",
            "P206",
            "P241",
            "P264",
            "P272",
            "P276",
            "P279",
            "P355",
            "P361",
            "P364",
            "P400",
            "P403",
            "P449",
            "P463",
            "P488",
            "P495",
            "P527",
            "P551",
            "P569",
            "P570",
            "P571",
            "P576",
            "P577",
            "P580",
            "P582",
            "P585",
            "P607",
            "P674",
            "P676",
            "P706",
            "P710",
            "P737",
            "P740",
            "P749",
            "P800",
            "P807",
            "P840",
            "P937",
            "P1001",
            "P1056",
            "P1198",
            "P1336",
            "P1344",
            "P1365",
            "P1366",
            "P1376",
            "P1412",
            "P1441",
            "P3373"],
        rel2word = {
            "P6": "head of government",
            "P17": "country",
            "P19": "place of birth",
            "P20": "place of death",
            "P22": "father",
            "P25": "mother",
            "P26": "spouse",
            "P27": "country of citizenship",
            "P30": "continent",
            "P31": "instance of",
            "P35": "head of state",
            "P36": "capital",
            "P37": "official language",
            "P39": "position held",
            "P40": "child",
            "P50": "author",
            "P54": "member of sports team",
            "P57": "director",
            "P58": "screenwriter",
            "P69": "educated at",
            "P86": "composer",
            "P102": "member of political party",
            "P108": "employer",
            "P112": "founded by",
            "P118": "league",
            "P123": "publisher",
            "P127": "owned by",
            "P131": "located in the administrative territorial entity",
            "P136": "genre",
            "P137": "operator",
            "P140": "religion",
            "P150": "contains administrative territorial entity",
            "P155": "follows",
            "P156": "followed by",
            "P159": "headquarters location",
            "P161": "cast member",
            "P162": "producer",
            "P166": "award received",
            "P170": "creator",
            "P171": "parent taxon",
            "P172": "ethnic group",
            "P175": "performer",
            "P176": "manufacturer",
            "P178": "developer",
            "P179": "series",
            "P190": "sister city",
            "P194": "legislative body",
            "P205": "basin country",
            "P206": "located in or next to body of water",
            "P241": "military branch",
            "P264": "record label",
            "P272": "production company",
            "P276": "location",
            "P279": "subclass of",
            "P355": "subsidiary",
            "P361": "part of",
            "P364": "original language of work",
            "P400": "platform",
            "P403": "mouth of the watercourse",
            "P449": "original network",
            "P463": "member of",
            "P488": "chairperson",
            "P495": "country of origin",
            "P527": "has part",
            "P551": "residence",
            "P569": "date of birth",
            "P570": "date of death",
            "P571": "inception",
            "P576": "dissolved, abolished or demolished",
            "P577": "publication date",
            "P580": "start time",
            "P582": "end time",
            "P585": "point in time",
            "P607": "conflict",
            "P674": "characters",
            "P676": "lyrics by",
            "P706": "located on terrain feature",
            "P710": "participant",
            "P737": "influenced by",
            "P740": "location of formation",
            "P749": "parent organization",
            "P800": "notable work",
            "P807": "separated from",
            "P840": "narrative location",
            "P937": "work location",
            "P1001": "applies to jurisdiction",
            "P1056": "product or material produced",
            "P1198": "unemployment rate",
            "P1336": "territory claimed by",
            "P1344": "participant of",
            "P1365": "replaces",
            "P1366": "replaced by",
            "P1376": "capital of",
            "P1412": "languages spoken, written or signed",
            "P1441": "present in work",
            "P3373": "sibling"},
        attention_window=[512]*24,
        classifier_dropout= None,
        position_embedding_type= "absolute",
        sep_token_id= 2,
        use_cache= True,
        **kwargs
    ):
        """Constructs LukeConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.hidden_size = hidden_size
        self.entity_emb_size = entity_emb_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_entity_aware_attention = use_entity_aware_attention
        self.attention_window = attention_window
        self.classifier_dropout = classifier_dropout
        self.position_embedding_type = position_embedding_type
        self.sep_token_id = sep_token_id
        self.use_cache = use_cache
        self.num_labels = num_labels
        self.ner2id = ner2id
        self.rel2id = rel2id
        self.rel2word = rel2word
        self.relations_code_list = relations_code_list
        self.id2ner = id2ner
        self.types_list = types_list
