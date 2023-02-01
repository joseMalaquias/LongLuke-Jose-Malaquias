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


class LukeConfig(PretrainedConfig):
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
        num_labels = 9,
        id2label = {0: 'entity', 1: 'event', 2: 'group', 3: 'location', 4: 'object', 5: 'organization', 6: 'person',
            7: 'place', 8: 'time'},
        label2id = {'entity': 0, 'event': 1, 'group': 2, 'location': 3, 'object': 4, 'organization': 5, 'person': 6, 'place': 7, 'time': 8},


        attention_window=[512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512],
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
        self.id2label = id2label
        self.label2id = label2id
