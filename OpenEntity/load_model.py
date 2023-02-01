import json
import logging
import os
import math
import copy
import numpy
import os.path
from dataclasses import dataclass, field
from tqdm import trange
# from turtle import forward
# from pandas import concat
import torch
import torch.nn.functional as funct
import time
from torch import nn
from torch.nn import functional as F
from transformers import LukeModel, LukeTokenizer, Trainer, TrainingArguments, TextDataset, \
    HfArgumentParser
from transformers import LukeForEntityClassification
from configurations.my_configuration import LukeConfig
from transformers import DataCollatorWithPadding

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# new self attention mechanism
class myLukeSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_heads = config.num_attention_heads
        self.head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_heads * self.head_size
        self.embed_dim = config.hidden_size
        self.use_entity_aware_attention = config.use_entity_aware_attention

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if self.use_entity_aware_attention: # editar no linear.py e mudar torch.empty por torch.zeros, no __init__
            self.w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.e2e_query = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.layer_id = layer_id

        attention_window = config.attention_window[self.layer_id]
        assert (
                attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
                attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    # LUKE
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        word_size = word_hidden_states.size(1)
        batch_size = word_hidden_states.size(0)

        # attention mask do Longformer.
        attention_mask_long = attention_mask.squeeze(dim=2).squeeze(dim=1)
        attention_mask_long_words = attention_mask_long[:,:word_size]
        is_index_masked = attention_mask_long_words < 0
        is_index_global_attn = attention_mask_long_words > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        if entity_hidden_states is None:
            concat_hidden_states = word_hidden_states
        else:
            concat_hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)

        key_layer = self.transpose_for_scores(self.key(concat_hidden_states))
        value_layer = self.transpose_for_scores(self.value(concat_hidden_states))

        # Onde são feitos todos os cálculos da attention com sliding window attention.
        if self.use_entity_aware_attention and entity_hidden_states is not None:
            # LUKE ATT SCORES
            w2w_query_layer = self.transpose_for_scores(self.query(word_hidden_states))
            w2e_query_layer = self.transpose_for_scores(self.w2e_query(word_hidden_states))
            e2w_query_layer = self.transpose_for_scores(self.e2w_query(entity_hidden_states))
            e2e_query_layer = self.transpose_for_scores(self.e2e_query(entity_hidden_states))

            w2w_key_layer = key_layer[:, :, :word_size, :]
            e2w_key_layer = key_layer[:, :, :word_size, :]
            w2e_key_layer = key_layer[:, :, word_size:, :]
            e2e_key_layer = key_layer[:, :, word_size:, :]
            # LUKE ET shape - W2W terá os mesmos inputs mas o método matmul será diferente
            w2w_attention_scores = self._long_luke_matmul(
                w2w_query_layer,
                w2w_key_layer,
                self.one_sided_attn_window_size,
                word_size,
                batch_size
            )

            # values to pad for attention probs
            remove_from_windowed_attention_mask = (attention_mask_long_words != 0)[:, :, None, None]

            # cast to fp32/fp16 then replace 1's with -inf
            float_mask = remove_from_windowed_attention_mask.type_as(w2w_query_layer).masked_fill(
                remove_from_windowed_attention_mask, -10000.0
            )
            # diagonal mask with zeros everywhere and -inf inplace of padding
            diagonal_mask = self._sliding_chunks_query_key_matmul(
                float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size
            )

            # pad local attention probs
            w2w_attention_scores += diagonal_mask
            assert list(w2w_attention_scores.size()) == [
                batch_size,
                word_size,
                self.num_heads,
                self.one_sided_attn_window_size * 2 + 1,
            ], f"local_attn_probs should be of size ({batch_size}, {word_size}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {w2w_attention_scores.size()}"

            # In Implementation A, I convert here
            start = time.time()
            w2w_attention_scores = self._conversion(w2w_attention_scores, self.one_sided_attn_window_size)
            end = time.time()
            conversion_time = end-start
            # Neste ponto w2w tem que ser (batch, 16, 4096, 4096)
            w2e_attention_scores = torch.matmul(w2e_query_layer, w2e_key_layer.transpose(-1, -2))
            e2w_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))
            e2e_attention_scores = torch.matmul(e2e_query_layer, e2e_key_layer.transpose(-1, -2))

            # combine attention scores to create the final attention score matrix
            word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
            entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
            attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        else:  # Se não há entity aware attention faço assim ou faço longformer completo?????
            query_layer = self.transpose_for_scores(self.query(concat_hidden_states))
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores /math.sqrt(self.head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # delete attn scores
        del w2w_attention_scores, w2e_attention_scores, e2w_attention_scores, e2e_attention_scores, entity_attention_scores, word_attention_scores, attention_scores

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        output_word_hidden_states = context_layer[:, :word_size, :]
        if entity_hidden_states is None:
            output_entity_hidden_states = None
        else:
            output_entity_hidden_states = context_layer[:, word_size:, :]

        if output_attentions:
            outputs = (output_word_hidden_states, output_entity_hidden_states, attention_probs)
        else:
            outputs = (output_word_hidden_states, output_entity_hidden_states)
        del context_layer
        return outputs

    def _conversion(
            self, attn_probs: torch.Tensor, window_overlap: int
    ):
        """Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors.
           Returned tensor will be of the same shape as `attn_probs`"""
        attn_probs=attn_probs.transpose(1,2)
        batch_size = attn_probs.size(0)
        num_heads = attn_probs.size(1)
        seq_len = attn_probs.size(2)
        hidden_dim = attn_probs.size(3)
        multiplier = seq_len // window_overlap
        attn_probs = F.pad(attn_probs, (0, multiplier * window_overlap + 1))
        attn_probs = attn_probs.view(batch_size, num_heads, -1)
        attn_probs = attn_probs[:, :, :-seq_len]
        attn_probs = attn_probs.view(batch_size, num_heads, seq_len, multiplier * window_overlap + hidden_dim)
        attn_probs = attn_probs[:, :, :, :-1]

        attn_probs = F.pad(attn_probs[:, :, :, window_overlap:], pad=(0, window_overlap),
                           value=-10000)  # Replace all Infs Nans and zeros with -10000. Manual mask.
        attn_probs = attn_probs[:, :, :, :seq_len]

        attn_probs = torch.nan_to_num(attn_probs, nan=-10000, posinf=-10000, neginf=-10000)
        attn_probs[attn_probs == 0] = -10000

        return attn_probs

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
            seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
            :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
            :, :, -(window_overlap + 1) : -1, window_overlap + 1 :
        ]

        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
            :, 0, : window_overlap - 1, 1 - window_overlap :
        ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores



    def _long_luke_matmul(
            self, query: torch.Tensor, key: torch.Tensor, window_overlap: int, word_size: int, batch_size: int
    ):
        query = query.view(batch_size, self.num_heads, word_size, self.head_size).transpose(1, 2)
        key = key.view(batch_size, self.num_heads, word_size, self.head_size).transpose(1, 2)
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
                seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()
        chunks_count = seq_len // window_overlap - 1
        """
            Não é a copia direta dos valores. POR OPERACOES MATRICIAIS
        """
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)
        # matrix multipication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x window_overlap
        chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (chunked_query, chunked_key))  # multiply
        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            chunked_attention_scores, padding=(0, 0, 0, 1)
        )
        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty(
            (batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1)
        )
        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                                :, :, :window_overlap, : window_overlap + 1
        ]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
                                                               :, -1, window_overlap:, : window_overlap + 1
        ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
                                                               :, :, -(window_overlap + 1): -1, window_overlap + 1:
        ]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
                                                                              :, 0, : window_overlap - 1,
                                                                              1 - window_overlap:
        ]
        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)

        return diagonal_attention_scores

    @staticmethod
    def _pad_by_window_overlap_except_last_row(chunked_hidden_states, batch_size, num_heads, seq_len):
        """shift every row 1 step right, converting columns into diagonals"""
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = F.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
            :, :, :-window_overlap
        ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        chunked_hidden_states = F.pad(chunked_hidden_states, pad = (0,seq_len-chunked_hidden_states.shape[3],0,0))
        chunked_hidden_states[:,0,:,:] = F.pad(chunked_hidden_states[:,0,:,window_overlap:], pad = (0, window_overlap), value=0  )
        chunked_hidden_states=chunked_hidden_states.view(batch_size, num_heads, seq_len, seq_len) # batch x num_heads x seq_len x seq_len

        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""

        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            hidden_states.size(1) // (window_overlap * 2),
            window_overlap * 2,
            hidden_states.size(2),
        )

        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2

        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)

    def _mask_invalid_locations(self, input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float("inf"))  # `== 1` converts to bool or uint8

    def _concat_with_global_key_attn_probs(
        self,
        key_vectors,
        query_vectors,
        max_num_global_attn_indices,
        is_index_global_attn_nonzero,
        is_local_index_global_attn_nonzero,
        is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (query_vectors, key_vectors_only_global))

        attn_probs_from_global_key[
            is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]
        ] = -10000.0

        return attn_probs_from_global_key

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = F.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )




class LukeLongSelfAttention(myLukeSelfAttention):
    def forward(
            self,
            word_hidden_states,
            entity_hidden_states,
            attention_mask=None,
            head_mask=None,
            output_attentions=False,
    ):
        return super().forward(
            word_hidden_states,
            entity_hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions
        )


class ModifiedLuke(LukeForEntityClassification):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.luke.encoder.layer):
            # replace the `LukeSelfAtt` object with `MySelfAtt`
            layer.attention.self = LukeLongSelfAttention(config, layer_id=i)
        #self.luke.entity_embeddings.position_embeddings = nn.Embedding(4112, self.config.hidden_size) # provisorio


# Luke model with some different classes
def create_new_model(save_model_to, attention_window, max_pos):
    model = LukeForEntityClassification.from_pretrained(
        "studio-ousia/luke-large-finetuned-open-entity"
    )
    tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
    model.config = LukeConfig()

    tokenizer.model_max_length = max_pos
    tokenizer.init_kwargs['model_max_length'] = max_pos
    current_max_pos, embed_size = model.luke.embeddings.position_embeddings.weight.shape
    max_pos += 2  # NOTE: Max_pos_embeddings is 4096 + 2
    model.config.max_position_embeddings = max_pos
    assert max_pos > current_max_pos
    # allocate a larger position embedding matrix
    new_pos_embed = model.luke.embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    new_ent_pos_embed = model.luke.entity_embeddings.position_embeddings.weight.new_empty(max_pos, embed_size)
    # copy position embeddings over and over to initialize the new position embeddings

    k = 0
    step = current_max_pos -2
    while k < max_pos - 3:
        new_pos_embed[k:(k + step)] = model.luke.embeddings.position_embeddings.weight[:-2] # anterior 2:
        new_ent_pos_embed[k:(k + step)] = model.luke.entity_embeddings.position_embeddings.weight[:-2] # anterior 2:
        k += step
    new_pos_embed[-2:] = model.luke.embeddings.position_embeddings.weight[-2:]
    new_ent_pos_embed[-2:] = model.luke.entity_embeddings.position_embeddings.weight[-2:]
    
    model.luke.embeddings.position_embeddings = nn.Embedding(max_pos, model.config.hidden_size) # padding_idx = 1
    model.luke.entity_embeddings.position_embeddings = nn.Embedding(max_pos, model.config.hidden_size)
    model.luke.embeddings.position_embeddings.weight.data = new_pos_embed
    model.luke.entity_embeddings.position_embeddings.weight.data = new_ent_pos_embed

    model.config.attention_window = [attention_window] * model.config.num_hidden_layers
    for i, layer in enumerate(model.luke.encoder.layer):
        # aqui inserir a self attention que quero
        my_self_attn = myLukeSelfAttention(model.config, layer_id=i)
        my_self_attn.query = layer.attention.self.query
        my_self_attn.key = layer.attention.self.key
        my_self_attn.value = layer.attention.self.value
        my_self_attn.w2e_query = copy.deepcopy(layer.attention.self.w2e_query)
        my_self_attn.e2w_query = copy.deepcopy(layer.attention.self.e2w_query)
        my_self_attn.e2e_query = copy.deepcopy(layer.attention.self.e2e_query)

        my_self_attn.query_global = copy.deepcopy(layer.attention.self.query)
        my_self_attn.key_global = copy.deepcopy(layer.attention.self.key)
        my_self_attn.value_global = copy.deepcopy(layer.attention.self.value)

        layer.attention.self = my_self_attn
    logger.info(f"Saving model to {save_model_to}")
    tokenizer.save_pretrained(save_model_to)
    model.save_pretrained(save_model_to, save_config=True)
    return model, tokenizer



@dataclass
class ModelArgs:
    attention_window: int = field(default=512, metadata={"help": "Size of attention window"})
#    max_pos: int = field(default=4096, metadata={"help": "Maximum position"})
    max_pos: int = field(default=2048, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))

training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', 'tmp',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    '--max_steps', '3000',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '5.0',
    '--per_device_eval_batch_size', '8',
    '--per_device_train_batch_size', '2',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '32',
    # '--evaluate_during_training',
    # '--do_train',
    '--do_eval',
])

model_path = f'{training_args.output_dir}/LongLuke_OpenEntity_new_attention'
if not os.path.exists(model_path):
    os.makedirs(model_path)

logger.info('Converting luke-base into luke-Modified')
model, tokenizer = create_new_model(save_model_to=model_path, attention_window=model_args.attention_window, max_pos=model_args.max_pos)

#model = ModifiedLuke.from_pretrained(model_path)
#tokenizer = LukeTokenizer.from_pretrained(model_path)

# sets model to enter evaluation mode
max_pos = model_args.max_pos
model.eval()
