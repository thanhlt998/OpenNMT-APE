from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertConfig
from transformers.models.roberta import RobertaModel, RobertaConfig
import torch.nn as nn
import torch
from torch import Tensor, device
from typing import *

from onmt.encoders.encoder import EncoderBase


class MyEncoderRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, bert_embeddings):
        super(MyEncoderRobertaEmbeddings, self).__init__()
        self.word_lut = bert_embeddings.word_embeddings
        self.position_embeddings = bert_embeddings.position_embeddings
        self.token_type_embeddings = bert_embeddings.token_type_embeddings

        self.padding_idx = bert_embeddings.padding_idx
        self.LayerNorm = bert_embeddings.LayerNorm
        self.dropout = bert_embeddings.dropout

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length,
            dtype=torch.long,
            device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # The point of this code block is to reset the position ids
        # for sentence B (token_type_id=1)
        token_type = (token_type_ids > 0)
        position_aux = \
            ((token_type.cumsum(1) == 1) & token_type).max(1)[1].unsqueeze(1)
        position_aux = position_aux * token_type_ids.clone()
        position_ids = position_ids - position_aux

        words_embeddings = self.word_lut(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = \
            words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RobertaEncoder(EncoderBase):
    """
    Returns:
        (`FloatTensor`, `FloatTensor`):

        * embeddings `[src_len x batch_size x model_dim]`
        * memory_bank `[src_len x batch_size x model_dim]`
    """

    def __init__(self, vocab_size, pad_idx):
        super(RobertaEncoder, self).__init__()
        self.config = RobertaConfig(vocab_size=vocab_size, pad_token_id=pad_idx)
        bert = RobertaModel(self.config)
        self.embeddings = MyEncoderRobertaEmbeddings(bert.embeddings)
        self.encoder = bert.encoder

        self.pad_idx = pad_idx

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            embeddings.word_lut.weight.size(0),
            embeddings.word_padding_idx
        )

    def forward(self, src, lengths=None, **kwargs):
        """ See :obj:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # bert receives a tensor of shape [batch_size x src_len]
        input_ids = src[:, :, 0].t()
        input_shape = input_ids.size()
        device = input_ids.device

        attention_mask = (input_ids != self.pad_idx).type(torch.int)
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            # position_ids=None,
            token_type_ids=token_type_ids,
            # inputs_embeds=None,
        )

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        last_hidden_state = encoder_outputs[0]

        return embedding_output, last_hidden_state.transpose(0, 1), lengths

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.embeddings.word_lut.weight.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_head_mask(
            self, head_mask: Optional[Tensor], num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (:obj:`torch.Tensor` with shape :obj:`[num_heads]` or :obj:`[num_hidden_layers x num_heads]`, `optional`):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (:obj:`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (:obj:`bool`, `optional, defaults to :obj:`False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            :obj:`torch.Tensor` with shape :obj:`[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or
            list with :obj:`[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def initialize_bert(self, bert_type):
        print(f'load pretrained roberta {bert_type} in roberta encoder')
        bert = RobertaModel.from_pretrained(bert_type)
        bert.resize_token_embeddings(self.embeddings.word_lut.num_embeddings)

        self.embeddings = \
            MyEncoderRobertaEmbeddings(bert.embeddings)

        self.encoder = bert.encoder
