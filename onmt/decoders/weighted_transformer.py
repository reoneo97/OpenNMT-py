import torch
import torch.nn as nn

from onmt.decoders.decoder import DecoderBase
from onmt.modules import WeightedAttention, MultiHeadedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask
from onmt.decoders.transformer import TransformerDecoderLayerBase, TransformerDecoderBase


class WeightedTransformerDecoderLayer(TransformerDecoderLayerBase):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(WeightedTransformerDecoderLayer, self).__init__(
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type,
            max_relative_positions,
            aan_useffn,
            full_context_alignment,
            alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
        )
        self.context_attn = WeightedAttention(
            heads, d_model, dropout=attention_dropout
        )
        self.alpha_w = nn.Parameter(torch.rand(heads))
        self.softmax = nn.Softmax(0)

        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def update_dropout(self, dropout, attention_dropout):
        super(WeightedTransformerDecoderLayer, self).update_dropout(
            dropout, attention_dropout
        )
        self.context_attn.update_dropout(attention_dropout)

    def _forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=None,
        step=None,
        future=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None

        if inputs.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)

        inputs_norm = self.layer_norm_1(inputs)

        query, _ = self._forward_self_attn(
            inputs_norm, dec_mask, layer_cache, step
        )

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            attn_type="context",
        )

        mid = self.drop(mid)
        ffn_out = self.feed_forward(mid)

        alpha = self.softmax(self.alpha_w)

        alpha_out = torch.mul(ffn_out.transpose(1, 3).contiguous(), alpha)
        alpha_out = alpha_out.transpose(1, 3).contiguous()
        output = alpha_out.sum(dim=1)

        return output, attns

class WeightedTransformerDecoder(TransformerDecoder):
    """The Transformer decoder from "Weighted Transformer Network for Machine 
    Translation".
    :cite:`https://arxiv.org/pdf/1711.02132.pdf`
    #TODO: Add Mermaid Graph for Weighted Transformer
    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (onmt.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        pos_ffn_activation_fn=ActivationFunction.relu,
    ):
        super(WeightedTransformerDecoder, self).__init__(
            num_layers,d_model,heads,d_ff,copy_attn,
            self_attn_type,dropout,attention_dropout,
            embeddings,max_relative_positions,aan_useffn,
            full_context_alignment,alignment_layer,
            alignment_heads, pos_ffn_activation_fn

        )

        self.transformer_layers = nn.ModuleList(
            [
                WeightedTransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                )
                for i in range(num_layers)
            ]
        )

