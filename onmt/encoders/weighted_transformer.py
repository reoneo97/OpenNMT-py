import torch
import torch.nn as nn

from onmt.encoders.encoder import EncoderBase
from onmt.modules import WeightedAttention
from onmt.modules.position_ffn import PositionwiseFeedForward
from onmt.modules.position_ffn import ActivationFunction
from onmt.utils.misc import sequence_mask


class WeightedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout,
                 max_relative_positions=0,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(WeightedTransformerEncoderLayer, self).__init__()

        self.self_attn = WeightedAttention(
            heads, d_model, dropout=attention_dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout,
                                                    pos_ffn_activation_fn)

        self.alpha_w = nn.Parameter(torch.rand(heads))
        self.softmax = nn.Softmax(0)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask=None):
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, attn_type="self")
        # Context of (Batch x Head x Seq_len x Model_dim)
        # Apply to each head and add
        context = self.dropout(context)
        ffn_out = self.feed_forward(context)

        alpha = self.softmax(self.alpha_w)

        alpha_out = torch.mul(ffn_out.transpose(1, 3).contiguous(), alpha)
        alpha_out = alpha_out.transpose(1, 3).contiguous()
        out = alpha_out.sum(dim=1)
        return out

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class WeightedTransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout,
                 attention_dropout, embeddings, max_relative_positions,
                 pos_ffn_activation_fn=ActivationFunction.relu):
        super(WeightedTransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [WeightedTransformerEncoderLayer(
                d_model, heads, d_ff, dropout, attention_dropout,
                max_relative_positions=max_relative_positions,
                pos_ffn_activation_fn=pos_ffn_activation_fn)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.attention_dropout[0] if type(opt.attention_dropout)
            is list else opt.attention_dropout,
            embeddings,
            opt.max_relative_positions,
            pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
        )

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
