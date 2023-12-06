import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import List, Optional

from model.position_encoding import TimeEmbeddingLearned, TimeEmbeddingSine, PositionalEmbeddingLearned, \
    PositionalEncoding2D, PositionalEncoding1D


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        video_max_len=8*4,
        obj_max_num=10, # TODO pass from the step before
        no_tsa=False,
        return_weights=False,
        learn_time_embed=True,
        rd_init_tsa=False,
        no_time_embed=False,
    ):
        """
        :param d_model: transformer embedding dimension
        :param nhead: transformer number of heads
        :param num_decoder_layers: transformer decoder number of layers
        :param dim_feedforward: transformer dimension of feedforward
        :param dropout: transformer dropout
        :param activation: transformer activation
        :param return_intermediate_dec: whether to return intermediate outputs of the decoder
        :param pass_pos_and_query: if True tgt is initialized to 0 and position is added at every layer
        :param video_max_len: maximum number of frames in the model
        :param no_tsa: whether to use temporal self-attention
        :param return_weights: whether to return attention weights
        :param learn_time_embed: whether to learn time encodings
        :param rd_init_tsa: whether to randomly initialize temporal self-attention weights
        :param no_time_embed: whether to use time encodings
        """
        super().__init__()

        # self.pass_pos_and_query = pass_pos_and_query

        decoder_layer = TransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            no_tsa=no_tsa,
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            return_weights=return_weights,
        )

        self._reset_parameters()

        self.return_weights = return_weights

        self.learn_time_embed = learn_time_embed
        self.use_time_embed = not no_time_embed
        # if self.use_time_embed:
        #     if learn_time_embed:
        #         self.time_embed = TimeEmbeddingLearned(video_max_len, d_model)
        #     else:
        #         self.time_embed = TimeEmbeddingSine(video_max_len, d_model)
        self.obj_embed = PositionalEncoding1D(d_model)

        self.rd_init_tsa = rd_init_tsa
        self._reset_temporal_parameters()

        self.expander_dropout = 0.1

        self.d_model = d_model
        self.nhead = nhead
        self.video_max_len = video_max_len

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _reset_temporal_parameters(self):
        for n, p in self.named_parameters():
            if self.rd_init_tsa and "decoder" in n and "self_attn" in n:
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(
        self,
        query_encoding,
        vt_encoding,
        query_mask=None,
        vt_mask=None,
        video_max_len=32,
    ):

        # time-space-text attention
        n_queries, bt, d_model = query_encoding.size()
        bsize = bt//video_max_len
        obj_pos = self.obj_embed(query_encoding.transpose(0, 1)) # n_queriesx(bsize*t)xd_model -> (bsize*t)xn_queriesxd_model
        # time_pos = self.time_embed(video_max_len).repeat(bsize, n_queries,1).transpose(0,1)

        print("** SPACE TIME DECODER **")
        print("query_encoding shape:", query_encoding.shape)
        print("time_pos shape:", obj_pos.shape)
        hs = self.decoder(
            query_encoding=query_encoding,  # n_queriesx(bsize*t)xd_model
            vt_encoding=vt_encoding,  # 1x(bsize*t)xd_model
            query_mask=query_mask,  # (bsize*t)xn_queries
            vt_mask=vt_mask, # (bsizext)x1
            query_pos=obj_pos.transpose(0, 1), # n_queriesx(bsize*t)xd_model
        )  # n_layersxn_queriesx(bsize*t)xdmodel
        if self.return_weights:
            hs, weights, cross_weights = hs

        if not self.return_weights:
            return hs.transpose(1, 2)
        else:
            return hs.transpose(1, 2), weights, cross_weights



class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        return_weights=False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights

    def forward(
        self,
        query_encoding,
        vt_encoding,
        query_mask: Optional[Tensor] = None,
        vt_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = query_encoding

        intermediate = []
        intermediate_weights = []
        intermediate_cross_weights = []

        for i_layer, layer in enumerate(self.layers):
            output, weights, cross_weights = layer(
                output,
                vt_encoding,
                query_mask=query_mask,
                vt_mask=vt_mask,
                query_pos=query_pos
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.return_weights:
                    intermediate_weights.append(weights)
                    intermediate_cross_weights.append(cross_weights)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if not self.return_weights:
                return torch.stack(intermediate)
            else:
                return (
                    torch.stack(intermediate),
                    torch.stack(intermediate_weights),
                    torch.stack(intermediate_cross_weights),
                )

        if not self.return_weights:
            return output
        else:
            return output, weights, cross_weights



class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        no_tsa=False,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.no_tsa = no_tsa

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        query,
        vt_encoding,
        query_mask: Optional[Tensor] = None,
        vt_mask: Optional[Tensor] = None,
        att_mask: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):

        q = self.with_pos_embed(query, query_pos)
        k = v = query
        print("query shape:", query.shape)
        print("vt_encoding shape:", vt_encoding.shape)
        print("vt_mask shape:", vt_mask.shape)
        # Temporal Self attention
        tgt2, weights = self.self_attn(
            q,
            k,
            value=v,
            attn_mask=query_mask
        ) # bxtxf

        tgt = query + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Time Aligned Cross attention
        t, b, _ = tgt.shape
        n_tokens, bs, f = vt_encoding.shape # bs = bsize*t
        tgt_cross = (
            tgt.reshape(bs, -1, f).transpose(0, 1)
        )  # bxtxf -> (b*t)x1xf -> 1x(b*t)xf
        query_pos_cross = (
            query_pos.reshape(bs, -1, f).transpose(0, 1)
        )  # bxtxf -> (b*t)x1xf -> 1x(b*t)xf

        tgt2, cross_weights = self.cross_attn_image(
            query=self.with_pos_embed(tgt_cross, query_pos_cross),
            key=vt_encoding,
            value=vt_encoding,
            # key_padding_mask=vt_mask, # TODO expecting key_padding_mask shape of (2048, 1), but got torch.Size([64, 8])
        ) # 1x(b*t)xf

        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, weights, cross_weights



def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.embd_dim,
        dropout=args.dropout,
        nhead=args.n_heads,
        dim_feedforward=args.ff_dim,
        num_decoder_layers=args.loc_dec_layers,
        return_intermediate_dec=True,
        # pass_pos_and_query=args.pass_pos_and_query,
        video_max_len=args.video_max_len,
        no_tsa=args.no_tsa,
        return_weights=args.guided_attn,
        learn_time_embed=args.learn_time_embed,
        rd_init_tsa=args.rd_init_tsa,
    )
