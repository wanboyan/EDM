from typing import Optional, List


from torch import nn, Tensor
from .multi_head_attention_equi import MultiheadAttention_equi
from .utils import *


class TransformerDecoderLayer_equi(nn.Module):

    def __init__(self, version, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu",
                 ctx_rpe_query=None, ctx_rpe_key=None, ctx_rpe_value=None):
        super().__init__()

        self.multihead_attn = MultiheadAttention_equi(version, d_model, nhead, dropout=dropout)



        self.norm1 = VNLayerNorm(d_model)
        self.norm2 = VNLayerNorm(d_model)

        self.ff=VNFeedForward(d_model,dim_feedforward)


        # define positional encoding in transformer decoder layer.
        # partial function for initialization.
        self.ctx_rpe_query = ctx_rpe_query() if ctx_rpe_query is not None else None
        self.ctx_rpe_key = ctx_rpe_key() if ctx_rpe_key is not None else None
        self.ctx_rpe_value = ctx_rpe_value() if ctx_rpe_value is not None else None

    def forward(self, tgt, memory,
                index_pair,
                query_batch_cnt,
                key_batch_cnt,
                index_pair_batch,
                attn_mask: Optional[Tensor] = None,

                relative_atten_weights=None,
                rpe_distance=None,):
        tgt2 = self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            index_pair=index_pair,
            query_batch_cnt=query_batch_cnt,
            key_batch_cnt=key_batch_cnt,
            index_pair_batch=index_pair_batch,
            attn_mask=attn_mask,

            relative_atten_weights=relative_atten_weights,

            ctx_rpe_query=self.ctx_rpe_query,
            ctx_rpe_key=self.ctx_rpe_key,
            ctx_rpe_value=self.ctx_rpe_value,
            rpe_distance=rpe_distance)
        tgt=tgt.reshape(tgt.shape[0],-1,3)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.ff(tgt)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt=tgt.reshape(tgt.shape[0],-1)
        return tgt