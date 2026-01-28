from typing import Optional, List


from torch import nn, Tensor
from .multi_head_attention_equi import MultiheadAttention_equi
from .utils import *
from e3nn import o3

class TransformerDecoderLayer_equi_v3(nn.Module):

    def __init__(self, version, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu",):
        super().__init__()
        self.irreps_emb=o3.Irreps('{}x1e'.format(d_model))
        self.linear_src=LinearRS(self.irreps_emb, self.irreps_emb, bias=True)
        self.linear_dst=LinearRS(self.irreps_emb, self.irreps_emb, bias=True)

        self.ga=GraphAttentionMLP(irreps_emb=self.irreps_emb,
                                  irreps_node_output=self.irreps_emb,
                                  irreps_edge_attr=o3.Irreps('1x0e+1x1e'),
                                  irreps_head=o3.Irreps('{}x1e'.format(d_model//nhead)),
                                  mul_alpha=64,
                                  fc_neurons=[64,32,32],
                                  num_heads=nhead,
                                  )



        self.norm1 = VNLayerNorm(d_model)
        self.norm2 = VNLayerNorm(d_model)

        self.ff=VNFeedForward(d_model,dim_feedforward)


    def forward(self, tgt, memory,
                edge_sh,edge_scalars
                ):
        feat_dim=memory.shape[2]
        neighbor_num=memory.shape[1]
        tgt=self.linear_dst(tgt)
        memory=self.linear_src(memory.reshape(-1,feat_dim)).reshape(-1,neighbor_num,feat_dim)

        message=tgt.unsqueeze(1)+memory
        message=message.reshape(-1,feat_dim)

        tgt2=self.ga(message,edge_sh,edge_scalars,neighbor_num)

        tgt2=tgt2.reshape(tgt.shape[0],-1,3)
        tgt=tgt.reshape(tgt.shape[0],-1,3)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.ff(tgt)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt=tgt.reshape(tgt.shape[0],-1)
        return tgt