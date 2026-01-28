import warnings
import torch
from torch.nn import Linear
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from torch.nn import Module
from torch.nn import functional as F
import torch.nn as nn
from e3nn import o3
from eqnet.ops import attention
from eqnet.transformer.utils import *
from eqnet.utils import attention_helper


class MultiheadAttention_v6(Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(
        self,
        version,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super(MultiheadAttention_v6, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.linear_key=Linear(embed_dim, embed_dim)
        self.linear_value=Linear(embed_dim, embed_dim)
        self.linear_query=Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim, bias=True)

        self.fc_gamma = nn.Sequential(
            nn.Linear(embed_dim,num_heads*3),
            nn.ReLU(inplace=True),
            nn.Linear(num_heads*3,num_heads*3),
        )



        self.irreps_sh=o3.Irreps('1x0e+1x1e')
        self.sh = o3.SphericalHarmonics(irreps_out = self.irreps_sh, normalize = True, normalization='component')
        self.length_range=(0.001,0.005,None,None)


    def forward(
        self,
        query,  # total_q_num, c
        key,  # total_k_num, c
        value,  # total_k_num, c
        index_pair,  # total_q_num, max_memory_num
        query_batch_cnt,  # bs: query_amount of each batch
        key_batch_cnt,  # bs: key_amount of each batch.
        index_pair_batch,  # total_q_num, batch_index of each query.
        attn_mask=None,  # total_q_num, max_memory_num

        # positional encoding setting.
        relative_atten_weights=None,  # total_q_num, max_memory_num, nhead

        # crpe module.
        ctx_rpe_query=None,
        ctx_rpe_key=None,
        ctx_rpe_value=None,
        rpe_distance=None,
    ):


        rpe_length=rpe_distance.norm(dim=-1, p=2)
        rpe_cutoff_nonscalar = soft_square_cutoff_2(x=rpe_length.reshape(-1), ranges=self.length_range)
        rpe_edge_sh=self.sh(rpe_distance.reshape(-1,3))
        rpe_edge_sh = cutoff_irreps(f=rpe_edge_sh,
                                   edge_cutoff=None,
                                   cutoff_scalar=None,
                                   cutoff_nonscalar=rpe_cutoff_nonscalar,
                                   irreps=self.irreps_sh)[:,1:]
        neighbor_num=index_pair.shape[1]
        query=self.linear_query(query)
        key=self.linear_key(key)
        value=self.linear_value(value)


        batch_size=index_pair_batch.max().item()+1
        key=attention_helper.mapper_v6(index_pair,key,batch_size)
        value=attention_helper.mapper_v6(index_pair,value,batch_size)

        dot_atten=torch.einsum('phd,pnhd->pnh', query.reshape(-1,self.num_heads,self.head_dim),
                               key.reshape(-1,neighbor_num,self.num_heads,self.head_dim))

        pos_atten=self.fc_gamma(key-query[:,None,:]).reshape(-1,neighbor_num,self.num_heads,3)

        pos_atten=pos_atten*(rpe_edge_sh.reshape(-1,neighbor_num,1,3).repeat(1,1,self.num_heads,1))
        pos_atten=pos_atten.sum(-1)
        atten=dot_atten+pos_atten+relative_atten_weights
        atten=atten/self.head_dim ** 0.5
        atten=F.softmax(atten, dim=1)
        atten=F.dropout(atten, p=self.dropout, training=self.training)
        value=value.reshape(-1,neighbor_num,self.num_heads,self.head_dim)
        attn_output=torch.einsum('pnh,pnhc->phc', atten, value).reshape(-1,self.embed_dim)
        return attn_output