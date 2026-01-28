from typing import Optional, List


from torch import nn, Tensor
from .multi_head_attention_equi import MultiheadAttention_equi
from .utils import *
from eqnet.utils import attention_helper
import absl.flags as flags
FLAGS = flags.FLAGS
class TransformerEncoderLayer_equi_an(nn.Module):

    def __init__(self, version, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu",):
        super().__init__()
        self.d_model=d_model
        self.to_q = nn.Linear(d_model, d_model, bias = False)
        self.to_k = nn.Linear(d_model, d_model, bias = False)
        self.to_v = nn.Linear(d_model, d_model, bias = False)
        self.linear_out=nn.Linear(d_model, d_model, bias = False)
        self.nhead=nhead
        self.dropout=dropout
        self.d_per_head=d_model//nhead

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation=F.relu


    def forward(self, tgt, index_pair,cnt,sh,dist_atten,
                ):
        if FLAGS.fea_type=='a5':
            an=5
        elif FLAGS.fea_type=='a6':
            an=6

        neighbor_num=index_pair.shape[1]
        query=self.to_q(tgt.reshape(-1,self.d_model,an).permute(0,2,1)).permute(0,2,1)
        key=self.to_k(tgt.reshape(-1,self.d_model,an).permute(0,2,1)).permute(0,2,1)
        value=self.to_q(tgt.reshape(-1,self.d_model,an).permute(0,2,1)).permute(0,2,1)
        key=attention_helper.sa_attention_mapper_v3(index_pair,key.reshape(-1,self.d_model*an),cnt).reshape(-1,neighbor_num,self.d_model,an)
        value=attention_helper.sa_attention_mapper_v3(index_pair,value.reshape(-1,self.d_model*an),cnt).reshape(-1,neighbor_num,self.d_model,an)

        # sh=sh[:,1:].reshape(-1,neighbor_num,1,3)
        dot_atten=torch.einsum('phdi,pnhdi->pnh', query.reshape(-1,self.nhead,self.d_per_head,an),
                               key.reshape(-1,neighbor_num,self.nhead,self.d_per_head,an))
        # pos_atten=((key-query)*sh).sum(-1)
        # pos_atten=self.fc_gamma(pos_atten)
        # atten=dot_atten+pos_atten+dist_atten
        atten=dot_atten+dist_atten
        atten=atten/self.d_per_head ** 0.5
        atten=F.softmax(atten, dim=1)
        atten=F.dropout(atten, p=self.dropout, training=self.training)
        value=value.reshape(-1,neighbor_num,self.nhead,self.d_model//self.nhead,an)
        tgt2=torch.einsum('pnh,pnhci->phci', atten, value).reshape(-1,self.d_model,an)
        tgt2=self.linear_out(tgt2.permute(0,2,1)).permute(0,2,1)
        tgt=tgt.reshape(tgt.shape[0],-1,an)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt.permute(0,2,1))

        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + tgt2
        tgt = self.norm2(tgt).permute(0,2,1)
        tgt=tgt.reshape(tgt.shape[0],-1)
        return tgt