from typing import Optional, List


from torch import nn, Tensor
from .multi_head_attention_equi import MultiheadAttention_equi
from .utils import *
from e3nn import o3
from eqnet.utils import attention_helper

class TransformerDecoderLayer_equi_v4(nn.Module):

    def __init__(self, version, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu",):
        super().__init__()
        self.d_model=d_model
        self.linear_key=VNLinear(d_model, d_model)
        self.linear_value=VNLinear(d_model, d_model)
        self.linear_query=VNLinear(d_model, d_model)
        self.linear_out=VNLinear(d_model, d_model)
        self.nhead=nhead
        self.dropout=dropout
        self.d_per_head=d_model//nhead

        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model,nhead),
            nn.ReLU(inplace=True),
            nn.Linear(nhead,nhead),
        )

        self.norm1 = VNLayerNorm(d_model)
        self.norm2 = VNLayerNorm(d_model)

        self.ff=VNFeedForward(d_model,dim_feedforward)


    def forward(self, tgt, memory,index_pair,cnt1,cnt2,
                sh,dist_atten,
                ):

        neighbor_num=index_pair.shape[1]
        query=self.linear_query(tgt.reshape(-1,self.d_model,3)).reshape(-1,1,self.d_model,3)
        key=self.linear_key(memory.reshape(-1,self.d_model,3))
        value=self.linear_value(memory.reshape(-1,self.d_model,3))

        key=attention_helper.ca_attention_mapper_v3(index_pair,key.reshape(-1,self.d_model*3),cnt1,cnt2).reshape(-1,neighbor_num,self.d_model,3)
        value=attention_helper.ca_attention_mapper_v3(index_pair,value.reshape(-1,self.d_model*3),cnt1,cnt2).reshape(-1,neighbor_num,self.d_model,3)

        sh=sh[:,1:].reshape(-1,neighbor_num,1,3)
        dot_atten=torch.einsum('phdi,pnhdi->pnh', query.reshape(-1,self.nhead,self.d_per_head,3),
                               key.reshape(-1,neighbor_num,self.nhead,self.d_per_head,3))
        pos_atten=((key-query)*sh).sum(-1)
        pos_atten=self.fc_gamma(pos_atten)
        atten=dot_atten+pos_atten+dist_atten
        atten=atten/self.d_per_head ** 0.5
        atten=F.softmax(atten, dim=1)
        atten=F.dropout(atten, p=self.dropout, training=self.training)
        value=value.reshape(-1,neighbor_num,self.nhead,self.d_model//self.nhead,3)
        tgt2=torch.einsum('pnh,pnhci->phci', atten, value).reshape(-1,self.d_model,3)
        tgt2=self.linear_out(tgt2)
        tgt=tgt.reshape(tgt.shape[0],-1,3)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt)

        tgt2 = self.ff(tgt)
        tgt = tgt + tgt2
        tgt = self.norm2(tgt)
        tgt=tgt.reshape(tgt.shape[0],-1)
        return tgt