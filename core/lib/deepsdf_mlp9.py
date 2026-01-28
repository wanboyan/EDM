import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.lib.layers_equi import *

class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        self.dropout = True
        dropout_prob = 0.2
        self.use_tanh = True
        in_ch = 257
        out_ch = 1
        feat_ch = 512
        self.z_in = VNLinear(128, 128)
        print("[DeepSDF MLP-9] Dropout: {}; Do_prob: {}; in_ch: {}; hidden_ch: {}".format(self.dropout, dropout_prob, in_ch, feat_ch))
        if self.dropout is False:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True)
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Linear(feat_ch, out_ch)
            )
        else:
            self.net1 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(in_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch - in_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
            )

            self.net2 = nn.Sequential(
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.utils.weight_norm(nn.Linear(feat_ch, feat_ch)),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_prob, inplace=False),
                nn.Linear(feat_ch, out_ch)
            )

        num_params = sum(p.numel() for p in self.parameters())
        print('[num parameters: {}]'.format(num_params))

    # z: [B 131] 128-dim latent code + 3-dim xyz
    def forward(self, z,p):
        bs=z.shape[0]
        bs, T, D = p.shape
        net = (p * p).sum(2, keepdim=True)

        z = z.view(bs, -1, 3).contiguous()
        net_z = torch.einsum('bmi,bni->bmn', p, z)
        z_dir = self.z_in(z)
        z_inv = (z * z_dir).sum(-1).unsqueeze(1).repeat(1, T, 1)
        net = torch.cat([net, net_z, z_inv], dim=2)
        in1 = net
        out1 = self.net1(in1)
        in2 = torch.cat([out1, in1], dim=-1)
        out2 = self.net2(in2)
        if self.use_tanh:
            out2 = torch.tanh(out2)
        return out2
 
