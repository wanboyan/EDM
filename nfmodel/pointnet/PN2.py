import torch
import torch.nn as nn
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from module import PointNet2MSG


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.pn2msg = PointNet2MSG(radii_list=[[0.01, 0.02], [0.02,0.04], [0.04,0.08], [0.08,0.16]])
        self.t_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3, 1),
        )
        self.t_mlp[-1].bias.data.zero_()
        self.s_mlp = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 3, 1),
        )

    def forward(self, pts):


        x = torch.cat([pts, pts, pts], dim=2)
        x = self.pn2msg(x)

        t = self.t_mlp(x)
        t = torch.mean(t, dim=2)

        s = self.s_mlp(x).mean(2)


        return t,s


class Loss(nn.Module):
    def __init__(self, cfg):
        super(Loss, self).__init__()
        self.cfg = cfg
        self.criterion = nn.L1Loss()

    def forward(self, pred, gt):
        loss_t = self.criterion(pred['translation'], gt['translation_label'])
        loss_s = self.criterion(pred['size'], gt['size_label'])

        loss =  self.cfg.t_weight*loss_t+self.cfg.s_weight*loss_s
        return {
            'loss': loss,
            't': loss_t,
            's': loss_s,
        }