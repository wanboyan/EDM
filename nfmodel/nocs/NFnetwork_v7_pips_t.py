import sysconfig

import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net_v7 import MyOccnet


from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
from nnutils.torch_util import *
import torch.optim as optim

from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
)

def KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    KLD = torch.mean(KLD)
    return KLD



class PIPS_T(nn.Module):
    def __init__(self):
        super(PIPS_T, self).__init__()
        self.occnet=MyOccnet()

        self.loss_scale=nn.SmoothL1Loss(beta=0.5,reduction='mean')
        self.loss_coord=HuberPnPCost()
        self.bce_loss=nn.BCELoss()
        gt_regular_grid,gt_regular_grid_bin=make_regular_grid(FLAGS.gt_regular_grid_spilt)
        self.gt_regular_grid=gt_regular_grid.cuda()
        self.gt_regular_grid_bin=gt_regular_grid_bin.cuda()

        print(1)



    def forward(self, depth, obj_id, camK,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None,
                coefficient=None,control_points=None,re_control_points=None,re_coefficient=None,
                model_idx=None,deform_sdf_points=None,
                std_model=None,
                regular_grid=None,
                generated_gt_coord=None,
                generated_gt_log_stds=None,
                generated_gt_quat=None,
                do_refine=False):


        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None




        PC = PC_sample(gt_mask, depth, camK, gt_2D)
        real_scale=mean_shape+gt_s
        PC = PC.detach()
        real_scale=mean_shape+gt_s

        nocs_scale=torch.norm(real_scale,dim=-1)



        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        bs = PC.shape[0]

        canonical_pc=torch.bmm(PC-gt_t.reshape(-1,1,3),gt_R)/(nocs_scale.reshape(-1,1,1))

        # show_open3d(canonical_pc[0].detach().cpu().numpy(),regular_grid[0].detach().cpu().numpy())
        logits=self.occnet(canonical_pc,regular_grid)
        weights=F.softmax(logits,dim=1)

        samples=torch.einsum('bps,bpi->bsi',weights,regular_grid)
        samples_num=samples.shape[1]

        gt_grid_log_std=generated_gt_log_stds.permute(0,2,1).reshape(bs,3,FLAGS.gt_regular_grid_spilt,
                                                                     FLAGS.gt_regular_grid_spilt,
                                                                     FLAGS.gt_regular_grid_spilt)
        gt_grid_quat=generated_gt_quat.permute(0,2,1).reshape(bs,4,FLAGS.gt_regular_grid_spilt,
                                                                     FLAGS.gt_regular_grid_spilt,
                                                                     FLAGS.gt_regular_grid_spilt)
        samples_log_std=F.grid_sample(gt_grid_log_std,
                                      samples.unsqueeze(-2).unsqueeze(-2)*2,
                                      align_corners=False).squeeze(-1).squeeze(-1).permute(0,2,1)
        samples_quat=F.grid_sample(gt_grid_quat,samples
                                   .unsqueeze(-2).unsqueeze(-2)*2,align_corners=False)\
            .squeeze(-1).squeeze(-1).permute(0,2,1)
        samples_var_R=quaternion_to_matrix(samples_quat).contiguous()
        inv_stds=torch.diag_embed(torch.exp(-samples_log_std))
        inv_sigma=torch.einsum('bpij,bpkj->bpik',inv_stds,samples_var_R)

        p=samples
        p_center=torch.mean(p,dim=1,keepdim=True)
        p=p-p_center
        p_norm=torch.norm(p,dim=-1,keepdim=True)
        p_norm_mean=torch.mean(p_norm,dim=1,keepdim=True)
        p=p/p_norm_mean
        FT=torch.zeros(bs,samples_num,3,6)
        h=hat(p)
        c=torch.einsum('psij,psjk->psik',-inv_sigma,h)
        FT[:,:,:,:3]=c
        FT[:,:,:,3:]=inv_sigma
        C=torch.einsum('psij,psik->pjk',FT,FT)
        values=torch.linalg.eigvalsh(C)
        conds=values[:,0]/values[:,-1]

        samples_var=samples_log_std.sum(-1)
        var_loss=samples_var.mean()
        cond_loss=-conds.mean()


        loss_dict={}
        loss_dict['interpo_loss']={'cond':cond_loss*100,'var':var_loss}
        return loss_dict

    def generate_gt(self,gt_index,rgb,depth, obj_id, camK, regular_grid,
                    gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                    aug_rt_t=None, aug_rt_r=None, def_mask=None, gt_mask=None, model_point=None, nocs_scale=None):
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None

        PC = PC_sample(gt_mask, depth, camK, gt_2D)
        real_scale=mean_shape+gt_s
        PC = PC.detach()
        nocs_scale=torch.norm(real_scale,dim=-1)
        bs = PC.shape[0]
        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        feature_dict,pred_scale= self.backbone1(pc_center)
        regular_grid_camera=torch.bmm((regular_grid.detach()*nocs_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()

        # show_open3d(regular_grid_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
        pred_dict=self.qnet(regular_grid_camera,feature_dict,pred_scale.detach())
        pred_log_stds=pred_dict['log_stds']
        pred_var=torch.exp(torch.sum(pred_log_stds,dim=-1))
        v,topk_index=torch.topk(pred_var, FLAGS.gt_topk, dim=-1,largest=True)
        v,lowk_index=torch.topk(pred_var, FLAGS.gt_topk, dim=-1,largest=False)
        return np.concatenate([topk_index.cpu().numpy(),lowk_index.cpu().numpy()],axis=0)





    def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         model_point, nocs_scale, obj_ids):
        # augmentation
        bs = PC.shape[0]
        for i in range(bs):
            obj_id = int(obj_ids[i])
            prop_rt = torch.rand(1)
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC[i, ...], gt_R[i, ...],
                                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
                PC[i, ...] = PC_new
                gt_R[i, ...] = gt_R_new
                gt_t[i, ...] = gt_t_new.view(-1)

            prop_bc = torch.rand(1)
            # only do bc for mug and bowl


            prop_pc = torch.rand(1)
            if prop_pc < FLAGS.aug_pc_pro:
                PC_new = defor_3D_pc(PC[i, ...], FLAGS.aug_pc_r)
                PC[i, ...] = PC_new


            prop_bb = torch.rand(1)
            model_point_new=model_point[i,...]
            if prop_bb < FLAGS.aug_bb_pro:
                #  R, t, s, s_x=(0.9, 1.1), s_y=(0.9, 1.1), s_z=(0.9, 1.1), sym=None
                PC_new, gt_s_new,model_point_new = defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                               gt_t[i, ...], gt_s[i, ...] + mean_shape[i, ...],
                                               sym=sym[i, ...], aug_bb=aug_bb[i, ...],
                                                               model_points=model_point[i,...],nocs_scale=nocs_scale[i, ...])
                gt_s_new = gt_s_new - mean_shape[i, ...]
                PC[i, ...] = PC_new
                gt_s[i, ...] = gt_s_new

            #  augmentation finish
        return PC, gt_R, gt_t, gt_s


    def build_params(self,):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        # pose
        params_lr_list.append(
            {
                "params": self.occnet.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_backbone,
            }
        )


        return params_lr_list

class HuberPnPCost(nn.Module):

    def __init__(self, delta=None, eps=1e-10,relative_delta=0.1):
        super(HuberPnPCost, self).__init__()
        self.eps = eps
        self.delta = delta
        self.relative_delta = relative_delta
        self.register_buffer('mean_inv_std', torch.tensor(1.0, dtype=torch.float))
        self.momentum = 0.01


    def forward(self,diff,log_stds,var_R, momentum=1.0,delta=1.414,
                eps=1e-4, training=True):
        inv_stds=torch.diag_embed(torch.exp(-log_stds)).clamp(max=1/eps)
        inv_sigma=torch.bmm(inv_stds,var_R.permute(0,2,1))
        diff_weighted=torch.einsum('kij,kj->ki',inv_sigma,diff)
        loss = torch.where(diff_weighted < delta,
                           0.5 * torch.square(diff_weighted),
                           delta * diff_weighted-1)
        if training:
            self.mean_inv_std *= 1 - momentum
            self.mean_inv_std += momentum * torch.mean(torch.exp(-log_stds).detach())
        loss.div_(self.mean_inv_std.clamp(min=1e-6))
        log_stds=log_stds/(self.mean_inv_std.clamp(min=1e-6))
        return loss,log_stds

