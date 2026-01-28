import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net_v8 import Rot_red,Rot_green,MyQNet_equi,Pose_Ts,Point_center_res_cate,Equi_gcn2

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



class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        curdir=os.path.dirname(os.path.realpath(__file__))
        qnet_config_file=os.path.join(curdir,FLAGS.qnet_config)
        self.qnet=MyQNet_equi(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)

        self.backbone1=Equi_gcn2(FLAGS.equi_neighbor_num)

        self.ts=Point_center_res_cate()
        self.loss_scale=nn.SmoothL1Loss(beta=0.5,reduction='mean')
        self.loss_coord=HuberPnPCost()




    def forward_fsnet(self,point_fea,pc_center,center):
        feat_for_ts = pc_center
        objs=torch.zeros_like(feat_for_ts[:,0,0])
        T, s = self.ts(feat_for_ts.permute(0, 2, 1),objs)
        p_green_R=self.rot_green(point_fea.permute(0,2,1))
        p_red_R=self.rot_red(point_fea.permute(0,2,1))
        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)
        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Tran': T + center.squeeze(1),
            'Size': s,
        }
        return pred_fsnet_list

    def forward(self, depth, obj_id, camK,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None,
                coefficient=None,control_points=None,re_control_points=None,re_coefficient=None,
                model_idx=None,deform_sdf_points=None,
                std_model=None,
                do_refine=False):


        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None




        PC = PC_sample(def_mask, depth, camK, gt_2D)
        real_scale=mean_shape+gt_s
        PC = PC.detach()
        PC, gt_R, gt_t, gt_s = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb,
                                                     aug_rt_t, aug_rt_r, model_point, nocs_scale, obj_id)
        real_scale=mean_shape+gt_s

        nocs_scale=torch.norm(real_scale,dim=-1)


        query_nocs=sdf_points
        query_num=query_nocs.shape[1]

        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        bs = PC.shape[0]

        if FLAGS.backbone=='neuron':
            feature_dict,pred_scale,_= self.backbone1(pc_center)
        else:
            feature_dict= self.backbone1(pc_center)
        pred_scale=nocs_scale



        gt_query_camera=torch.bmm((query_nocs.detach()*nocs_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()

        # show_open3d(gt_query_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())


        pred_dict=self.qnet(gt_query_camera,feature_dict,pred_scale.detach())
        pred_coord=pred_dict['coord'].reshape(bs*query_num,-1)
        pred_log_stds=pred_dict['log_stds'].reshape(bs*query_num,-1)
        pred_quat=pred_dict['quat'].reshape(bs*query_num,-1)
        use_var=True
        if use_var:
            pred_var_R=quaternion_to_matrix(pred_quat).contiguous()
            diff=pred_coord-query_nocs.reshape(bs*query_num,-1)
            nocs_loss,var_loss=self.loss_coord(diff,pred_log_stds,pred_var_R)
            loss_scale=self.loss_scale(pred_scale,nocs_scale)
            # loss_scale=0
            nocs_loss=nocs_loss.mean()
            var_loss=var_loss.mean()

        else:
            loss_nocs=self.loss_scale(pred_coord,query_nocs.reshape(bs*query_num,-1))
            loss_var=0
            loss_scale=0

        fsnet_loss={
            'Rot1': 0,
            'Rot2': 0,
            'Rot1_cos':0,
            'Rot2_cos':0,
            'Rot_r_a':0,
            'Tran': 0,
            'Size': 0
        }
        loss_dict={}
        loss_dict['interpo_loss']={'nocs':nocs_loss,'var':var_loss,'scale':loss_scale}
        loss_dict['fsnet_loss'] = fsnet_loss
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

        if FLAGS.backbone=='neuron':
            feature_dict,pred_scale= self.backbone1(pc_center)
        else:
            feature_dict= self.backbone1(pc_center)
        pred_scale=nocs_scale



        regular_grid_camera=torch.bmm((regular_grid.detach()*nocs_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()

        # show_open3d(regular_grid_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
        pred_dict=self.qnet(regular_grid_camera,feature_dict,pred_scale.detach())
        return pred_dict

    def pips_pose(self, pips_t,depth, obj_id, camK,regular_grid,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None,
                coefficient=None,control_points=None,re_control_points=None,re_coefficient=None,
                model_idx=None,deform_sdf_points=None,
                std_model=None,
                do_refine=False):


        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None




        PC = PC_sample(def_mask, depth, camK, gt_2D)
        real_scale=mean_shape+gt_s
        PC = PC.detach()
        PC, gt_R, gt_t, gt_s = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb,
                                                 aug_rt_t, aug_rt_r, model_point, nocs_scale, obj_id)
        real_scale=mean_shape+gt_s

        nocs_scale=torch.norm(real_scale,dim=-1)


        query_nocs=sdf_points
        query_num=query_nocs.shape[1]

        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        bs = PC.shape[0]

        canonical_pc=torch.bmm(PC-gt_t.reshape(-1,1,3),gt_R)/(nocs_scale.reshape(-1,1,1))

        # show_open3d(canonical_pc[0].detach().cpu().numpy(),regular_grid[0].detach().cpu().numpy())
        # show_open3d(regular_grid_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
        if FLAGS.backbone=='neuron':
            feature_dict,pred_scale= self.backbone1(pc_center)
        else:
            feature_dict= self.backbone1(pc_center)
        pred_scale=nocs_scale

        if FLAGS.use_pick:

            logits=pips_t.occnet(canonical_pc,regular_grid)

            pick_index=torch.topk(logits,FLAGS.gt_topk,1,largest=False)[1]
            for i in range(bs):
                r=math.floor(FLAGS.gt_topk*(FLAGS.pick_ratio))
                random_index=torch.randperm(regular_grid.shape[1])[:FLAGS.gt_topk-r]
                pick_index[i,r:,0]=random_index


            regular_grid_pick=torch.gather(regular_grid,1,pick_index.expand(-1,-1,3))
        else:
            regular_grid_pick=regular_grid
        noise=torch.rand_like(regular_grid_pick)*0.5*(1/FLAGS.regular_grid_spilt)
        regular_grid_pick=regular_grid_pick+noise
        regular_grid_camera_pick=torch.bmm((regular_grid_pick.detach()*nocs_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()







        pred_dict=self.qnet(regular_grid_camera_pick,feature_dict,pred_scale.detach())
        pred_coord=pred_dict['coord']





        nocs_loss=self.loss_scale(pred_coord.reshape(-1,3),regular_grid_pick.reshape(-1,3))
        var_loss=0
        loss_scale=0

        fsnet_loss={
            'Rot1': 0,
            'Rot2': 0,
            'Rot1_cos':0,
            'Rot2_cos':0,
            'Rot_r_a':0,
            'Tran': 0,
            'Size': 0
        }
        loss_dict={}
        loss_dict['interpo_loss']={'nocs':nocs_loss,'var':var_loss,'scale':loss_scale}
        loss_dict['fsnet_loss'] = fsnet_loss
        return loss_dict



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
                "params": self.backbone1.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_backbone,
            }
        )
        if FLAGS.two_back:
            params_lr_list.append(
                {
                    "params": self.backbone2.parameters(),
                    "lr": float(FLAGS.lr) * FLAGS.lr_backbone,
                }
            )
        params_lr_list.append(
            {
                "params": self.rot_red.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_rot,
            }
        )
        params_lr_list.append(
            {
                "params": self.rot_green.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_rot,
            }
        )

        params_lr_list.append(
            {
                "params": self.qnet.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_interpo,
                "betas":(0.9, 0.99)
            }
        )
        params_lr_list.append(
            {
                "params": self.ts.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_ts,

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

