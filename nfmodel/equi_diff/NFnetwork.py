import torch
import torch.nn as nn
import absl.flags as flags
import os

import nfmodel.nocs.NFnetwork_v2

FLAGS = flags.FLAGS
from nfmodel.part_net_v8 import Equi_diff_gcn,Equi_diff_gcn_a5,Equi_diff_gcn_a6,MyQNet_equi_v4,MyQNet_equi_an
from equi_diff_models.diffusion import DiffusionModel_v2
from equi_diff_models.diffusion_ddm import DiffusionModel_ddm
from equi_diff_models.archs.diffusion_arch import DiffusionNet_v2,DiffusionNet_v3
from nfmodel.pointnet.PN2 import Net
from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_scale, defor_3D_rt, defor_3D_bc,get_rotation_torch
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
from nnutils.torch_util import *
import torch.optim as optim
from pytorch3d.ops import sample_farthest_points
from tools.rot_utils import get_rot_mat_y_first
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
)



class PIPS_s(nn.Module):
    def __init__(self):
        super(PIPS_s, self).__init__()
        curdir=os.path.dirname(os.path.realpath(__file__))
        qnet_config_file=os.path.join(curdir,FLAGS.qnet_config)
        if FLAGS.fea_type=='a5':
            self.backbone1=Equi_diff_gcn_a5(FLAGS.equi_neighbor_num)
        elif FLAGS.fea_type=='a6':
            self.backbone1=Equi_diff_gcn_a6(FLAGS.equi_neighbor_num)
        else:
            self.backbone1=Equi_diff_gcn(FLAGS.equi_neighbor_num)


        self.diffusion_model = DiffusionModel_v2(model=DiffusionNet_v3())

        if FLAGS.fea_type=='a5' or FLAGS.fea_type=='a6':
            self.qnet=MyQNet_equi_an(qnet_config_file)
        else:
            self.qnet=MyQNet_equi_v4(qnet_config_file)

        self.loss_coord=HuberPnPCost()

        self.loss_coord_2=nn.SmoothL1Loss(beta=0.5,reduction='mean')


    def forward(self, depth, def_mask, camK,gt_2D, gt_R, gt_t, aug_rt_t=None, aug_rt_r=None,aug_scale=None,
                latent_code=None,
                sphere_points=None,model_size=None,model_points=None):
        PC = PC_sample(def_mask, depth, camK, gt_2D)
        bs=PC.shape[0]


        nocs_scale=model_size
        gt_t=gt_t/1000.0

        PC = PC.detach()
        PC, gt_R, gt_t,nocs_scale= self.data_augment(PC, gt_R, gt_t,nocs_scale,aug_rt_t, aug_rt_r,aug_scale)


        query_nocs=sphere_points
        query_num=query_nocs.shape[1]



        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        noise_t = torch.from_numpy(np.random.uniform(-0.02, 0.02, (bs,3))).float().to(pc_center.device)
        pc_center=pc_center+noise_t[:,None,:]
        gt_t=gt_t+noise_t





        bs = PC.shape[0]

        if FLAGS.stage==2:
            with torch.no_grad():
                self.backbone1.eval()
                final_fea,final_point,fea_dict= self.backbone1(pc_center)
            # gt_final_point_norm=(final_point-(gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()))/nocs_scale.reshape(-1,1,1)
            gt_latent_code=torch.einsum('bij,bqj->bqi',gt_R,latent_code)
            if FLAGS.use_diff_ts:
                diff_loss, loss_100_latent,loss_100_trans,loss_100_scale, loss_1000_latent,loss_1000_trans,loss_1000_scale\
                    =self.diffusion_model.diffusion_model_from_latent_ts(gt_latent_code, gt_t-center.squeeze(1), nocs_scale,cond=final_fea, point=final_point)
            else:
                diff_loss, loss_100, loss_1000, model_out=self.diffusion_model.diffusion_model_from_latent(gt_latent_code, cond=final_fea)
            loss_dict={'loss':diff_loss,
                       'loss_100_latent':loss_100_latent,
                       'loss_100_trans':loss_100_trans,
                       'loss_100_scale':loss_100_scale,
                       'loss_1000_latent':loss_1000_latent,
                       'loss_1000_trans':loss_1000_trans,
                       'loss_1000_scale':loss_1000_scale,
                       'nocs_loss':0,
                       'var_loss':0}
            return loss_dict
        else:
            final_fea,_,fea_dict= self.backbone1(pc_center)
            noise_Rs=[]
            for i in range(bs):

                x=torch.Tensor(1)
                x.uniform_(-5,5)
                y=torch.Tensor(1)
                y.uniform_(-5,5)
                z=torch.Tensor(1)
                z.uniform_(-5,5)
                delta_r1 = get_rotation_torch(x, y, z)
                noise_Rs.append(delta_r1)
            noise_Rs=torch.stack(noise_Rs,dim=0).float().to(pc_center.device)
            noise_t = torch.from_numpy(np.random.uniform(-0.00, 0.00, (bs,3))).float().to(pc_center.device)
            noise_s = torch.from_numpy(np.random.uniform(1, 1, (bs,1))).float().to(pc_center.device)

            query_nocs=query_nocs*noise_s.reshape(-1,1,1)
            query_nocs=torch.bmm(query_nocs,noise_Rs.permute(0,2,1))
            gt_query_camera=torch.bmm((query_nocs.detach()*nocs_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()+noise_t.reshape(-1,1,3)
            model_camera=torch.bmm((model_points.detach()),gt_R.permute(0,2,1))*nocs_scale.reshape(-1,1,1)+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()
            gt_query_camera_norm=(gt_query_camera-(gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()))/nocs_scale.reshape(-1,1,1)
            gt_query_nocs=torch.bmm(gt_query_camera_norm,gt_R)

            pred_scale=torch.ones_like(nocs_scale)
            pred_dict=self.qnet(gt_query_camera,fea_dict,pred_scale.detach())

            # show_open3d(gt_query_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
            show_open3d(model_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
            # show_open3d(model_camera[1].detach().cpu().numpy(),pc_center[1].detach().cpu().numpy())
            # show_open3d(model_camera[2].detach().cpu().numpy(),pc_center[2].detach().cpu().numpy())
            # show_open3d(model_camera[3].detach().cpu().numpy(),pc_center[3].detach().cpu().numpy())
            pred_coord=pred_dict['coord'].reshape(bs*query_num,-1)
            pred_log_stds=pred_dict['log_stds'].reshape(bs*query_num,-1)
            pred_rot_1=pred_dict['rot_vec_1'].reshape(bs*query_num,-1)
            pred_rot_2=pred_dict['rot_vec_2'].reshape(bs*query_num,-1)
            pred_var_R=get_rot_mat_y_first(pred_rot_1,pred_rot_2).contiguous()
            diff=pred_coord-gt_query_camera_norm.reshape(bs*query_num,-1)

            if FLAGS.fea_type=='a5' or FLAGS.fea_type=='a6' or not FLAGS.use_var:
                nocs_loss=self.loss_coord_2(pred_coord,gt_query_camera_norm.reshape(bs*query_num,-1))
                total_loss=nocs_loss
                var_loss=nocs_loss
            else:
                nocs_loss,var_loss=self.loss_coord(diff,pred_log_stds,pred_var_R)
                nocs_loss=nocs_loss.mean()
                var_loss=var_loss.mean()
                total_loss=nocs_loss+var_loss
            loss_dict={'loss':total_loss,'loss_100':0,'loss_1000':0,
                       'loss_100_latent':0,
                       'loss_100_trans':0,
                       'loss_100_scale':0,
                       'loss_1000_latent':0,
                       'loss_1000_trans':0,
                       'loss_1000_scale':0,
                       'nocs_loss':nocs_loss.detach(),
                       'var_loss':var_loss.detach()}
            return loss_dict


    def data_augment(self, PC, gt_R, gt_t, gt_s,aug_rt_t, aug_rt_r,aug_scale):
        # augmentation
        bs = PC.shape[0]
        for i in range(bs):
            prop_rt = torch.rand(1)
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC[i, ...], gt_R[i, ...],
                                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
                PC[i, ...] = PC_new
                gt_R[i, ...] = gt_R_new
                gt_t[i, ...] = gt_t_new.view(-1)

            prop_pc = torch.rand(1)
            if prop_pc < FLAGS.aug_pc_pro:
                PC_new = defor_3D_pc(PC[i, ...], FLAGS.aug_pc_r)
                PC[i, ...] = PC_new

            prop_bb = torch.rand(1)
            if prop_bb < FLAGS.aug_bb_pro:
                #  R, t, s, s_x=(0.9, 1.1), s_y=(0.9, 1.1), s_z=(0.9, 1.1), sym=None
                PC_new, gt_s_new= defor_scale(PC[i, ...], gt_R[i, ...],
                                                               gt_t[i, ...], gt_s[i, ...] ,aug_bb=aug_scale[i, ...],)
                gt_s_new = gt_s_new
                PC[i, ...] = PC_new
                gt_s[i, ...] = gt_s_new



        return PC, gt_R, gt_t,gt_s


    def build_params(self,):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        # pose
        params_lr_list.append(
            {
                "params": self.backbone1.parameters(),
                "lr": float(FLAGS.lr) *FLAGS.lr_nocs,
            }
        )

        params_lr_list.append(
            {
                "params": self.diffusion_model.parameters(),
                "lr": float(FLAGS.lr),
            }
        )
        params_lr_list.append(
            {
                "params": self.qnet.parameters(),
                "lr": float(FLAGS.lr) *FLAGS.lr_nocs,
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
        if var_R!=None:
            inv_sigma=torch.bmm(inv_stds,var_R.permute(0,2,1))
        else:
            inv_sigma=inv_stds
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

