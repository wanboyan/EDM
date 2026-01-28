import torch
import torch.nn as nn
import absl.flags as flags
import os

import nfmodel.nocs.NFnetwork_v2

FLAGS = flags.FLAGS
from nfmodel.part_net_v8 import Weight_model2,MyQNet_equi_v5,Equi_gcn3,GCN3D_segR,MyQNet_v6
from nfmodel.pointnet.PN2 import Net
from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
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

        if FLAGS.qnet_version=='v5':
            self.qnet=MyQNet_equi_v5(qnet_config_file)

        elif FLAGS.qnet_version=='v6':
            self.qnet=MyQNet_v6(qnet_config_file)



        self.weight_model=Weight_model2()

        if FLAGS.backbone=='gcn_equi3':
            self.backbone1=Equi_gcn3(FLAGS.equi_neighbor_num)
        elif FLAGS.backbone=='gcn':
            self.backbone1=GCN3D_segR(FLAGS.gcn_sup_num,FLAGS.gcn_n_num)

        self.ts=Net()
        self.loss_fs_net=fs_net_loss()

        self.loss_coord=HuberPnPCost()



    def forward_fsnet_1(self,point_fea,query,center):
        feat_for_ts = query
        T,s= self.ts(feat_for_ts)
        pred_fsnet_list = {
            'Tran': T + center.squeeze(1),
            'Size': s,
        }
        return pred_fsnet_list

    def forward_fsnet_2(self,point_fea,query,center):
        feat_for_ts = torch.cat([point_fea,query.unsqueeze(2)],dim=2)
        T= self.t_2(feat_for_ts)
        s= self.s_2(feat_for_ts)
        p_green_R=self.rot_green_2(feat_for_ts)
        p_red_R=self.rot_red_2(feat_for_ts)
        p_green_R = F.normalize(p_green_R,dim=-1)
        p_red_R = F.normalize(p_red_R,dim=-1)
        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Tran': T + center.squeeze(1),
            'Size': s,
        }
        return pred_fsnet_list

    def forward(self, cloud,rgb,
                gt_R, gt_t, scale, sym=None,
                aug_rt_t=None, aug_rt_r=None, model_point=None,sphere_points=None):


        bs = cloud.shape[0]



        with torch.no_grad():
            PC, _ = sample_farthest_points(cloud, K=FLAGS.random_points)

        real_scale=scale
        PC = PC.detach()
        PC, gt_R, gt_t, gt_s = self.data_augment(PC, gt_R, gt_t, scale,
                                                 aug_rt_t, aug_rt_r)


        query_nocs=sphere_points
        query_num=query_nocs.shape[1]



        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        noise_t = torch.from_numpy(np.random.uniform(-0.02, 0.02, (bs,3))).float().to(pc_center.device)
        pc_center=pc_center+noise_t[:,None,:]
        gt_t=gt_t+noise_t





        bs = PC.shape[0]

        feature_dict= self.backbone1(pc_center)
        pred_scale=torch.ones_like(real_scale)[:,0]

        fsnet_loss={
            'Rot1': 0,
            'Rot2': 0,
            'Rot1_cos':0,
            'Rot2_cos':0,
            'Rot_r_a':0,
            'Tran': 0,
            'Size': 0
        }

        pips_s_loss = {'pair_loss':0,'pair_loss_nocs':0,
                       'rot_per_loss':0,'stable_loss':0,
                       'std_loss':0,'ratio_loss':0}
        NOCS_loss = {'nocs_loss':0,'var_loss':0,'c_nocs_loss':0,
                     'c_var_loss':0}

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
        noise_t = torch.from_numpy(np.random.uniform(-0.02, 0.02, (bs,3))).float().to(pc_center.device)
        noise_s = torch.from_numpy(np.random.uniform(0.8, 1.2, (bs,1))).float().to(pc_center.device)

        query_nocs=query_nocs*noise_s.reshape(-1,1,1)
        query_nocs=torch.bmm(query_nocs,noise_Rs.permute(0,2,1))
        gt_query_camera=torch.bmm((query_nocs.detach()),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()+noise_t.reshape(-1,1,3)
        model_camera=torch.bmm((model_point.detach()),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()
        gt_query_camera_norm=(gt_query_camera-(gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()))

        gt_query_nocs=torch.bmm(gt_query_camera_norm,gt_R)



        # show_open3d(gt_query_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
        # show_open3d(model_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
        # show_open3d(model_camera[1].detach().cpu().numpy(),pc_center[1].detach().cpu().numpy())
        # show_open3d(model_camera[2].detach().cpu().numpy(),pc_center[2].detach().cpu().numpy())
        # show_open3d(model_camera[3].detach().cpu().numpy(),pc_center[3].detach().cpu().numpy())



        pred_dict=self.qnet(gt_query_camera,feature_dict,pred_scale.detach())



        pred_coord=pred_dict['coord'].reshape(bs*query_num,-1)
        pred_log_stds=pred_dict['log_stds'].reshape(bs*query_num,-1)
        pred_rot_1=pred_dict['rot_vec_1'].reshape(bs*query_num,-1)
        pred_rot_2=pred_dict['rot_vec_2'].reshape(bs*query_num,-1)
        pred_var_R=get_rot_mat_y_first(pred_rot_1,pred_rot_2).contiguous()
        if sym[0][0]==0:
            diff=pred_coord-gt_query_nocs.reshape(bs*query_num,-1)
            nocs_loss,var_loss=self.loss_coord(diff,pred_log_stds,pred_var_R)
            NOCS_loss = {'nocs_loss':nocs_loss.mean(),'var_loss':var_loss.mean(),'c_nocs_loss':0,
                         'c_var_loss':0}
        else:
            gt_query_nocs_r=gt_query_nocs.clone()
            gt_query_nocs_r[:,:,0]=torch.norm(gt_query_nocs[:,:,(0,2)],dim=-1)
            gt_query_nocs_r[:,:,-1]=0

            plane_var_R=torch.zeros_like(pred_var_R[:,:2,:2])
            vec=F.normalize(pred_rot_1[:,:2],dim=-1)
            cos=vec[:,0]
            sin=vec[:,1]
            plane_var_R[:, 0, 0] = cos
            plane_var_R[:, 0, 1] = -sin
            plane_var_R[:, 1, 0] = sin
            plane_var_R[:, 1, 1] = cos
            plane_log_stds=pred_log_stds[:,:2]
            plane_diff=pred_coord-gt_query_nocs_r.reshape(bs*query_num,-1)
            plane_diff=plane_diff[:,:2]
            nocs_loss,var_loss=self.loss_coord(plane_diff,plane_log_stds,plane_var_R)
            NOCS_loss = {'nocs_loss':nocs_loss.mean(),'var_loss':var_loss.mean(),'c_nocs_loss':0,
                         'c_var_loss':0}












        name_fs_list=[ 'Tran', 'Size']
        pred_fsnet_list=self.forward_fsnet_1(None,pc_center,center)
        gt_fsnet_list = {
            'Tran': gt_t,
            'Size': real_scale,
        }
        fsnet_loss.update(self.loss_fs_net(name_fs_list,pred_fsnet_list,gt_fsnet_list,sym))



        loss_dict={}

        loss_dict['fsnet_loss'] = fsnet_loss
        loss_dict['pips_s_loss'] = pips_s_loss
        loss_dict['NOCS'] = NOCS_loss
        return loss_dict


    def data_augment(self, PC, gt_R, gt_t, gt_s, aug_rt_t, aug_rt_r):
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
        params_lr_list.append(
            {
                "params": self.weight_model.parameters(),
                "lr": float(FLAGS.lr) * FLAGS.lr_weight_model,
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

