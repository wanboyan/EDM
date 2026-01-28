import torch
import torch.nn as nn
import absl.flags as flags
import os

import nfmodel.nocs.NFnetwork_v2

FLAGS = flags.FLAGS
from nfmodel.part_net_v8 import Weight_model2,MyQNet_equi_v5,MyQNet_equi_v7,Equi_gcn3,Equi_gcn4,GCN3D_segR,MyQNet_v6
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
        elif FLAGS.qnet_version=='v7':
            self.qnet=MyQNet_equi_v7(qnet_config_file)
            self.c_loss_coord=HuberPnPCost()


        self.weight_model=Weight_model2()

        if FLAGS.backbone=='gcn_equi3':
            self.backbone1=Equi_gcn3(FLAGS.equi_neighbor_num)
        if FLAGS.backbone=='gcn_equi4':
            self.backbone1=Equi_gcn4(FLAGS.equi_neighbor_num)
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

    def forward(self, img_id,scene_id,roi_rgb,depth, def_mask, camK,gt_2D,
                gt_R, gt_t, aug_rt_t=None, aug_rt_r=None,
                sphere_points=None,model_size=None,model_points=None,
                sym_trans=None,):
        sym_trans[:,:3,3]=0
        PC = PC_sample(def_mask, depth, camK, gt_2D)
        bs=PC.shape[0]

        model_size=model_size/1000.0
        model_size=model_size[None,:].repeat(bs,1)
        model_points=model_points/1000.0
        model_points=model_points[None,:,:].repeat(bs,1,1)
        nocs_scale=torch.norm(model_size,dim=-1)
        gt_t=gt_t/1000.0

        PC = PC.detach()
        # PC, gt_R, gt_t = self.data_augment(PC, gt_R, gt_t,aug_rt_t, aug_rt_r)


        query_nocs=sphere_points
        query_num=query_nocs.shape[1]



        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        noise_t = torch.from_numpy(np.random.uniform(-0.02, 0.02, (bs,3))).float().to(pc_center.device)
        pc_center=pc_center+noise_t[:,None,:]
        gt_t=gt_t+noise_t
        bs = PC.shape[0]



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
        a=0
        for i in range(bs):

            x=torch.Tensor(1)
            x.uniform_(-a,a)
            y=torch.Tensor(1)
            y.uniform_(-a,a)
            z=torch.Tensor(1)
            z.uniform_(-a,a)
            delta_r1 = get_rotation_torch(x, y, z)
            noise_Rs.append(delta_r1)
        noise_Rs=torch.stack(noise_Rs,dim=0).float().to(pc_center.device)
        noise_t = torch.from_numpy(np.random.uniform(-0, 0, (bs,3))).float().to(pc_center.device)
        noise_s = torch.from_numpy(np.random.uniform(1, 1, (bs,1))).float().to(pc_center.device)

        query_nocs=query_nocs*noise_s.reshape(-1,1,1)
        query_nocs=torch.bmm(query_nocs,noise_Rs.permute(0,2,1))
        query_nocs_r=torch.norm(query_nocs[:,:,:2],dim=-1)
        sym_query_nocs=torch.zeros_like(query_nocs)
        sym_query_nocs[:,:,0]=0
        sym_query_nocs[:,:,1]=query_nocs_r
        sym_query_nocs[:,:,2]=query_nocs[:,:,2]
        gt_query_camera=torch.bmm((query_nocs.detach()*nocs_scale.reshape(-1,1,1)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()+noise_t.reshape(-1,1,3)
        model_camera=torch.bmm((model_points.detach()),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()
        gt_query_camera_norm=(gt_query_camera-(gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()))/nocs_scale.reshape(-1,1,1)
        gt_query_nocs=torch.bmm(gt_query_camera_norm,gt_R)

        pc_nocs=(pc_center-(gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()))/nocs_scale.reshape(-1,1,1)
        pc_nocs=torch.bmm(pc_nocs,gt_R)

        show_all(scene_id,img_id,roi_rgb,pc_center,gt_query_camera,sym_query_nocs)

        return




    def data_augment(self, PC, gt_R, gt_t, aug_rt_t, aug_rt_r):
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

        return PC, gt_R, gt_t


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

def show_all(scene_ids,img_ids,roi_rgbs,pc_list,query_list,query_nocs_list):
    import plotly.graph_objects as go
    import plotly.io as pio
    if not os.path.exists(FLAGS.pic_save_dir):
        os.makedirs(FLAGS.pic_save_dir)
    for j,(scene_id,im_id) in enumerate(zip(scene_ids,img_ids)):
        im_id=im_id.item()
        scene_id=scene_id.item()
        img=roi_rgbs[j].permute(1,2,0).cpu().numpy()
        rgb_path=os.path.join(FLAGS.pic_save_dir,f'{scene_id}_{im_id}_{j}_rgb.png')
        cv2.imwrite(rgb_path,img)
        pc=pc_list[0].cpu().numpy()
        query=query_list[0].cpu().numpy()
        color=query_nocs_list[0].cpu().numpy()+0.5
        trace = go.Scatter3d(x=pc[:,0], y=pc[:,1], z=pc[:,2],
                             mode='markers',
                             marker=dict(size=3,color='gray',
                                         opacity=1))
        trace_2 = go.Scatter3d(x=query[:,0], y=query[:,1], z=query[:,2],
                             mode='markers',
                             marker=dict(size=2,color=color,
                                         opacity=1))
        trace_3 = go.Scatter3d(x=query[:,0], y=query[:,1], z=query[:,2],
                               mode='markers',
                               marker=dict(size=2,color='gray',
                                           opacity=1))

        data = go.Data([trace])
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                zaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                bgcolor="rgba(0, 0, 0, 0)",
                camera=dict(
                    eye=dict(x=0, y=0, z=1.2),  # Adjust the camera position here
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            )
        )

        fig.show()


        data = go.Data([trace_3])
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                zaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                bgcolor="rgba(0, 0, 0, 0)",
                camera=dict(
                    eye=dict(x=0, y=0, z=1.2),  # Adjust the camera position here
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            )
        )

        fig.show()







        data = go.Data([trace,trace_2])
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                zaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    title='',
                    backgroundcolor="rgba(0, 0, 0, 0)"
                ),
                bgcolor="rgba(0, 0, 0, 0)",
                camera=dict(
                    eye=dict(x=0, y=0, z=1.2),  # Adjust the camera position here
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                )
            )
        )

        fig.show()
        point_path=os.path.join(FLAGS.pic_save_dir,f'{scene_id}_{im_id}_{j}_point.png')
        # pio.write_image(fig,point_path, width=2000,height=2000,)






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

