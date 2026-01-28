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

        # self.loss_coord=HuberPnPCost()
        self.loss_coord=nn.SmoothL1Loss(beta=0.5,reduction='mean')


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

    def forward(self, sphere_points=None,model_points=None,
                group=None,):
        pc_nocs=model_points
        bs=pc_nocs.shape[0]



        pc_nocs = pc_nocs.detach()

        noise_t = torch.from_numpy(np.random.uniform(-0.2, 0.2, (bs,1,3))).float().to(pc_nocs.device)
        query_nocs=sphere_points+noise_t
        query_num=query_nocs.shape[1]



        vis_flag=True

        if vis_flag:
            from sklearn.manifold import TSNE
            self.backbone1.eval()
            self.qnet.eval()
            with torch.no_grad():
                feature_dict= self.backbone1(pc_nocs[:1])

            x, y = np.mgrid[-1:1:100j,
                   -1:1:100j]
            split=x.shape[0]
            x=torch.from_numpy(x).float().cuda()
            y=torch.from_numpy(y).float().cuda()
            z=torch.zeros_like(x)
            plane_query=torch.stack([x,y,z],dim=-1).reshape(1,-1,3)
            pred_scale=torch.ones_like(pc_nocs[:1])[:,0,0]
            with torch.no_grad():
                dic=self.qnet(plane_query,feature_dict,pred_scale.detach())
            plane_fea=dic['z_inv']
            pred_coord=dic['coord']
            # show_open3d(pc_nocs[0].cpu().numpy(),pred_coord[0].detach().cpu().numpy())
            tsne = TSNE(n_components=1, init='pca', random_state=0)
            # feat_np=tsne.fit_transform(plane_fea[0].detach().cpu().numpy()).reshape(split,split)
            # feat_np=(plane_fea[0,:,233].detach().cpu().numpy()).reshape(split,split)
            if FLAGS.use_fund:
                feat_np=(pred_coord[0,:,0].detach().cpu().numpy()).reshape(split,split)
            else:
                feat_np=tsne.fit_transform(plane_fea[0].detach().cpu().numpy()).reshape(split,split)
            fig, ax = plt.subplots()

            x_np=x.detach().cpu().numpy()
            y_np=y.detach().cpu().numpy()
            x_min=x_np.min()
            x_max=x_np.max()
            y_min=y_np.min()
            y_max=y_np.max()
            cm=plt.cm.get_cmap('rainbow')
            c = plt.pcolormesh(x_np, y_np, feat_np, cmap =cm, vmin = feat_np.min(), vmax = feat_np.max())
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(y_min,y_max)
            # plt.colorbar(c)
            plt.xticks([])
            plt.yticks([])
            # plt.show()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig(os.path.join(FLAGS.model_save,group+str(FLAGS.use_fund)+'fea.png'), dpi=300,bbox_inches='tight')
            return None

        noise_Rs=[]
        a=360
        for i in range(bs):

            x=torch.Tensor(1)
            x.uniform_(-a,a)
            y=torch.Tensor(1)
            y.uniform_(-a,a)
            z=torch.Tensor(1)
            z.uniform_(-a,a)
            delta_r1 = get_rotation_torch(x, y, z)
            noise_Rs.append(delta_r1)
        noise_Rs=torch.stack(noise_Rs,dim=0).float().to(pc_nocs.device)


        query_camera=torch.bmm((query_nocs.detach()),noise_Rs.permute(0,2,1))
        pc_camera=torch.bmm((pc_nocs),noise_Rs.permute(0,2,1))
        # show_open3d(pc_nocs[0].cpu().numpy(),query_nocs[0].cpu().numpy())
        feature_dict= self.backbone1(pc_camera)

        pred_scale=torch.ones_like(pc_camera)[:,0,0]
        pred_dict=self.qnet(query_camera,feature_dict,pred_scale.detach())
        pred_coord=pred_dict['coord']

        if FLAGS.use_fund:
            query_nocs_sym=torch.zeros_like(query_nocs)
            if group=='cinf':
                query_nocs_r=torch.norm(query_nocs[:,:,:2],dim=-1)
                query_nocs_sym[:,:,0]=query_nocs_r
                query_nocs_sym[:,:,2]=query_nocs[:,:,2]
            elif group=='c4v':
                query_nocs_sym=get_ro_sym_coord(4,query_nocs)
                query_nocs_sym[:,:,1]=torch.abs(query_nocs_sym[:,:,1])
            elif group=='c2v':
                query_nocs_sym=get_ro_sym_coord(2,query_nocs)
                query_nocs_sym[:,:,1]=torch.abs(query_nocs_sym[:,:,1])
            # show_open3d(query_nocs_sym[0].cpu().numpy(),pc_nocs[0].cpu().numpy())
            loss=self.loss_coord(pred_coord,query_nocs_sym)
        else:
            loss=self.loss_coord(pred_coord,query_nocs)


        return loss




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


def get_ro_sym_coord(discrete,vertices):
    bs=vertices.shape[0]
    vertices=vertices.reshape(-1,3)
    sym_r=[]# Convert degrees to radians
    for i in range(discrete):
        theta_degrees=(360/discrete)*i
        theta_radians = np.radians(theta_degrees)
        r=np.array([
            [np.cos(theta_radians), -np.sin(theta_radians)],
            [np.sin(theta_radians), np.cos(theta_radians)]
        ])
        new_r=np.eye(3)
        new_r[:2,:2]=r
        sym_r.append(torch.from_numpy(new_r).float())
    sym_r=torch.stack(sym_r,dim=0).to(vertices.device)
    vertices_sym=vertices[:,None,:].repeat(1,discrete,1)
    vertices_sym=torch.einsum(' p r j , r i j->  p r i',vertices_sym,sym_r)
    vertices_sym_flat=vertices_sym.reshape(-1,3)
    vertices_rad_flat=torch.arctan2(vertices_sym_flat[:,1],vertices_sym_flat[:,0])
    vertices_degree_flat=vertices_rad_flat
    vertices_degree=vertices_degree_flat.reshape(-1,discrete)
    degree_index=torch.argmin(torch.abs(vertices_degree-0),dim=-1)
    vertices_found_1=vertices_sym[np.arange(len(vertices_sym)),degree_index]
    vertices_found_1=vertices_found_1.reshape(bs,-1,3)
    return vertices_found_1



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

