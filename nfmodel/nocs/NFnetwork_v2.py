import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts,Point_center_res_cate
curdir=os.path.dirname(os.path.realpath(__file__))
qnet_config_file=os.path.join(curdir,'qnet.yaml')
from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        self.qnet=MyQNet(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.backbone=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        if FLAGS.feat_for_ts:
            self.ts=Pose_Ts(F=FLAGS.feat_c_ts,k=FLAGS.Ts_c)
        else:
            self.ts=Point_center_res_cate()
        self.loss_fs_net = fs_net_loss()
        self.loss_nocs=nocs_loss()
        self.loss_consistency=consistency_loss()
        self.loss_inter=inter_loss_v2()


    def forward(self, depth, obj_id, camK,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,sphere_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None,do_inter=False):

        # FLAGS.sample_method = 'basic'
        output_dict = {}
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None
        gt_s=gt_s+mean_shape
        PC = PC_sample(def_mask, depth, camK, gt_2D)
        if PC is None:
            return None
        PC=PC+torch.clamp(0.001*torch.randn(bs,PC.shape[1], 3).to(PC.device), -0.005, 0.005)
        PC = PC.detach()
        if  not do_inter:
            max_real_scale=torch.max(gt_s,dim=-1)[0]
            pad_points=pad_points*(max_real_scale.reshape(-1,1,1))/(gt_s.reshape(-1,1,3))
            query_nocs=torch.cat([sdf_points,pad_points],dim=1)
            query_points=torch.bmm((query_nocs*gt_s.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)



            PC_1=PC
            gt_s_1=gt_s
            gt_green_v, gt_red_v = get_gt_v(gt_R)

            gt_fsnet_list_1 = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Tran': gt_t,
                'Size': gt_s_1,
            }
            # show_open3d(model_point[0].cpu().detach().numpy(),PC_1[0].cpu().detach().numpy())
            name_fs_list=['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size']
            pred_fsnet_list_1,bin_list_1= \
                self.gt_branch(PC,query_nocs,query_points,mean_shape,sym,cat_name)
            fsnet_loss_1=self.loss_fs_net(name_fs_list, pred_fsnet_list_1, gt_fsnet_list_1, sym)
            loss_nocs_1=self.loss_nocs(bin_list_1,sym)
            interpo_loss={'Nocs':loss_nocs_1}
            if FLAGS.use_consistency:
                loss_consistency_1=self.loss_consistency(bin_list_1,pred_fsnet_list_1,mean_shape,sym)
            else:
                loss_consistency_1=0
            consistency_loss={'consistency':loss_consistency_1}
            fsnet_loss=fsnet_loss_1
            loss_dict = {}
            loss_dict['fsnet_loss'] = fsnet_loss
            loss_dict['interpo_loss'] = interpo_loss
            loss_dict['consistency_loss'] = consistency_loss
            loss_dict['inter_loss'] = {
                'inter_r':0,
                'inter_t':0,
                'inter_nocs':0,
            }
            return loss_dict
        else:

            max_mean_scale=torch.max(mean_shape,dim=-1)[0]
            query_points=sphere_points*(max_mean_scale.reshape(-1,1,1))*1.4


            pred_fsnet_list_1,bin_list_1= \
                self.pred_branch(PC,query_points,mean_shape,sym,cat_name)


            pred_fsnet_list_2,bin_list_2= \
                self.pred_branch(PC,query_points,mean_shape,sym,cat_name)


            inter_dict=self.loss_inter(pred_fsnet_list_1,pred_fsnet_list_2,bin_list_1,bin_list_2,sym)
            loss_dict = {}

            loss_dict['inter_loss'] = inter_dict
            return loss_dict







    def gt_branch(self,PC,query_nocs,query_points,mean_shape,sym,cat_name):

        # show_open3d(PC[0].cpu().detach().numpy(),query_points[0].cpu().detach().numpy())

        pc_num=PC.shape[1]
        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        query_points_center=query_points-center
        pc_center,query_points_center,delta_r1,delta_t1,delta_s1 = self.data_augment(pc_center, query_points_center)
        # show_open3d(pc_center[0].cpu().detach().numpy(),query_points_center[0].cpu().detach().numpy())
        recon,point_fea,global_fea,feature_dict= self.backbone(pc_center)
        # point_fea=torch.cat([point_fea,global_fea.unsqueeze(1).repeat(1,pc_num,1)],dim=-1)
        p_green_R=self.rot_green(point_fea.permute(0,2,1))
        p_red_R=self.rot_red(point_fea.permute(0,2,1))

        if FLAGS.feat_for_ts:
            feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        else:
            feat_for_ts = pc_center
            objs=torch.zeros_like(feat_for_ts[:,0,0])
            T, s = self.ts(feat_for_ts.permute(0, 2, 1),objs)

        T=delta_t1.squeeze(1) + delta_s1 * torch.bmm(delta_r1, T.unsqueeze(2)).squeeze(2)
        s=s * delta_s1
        p_green_R = torch.bmm(delta_r1, p_green_R.unsqueeze(2)).squeeze(2)
        p_red_R = torch.bmm(delta_r1, p_red_R.unsqueeze(2)).squeeze(2)

        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)



        Pred_T = T + center.squeeze(1)  # bs x 3
        Pred_s = s  # this s is
        recon=recon+center


        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Recon': recon,
            'Tran': Pred_T,
            'Size': Pred_s,
        }


        query_num=query_points.shape[1]
        batch_size=query_points.shape[0]

        pred_nocs_bin=self.qnet(query_points_center,feature_dict).reshape(batch_size,query_num,3,FLAGS.bin_size).permute(0,-1,1,2).contiguous()
        ratio_x=ratio_dict[cat_name][0]
        ratio_y=ratio_dict[cat_name][1]
        ratio_z=ratio_dict[cat_name][2]

        if sym[0][0]==1:
            x_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_y
            x_start=0
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_bin_resolution=0
            z_start=0
            query_nocs[:,:,0]=torch.norm(query_nocs[:,:,(0,2)],dim=-1)
            query_nocs[:,:,2]=0
            query_nocs_bin=torch.zeros_like(query_nocs).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
        else:
            x_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_y
            z_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_z
            x_start=(-FLAGS.pad_radius)*ratio_x
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=(-FLAGS.pad_radius)*ratio_z
            query_nocs_bin=torch.zeros_like(query_nocs).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,2]=torch.clamp(((query_nocs[:,:,2]-z_start)/z_bin_resolution),0,FLAGS.bin_size-1).long()

        bin_list={
            'query_nocs_bin':query_nocs_bin,
            'query_points':query_points,
            'pred_nocs_bin':pred_nocs_bin,
            'x_bin_resolution':x_bin_resolution,
            'y_bin_resolution':y_bin_resolution,
            'z_bin_resolution':z_bin_resolution,
            'x_start':x_start,
            'y_start':y_start,
            'z_start':z_start,
        }
        return pred_fsnet_list,bin_list




    def pred_branch(self,PC,query_points,mean_shape,sym,cat_name):



        pc_num=PC.shape[1]
        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        query_points_center=query_points

        pc_center,query_points_center,delta_r1,delta_t1,delta_s1 = self.data_augment(pc_center, query_points_center)
        # show_open3d(pc_center[0].cpu().detach().numpy(),query_points_center[0].cpu().detach().numpy())
        recon,point_fea,global_fea,feature_dict= self.backbone(pc_center)
        # point_fea=torch.cat([point_fea,global_fea.unsqueeze(1).repeat(1,pc_num,1)],dim=-1)
        p_green_R=self.rot_green(point_fea.permute(0,2,1))
        p_red_R=self.rot_red(point_fea.permute(0,2,1))

        if FLAGS.feat_for_ts:
            feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        else:
            feat_for_ts = pc_center
            objs=torch.zeros_like(feat_for_ts[:,0,0])
            T, s = self.ts(feat_for_ts.permute(0, 2, 1),objs)

        T=delta_t1.squeeze(1) + delta_s1 * torch.bmm(delta_r1, T.unsqueeze(2)).squeeze(2)
        s=s * delta_s1
        p_green_R = torch.bmm(delta_r1, p_green_R.unsqueeze(2)).squeeze(2)
        p_red_R = torch.bmm(delta_r1, p_red_R.unsqueeze(2)).squeeze(2)

        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)



        Pred_T = T + center.squeeze(1)  # bs x 3
        Pred_s = s  # this s is
        recon=recon+center


        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Recon': recon,
            'Tran': Pred_T,
            'Size': Pred_s,
        }


        query_num=query_points.shape[1]
        batch_size=query_points.shape[0]

        pred_nocs_bin=self.qnet(query_points_center,feature_dict).reshape(batch_size,query_num,3,FLAGS.bin_size).permute(0,-1,1,2).contiguous()

        bin_list={
            'query_points':query_points,
            'pred_nocs_bin':pred_nocs_bin,

        }
        return pred_fsnet_list,bin_list












    def data_augment(self, PC, query_points):
        # augmentation
        bs = PC.shape[0]
        a=15.0
        delta_t1 = torch.rand(bs, 1, 3).cuda()
        delta_t1 = delta_t1.uniform_(-0.02, 0.02)
        delta_r1=torch.zeros(bs,3,3).cuda()
        for i in range(bs):
            x=torch.Tensor(1).cuda()
            x.uniform_(-a,a)
            y=torch.Tensor(1).cuda()
            y.uniform_(-a,a)
            z=torch.Tensor(1).cuda()
            z.uniform_(-a,a)
            delta_r1[i] = get_rotation_torch(x, y, z)
        delta_s1 = torch.rand(bs, 1).cuda()
        delta_s1 = delta_s1.uniform_(0.8, 1.2)

        PC = (PC - delta_t1) / delta_s1.unsqueeze(2) @ delta_r1
        query_points = (query_points - delta_t1) / delta_s1.unsqueeze(2) @ delta_r1
        return PC, query_points,delta_r1,delta_t1,delta_s1


    def build_params(self,):
        #  training_stage is a list that controls whether to freeze each module
        params_lr_list = []

        # pose
        params_lr_list.append(
            {
                "params": self.backbone.parameters(),
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
        if FLAGS.use_prior:
            params_lr_list.append(
                {
                    "params": self.qnet.qnet.parameters(),
                    "lr": float(FLAGS.lr) * FLAGS.lr_interpo,
                    "betas":(0.9, 0.99)
                }
            )
            params_lr_list.append(
                {
                    "params": self.qnet.fc_out.parameters(),
                    "lr": 0,
                }
            )
        else:
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

    def load_prior(self,cat_name,prior_dir,prior_name):
        #'qnet.fc_out.weight', 'qnet.fc_out.bias'
        cat_model_path=os.path.join(prior_dir,cat_name,prior_name)
        state_dict=torch.load(cat_model_path)
        own_dict=self.state_dict()
        for name in ['qnet.fc_out.weight', 'qnet.fc_out.bias']:
            own_dict[name].copy_(state_dict[name])
        print('load prior')