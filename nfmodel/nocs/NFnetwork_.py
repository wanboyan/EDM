import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts
curdir=os.path.dirname(os.path.realpath(__file__))
qnet_config_file=os.path.join(curdir,'qnet.yaml')
from network.point_sample.pc_sample import PC_sample
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.consistency_loss import consistency_loss
import torch.nn.functional as F
class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        self.qnet=MyQNet(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.backbone=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        self.ts=Pose_Ts(F=FLAGS.feat_c_ts,k=FLAGS.Ts_c)
        self.loss_bin_fun=nn.CrossEntropyLoss(reduce=False)
        self.loss_fs_net = fs_net_loss()
        self.loss_consistency=consistency_loss()
        self.kl=nn.KLDivLoss(log_target=True,reduction='batchmean')

    def forward(self, depth, obj_id, camK,
                gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None, aug_bb=None,
                aug_rt_t=None, aug_rt_r=None, def_mask=None, model_point=None, nocs_scale=None,
                pad_points=None, sdf_points=None,do_aug=False,rgb=None,gt_mask=None,cat_name=None):

        # FLAGS.sample_method = 'basic'
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None
        PC = PC_sample(def_mask, depth, camK, gt_2D)

        PC = PC.detach()

        PC_1, PC_2, gt_s_1,gt_s_2,gt_R, gt_t, = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb,
                                                             aug_rt_t, aug_rt_r, model_point, nocs_scale, obj_id)


        real_scale_1=mean_shape+gt_s_1
        max_real_scale_1=torch.max(real_scale_1,dim=-1)[0]
        pad_points_1=pad_points*(max_real_scale_1.reshape(-1,1,1))/(real_scale_1.reshape(-1,1,3))
        query_nocs_1=torch.cat([sdf_points,pad_points_1],dim=1)
        query_points_1=torch.bmm((query_nocs_1*real_scale_1.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)
        # show_open3d(PC_1[0].cpu().detach().numpy(),query_points_1[0].cpu().detach().numpy())

        real_scale_2=mean_shape+gt_s_2
        max_real_scale_2=torch.max(real_scale_2,dim=-1)[0]
        pad_points_2=pad_points*(max_real_scale_2.reshape(-1,1,1))/(real_scale_2.reshape(-1,1,3))
        query_nocs_2=torch.cat([sdf_points,pad_points_2],dim=1)
        query_points_2=torch.bmm((query_nocs_2*real_scale_2.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)



        center_1=PC_1.mean(dim=1,keepdim=True)
        pc_center_1=PC_1-center_1
        recon_1,point_fea_1,global_fea_1,feature_dict_1= self.backbone(pc_center_1)

        p_green_R_1=self.rot_green(point_fea_1.permute(0,2,1))
        p_red_R_1=self.rot_red(point_fea_1.permute(0,2,1))
        feat_for_ts_1 = torch.cat([point_fea_1, pc_center_1], dim=2)
        T_1, s_1 = self.ts(feat_for_ts_1.permute(0, 2, 1))
        p_green_R_1 = p_green_R_1 / (torch.norm(p_green_R_1, dim=1, keepdim=True) + 1e-6)
        p_red_R_1 = p_red_R_1 / (torch.norm(p_red_R_1, dim=1, keepdim=True) + 1e-6)

        Pred_T_1 = T_1 + center_1.squeeze(1)  # bs x 3
        Pred_s_1 = s_1  # this s is
        recon_1=recon_1+center_1


        center_2=PC_2.mean(dim=1,keepdim=True)
        pc_center_2=PC_2-center_2
        recon_2,point_fea_2,global_fea_2,feature_dict_2= self.backbone(pc_center_2)

        p_green_R_2=self.rot_green(point_fea_2.permute(0,2,1))
        p_red_R_2=self.rot_red(point_fea_2.permute(0,2,1))
        feat_for_ts_2 = torch.cat([point_fea_2, pc_center_2], dim=2)
        T_2, s_2 = self.ts(feat_for_ts_2.permute(0, 2, 1))
        p_green_R_2 = p_green_R_2 / (torch.norm(p_green_R_2, dim=1, keepdim=True) + 1e-6)
        p_red_R_2 = p_red_R_2 / (torch.norm(p_red_R_2, dim=1, keepdim=True) + 1e-6)

        Pred_T_2 = T_2 + center_2.squeeze(1)  # bs x 3
        Pred_s_2 = s_2  # this s is
        recon_2=recon_2+center_2





        gt_green_v, gt_red_v = get_gt_v(gt_R)
        gt_fsnet_list = {
            'Rot1': gt_green_v,
            'Rot2': gt_red_v,
            'Recon': PC,
            'Tran': gt_t,
            'Size': gt_s,
        }
        pred_fsnet_list_1 = {
            'Rot1': p_green_R_1,
            'Rot2': p_red_R_1,
            'Recon': recon_1,
            'Tran': Pred_T_1,
            'Size': Pred_s_1,
        }
        pred_fsnet_list_2 = {
            'Rot1': p_green_R_2,
            'Rot2': p_red_R_2,
            'Recon': recon_2,
            'Tran': Pred_T_2,
            'Size': Pred_s_2,
        }
        name_fs_list=gt_fsnet_list
        fsnet_loss_1=self.loss_fs_net(name_fs_list, pred_fsnet_list_1, gt_fsnet_list, sym)
        fsnet_loss_2=self.loss_fs_net(name_fs_list, pred_fsnet_list_2, gt_fsnet_list, sym)


        query_num=query_points_1.shape[1]
        batch_size=query_points_1.shape[0]
        query_points_center_1=query_points_1-center_2
        query_points_center_2=query_points_2-center_2

        pred_nocs_bin_1=self.qnet(query_points_center_1,feature_dict_1).reshape(batch_size,query_num,3,FLAGS.bin_size).permute(0,-1,1,2).contiguous()
        pred_nocs_bin_2=self.qnet(query_points_center_2,feature_dict_2).reshape(batch_size,query_num,3,FLAGS.bin_size).permute(0,-1,1,2).contiguous()
        class_mean_shape=mean_shape[0]
        max_class_mean_shape=class_mean_shape.max()
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
        else:
            x_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_y
            z_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_z
            x_start=(-FLAGS.pad_radius)*ratio_x
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=(-FLAGS.pad_radius)*ratio_z


        def get_bin(query_nocs):
            if sym[0][0]==1:
                query_nocs[:,:,0]=torch.norm(query_nocs[:,:,(0,2)],dim=-1)
                query_nocs[:,:,2]=0
                query_nocs_bin=torch.zeros_like(query_nocs).long()
                query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
                query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
            else:
                query_nocs_bin=torch.zeros_like(query_nocs).long()
                query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
                query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
                query_nocs_bin[:,:,2]=torch.clamp(((query_nocs[:,:,2]-z_start)/z_bin_resolution),0,FLAGS.bin_size-1).long()
            return query_nocs_bin

        query_nocs_bin_1=get_bin(query_nocs_1)
        query_nocs_bin_2=get_bin(query_nocs_2)
        loss_nocs_1=self.loss_bin_fun(pred_nocs_bin_1,query_nocs_bin_1)
        loss_nocs_2=self.loss_bin_fun(pred_nocs_bin_2,query_nocs_bin_2)

        dis_1=F.log_softmax(pred_nocs_bin_1[:,:,:FLAGS.query_num,:].permute(0,2,3,1).reshape(-1,FLAGS.bin_size),dim=-1)
        dis_2=F.log_softmax(pred_nocs_bin_2[:,:,:FLAGS.query_num,:].permute(0,2,3,1).reshape(-1,FLAGS.bin_size),dim=-1)

        inter_loss=self.kl(dis_1,dis_2)*FLAGS.inter_lr
        loss_nocs_1=loss_nocs_1.mean()*FLAGS.interpo_w
        loss_nocs_2=loss_nocs_2.mean()*FLAGS.interpo_w
        # interpo_loss={'Nocs':loss_nocs_1+loss_nocs_2,
        #               'inter':inter_loss}
        interpo_loss={'Nocs':0,
                      'inter':inter_loss}
        show_open3d(pc_center_1[0].cpu().detach().numpy(),query_points_center_1[0].cpu().detach().numpy())


        bin_list_1={
           'query_points':query_points_1,
            'pred_nocs_bin':pred_nocs_bin_1,
            'x_bin_resolution':x_bin_resolution,
            'y_bin_resolution':y_bin_resolution,
            'z_bin_resolution':z_bin_resolution,
            'x_start':x_start,
            'y_start':y_start,
            'z_start':z_start,
        }
        bin_list_2={
            'query_points':query_points_2,
            'pred_nocs_bin':pred_nocs_bin_2,
            'x_bin_resolution':x_bin_resolution,
            'y_bin_resolution':y_bin_resolution,
            'z_bin_resolution':z_bin_resolution,
            'x_start':x_start,
            'y_start':y_start,
            'z_start':z_start,
        }

        loss_consistency_1=self.loss_consistency(bin_list_1,pred_fsnet_list_1,mean_shape,sym)
        loss_consistency_2=self.loss_consistency(bin_list_2,pred_fsnet_list_2,mean_shape,sym)
        consistency_loss={'consistency':loss_consistency_1+loss_consistency_2}

        # consistency_loss={'consistency':0}
        loss_dict = {}
        for k,v in fsnet_loss_1.items():
            fsnet_loss_1[k]=0
        loss_dict['fsnet_loss'] = fsnet_loss_1
        loss_dict['interpo_loss'] = interpo_loss
        loss_dict['consistency_loss'] = consistency_loss
        return loss_dict




    def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         model_point, nocs_scale, obj_ids):
        # augmentation
        bs = PC.shape[0]
        PC_1=torch.zeros_like(PC)
        PC_2=torch.zeros_like(PC)
        gt_s_1=torch.zeros_like(gt_s)
        gt_s_2=torch.zeros_like(gt_s)
        for i in range(bs):
            obj_id = int(obj_ids[i])
            prop_rt = torch.rand(1)
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC[i, ...], gt_R[i, ...],
                                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
                PC[i, ...] = PC_new
                gt_R[i, ...] = gt_R_new
                gt_t[i, ...] = gt_t_new.view(-1)


            # only do bc for mug and bowl


            prop_pc = torch.rand(1)
            if prop_pc < FLAGS.aug_pc_pro:
                PC_new = defor_3D_pc(PC[i, ...], FLAGS.aug_pc_r)
                PC[i, ...] = PC_new


            aug_bb_1 = torch.rand(3).cuda()
            aug_bb_1 = aug_bb_1.uniform_(0.8, 1.2)
            aug_bb_2 = torch.rand(3).cuda()
            aug_bb_2 = aug_bb_2.uniform_(0.8, 1.2)

            PC_new_1, gt_s_new_1 = defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                           gt_t[i, ...], gt_s[i, ...] + mean_shape[i, ...],
                                           sym=sym[i, ...], aug_bb=aug_bb_1)
            gt_s_new_1=gt_s_new_1-mean_shape[i, ...]
            PC_new_2, gt_s_new_2 = defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                               gt_t[i, ...], gt_s[i, ...] + mean_shape[i, ...],
                                               sym=sym[i, ...], aug_bb=aug_bb_2)
            gt_s_new_2=gt_s_new_2-mean_shape[i, ...]



            prop_bc = torch.rand(1)
            if prop_bc < FLAGS.aug_bc_pro and (obj_id == 5 or obj_id == 1):
                PC_new_1, gt_s_new_1 = defor_3D_bc(PC_new_1, gt_R[i, ...], gt_t[i, ...],
                                                gt_s_new_1 + mean_shape[i, ...],
                                               model_point[i, ...], nocs_scale[i, ...])
                gt_s_new_1=gt_s_new_1-mean_shape[i, ...]
                PC_new_2, gt_s_new_2 = defor_3D_bc(PC_new_2, gt_R[i, ...], gt_t[i, ...],
                                                   gt_s_new_2 + mean_shape[i, ...],
                                                   model_point[i, ...], nocs_scale[i, ...])
                gt_s_new_2=gt_s_new_2-mean_shape[i, ...]


            PC_1[i, ...] = PC_new_1
            PC_2[i, ...] = PC_new_2
            gt_s_1[i, ...] = gt_s_new_1
            gt_s_2[i, ...] = gt_s_new_2
            #  augmentation finish
        return PC_1,PC_2,gt_s_1,gt_s_2, gt_R, gt_t


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