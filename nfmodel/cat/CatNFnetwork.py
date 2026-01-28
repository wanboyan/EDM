import torch
import torch.nn as nn
import absl.flags as flags
import torch.nn.functional as F
import os
FLAGS = flags.FLAGS
from nfmodel.part_net import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts

from network.point_sample.pc_sample import PC_sample
from datasets.data_augmentation import defor_3D_pc, cat_defor_3D_bb, defor_3D_rt
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.Catfs_net_loss import fs_net_loss
class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        curdir=os.path.dirname(os.path.realpath(__file__))
        qnet_config_file=os.path.join(curdir,FLAGS.qnet_config)
        self.qnet=MyQNet(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.backbone=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        self.ts=Pose_Ts(F=FLAGS.feat_c_ts,k=FLAGS.Ts_c)
        self.loss_bin_fun=nn.CrossEntropyLoss(reduce=False)
        self.loss_fs_net = fs_net_loss()
    def SoftCross(self,input,target):
        log_like=-F.log_softmax(input,dim=1)
        loss=torch.sum(log_like*target,dim=1)
        return loss

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
        if do_aug:
            PC_da, gt_R_da, gt_t_da, gt_s_da = self.data_augment(PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb,
                                                                 aug_rt_t, aug_rt_r, model_point, nocs_scale, obj_id)
            PC = PC_da
            gt_R = gt_R_da
            gt_t = gt_t_da
            gt_s = gt_s_da

        real_scale=mean_shape+gt_s
        if sym[0][4]==1:
            real_scale[:,2]=real_scale[:,0]
        max_real_scale=torch.max(real_scale,dim=-1)[0]
        pad_points=pad_points*(max_real_scale.reshape(-1,1,1))/(real_scale.reshape(-1,1,3))
        query_nocs=torch.cat([sdf_points,pad_points],dim=1)
        query_points=torch.bmm((query_nocs*real_scale.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)
        # show_open3d((query_nocs*real_scale.reshape(-1,1,3))[0].cpu().detach().numpy(),model_point[0].cpu().detach().numpy())
        # show_open3d(PC[0].cpu().detach().numpy(),query_points[0].cpu().detach().numpy())
        # import matplotlib.pyplot as plt
        # plt.imshow(rgb[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        # plt.show()
        # plt.imshow(depth[0].cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        # plt.show()
        # plt.imshow(def_mask[0].cpu().numpy().astype(np.uint8))
        # plt.show()



        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        recon,point_fea,global_fea,feature_dict= self.backbone(pc_center)

        p_green_R=self.rot_green(point_fea.permute(0,2,1))
        p_red_R=self.rot_red(point_fea.permute(0,2,1))
        feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
        T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)

        Pred_T = T + center.squeeze(1)  # bs x 3
        Pred_s = s  # this s is
        recon=recon+center



        gt_green_v, gt_red_v = get_gt_v(gt_R)
        gt_fsnet_list = {
            'Rotation': gt_R,
            'Recon': PC,
            'Tran': gt_t,
            'Size': gt_s,
        }
        pred_fsnet_list = {
            'Rot1': p_green_R,
            'Rot2': p_red_R,
            'Recon': recon,
            'Tran': Pred_T,
            'Size': Pred_s,
        }
        name_fs_list=gt_fsnet_list
        fsnet_loss=self.loss_fs_net(name_fs_list, pred_fsnet_list, gt_fsnet_list, sym)


        query_num=query_points.shape[1]
        batch_size=query_points.shape[0]
        query_points_center=query_points-center
        pred_nocs_bin=self.qnet(query_points_center,feature_dict).reshape(batch_size,query_num,3,FLAGS.bin_size).permute(0,-1,1,2).contiguous()
        class_mean_shape=mean_shape[0]
        max_class_mean_shape=class_mean_shape.max()

        cat_id=obj_id[0].item()+1
        ratio_x=cat_property[cat_id]['ratio'][0]
        ratio_y=cat_property[cat_id]['ratio'][1]
        ratio_z=cat_property[cat_id]['ratio'][2]
        base=cat_property[cat_id]['base']

        if sym[0][2]==1:
            y_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_y
            query_nocs[:,:,1]=torch.abs(query_nocs[:,:,1])
            y_start=0
        else:
            y_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_y
            y_start=(-FLAGS.pad_radius)*ratio_y

        if sym[0][0]==1 or base>2:
            if sym[0][4]==1:
                r_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_x
            else:
                r_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_z

            r_start=0

            if sym[0][0]==1:
                query_nocs_r=torch.norm(query_nocs[:,:,(0,2)],dim=-1)
                query_nocs[:,:,0]=query_nocs_r
                query_nocs[:,:,2]=0
                query_nocs_bin=torch.zeros_like(query_nocs).long()
                query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-r_start)/r_bin_resolution),0,FLAGS.bin_size-1).long()
                query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()

            else:
                theta_start=0
                query_nocs_r=torch.norm(query_nocs[:,:,(0,2)],dim=-1)
                whole_range=(2*math.pi/base)
                query_nocs_theta=torch.atan2(query_nocs[:,:,2],query_nocs[:,:,0])
                query_nocs_theta=(query_nocs_theta+2*math.pi)%(2*math.pi)%whole_range
                if sym[0][1] ==1 or sym[0][3]==1:
                    half_range=(math.pi)/base
                    query_nocs_theta=torch.abs(query_nocs_theta-half_range)
                    theta_bin_resolution=half_range/FLAGS.bin_size
                else:
                    query_nocs_theta=query_nocs_theta
                    theta_bin_resolution=whole_range/FLAGS.bin_size
                query_nocs[:,:,0]=query_nocs_r
                query_nocs[:,:,2]=query_nocs_theta
                query_nocs_bin=torch.zeros_like(query_nocs).long()
                query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-r_start)/r_bin_resolution),0,FLAGS.bin_size-1).long()
                query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
                query_nocs_bin[:,:,2]=torch.clamp(((query_nocs[:,:,2]-theta_start)/theta_bin_resolution),0,FLAGS.bin_size-1).long()
        else:
            if sym[0][1]==1:
                x_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_x
                query_nocs[:,:,0]=torch.abs(query_nocs[:,:,0])
                x_start=0
            else:
                x_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_x
                x_start=(-FLAGS.pad_radius)*ratio_x
            if sym[0][3]==1:
                z_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_z
                query_nocs[:,:,2]=torch.abs(query_nocs[:,:,2])
                z_start=0
            else:
                z_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_z
                z_start=(-FLAGS.pad_radius)*ratio_z
            query_nocs_bin=torch.zeros_like(query_nocs).long()
            query_nocs_bin[:,:,0]=torch.clamp(((query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,1]=torch.clamp(((query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
            query_nocs_bin[:,:,2]=torch.clamp(((query_nocs[:,:,2]-z_start)/z_bin_resolution),0,FLAGS.bin_size-1).long()


        denom=torch.ones_like(query_nocs_bin).sum()
        loss_nocs_theta=None
        if base>2:
            # inv_std=100
            # query_nocs_bin_theta=query_nocs_bin[:,:,2]
            # bins=torch.arange(FLAGS.bin_size).reshape(1,-1,1).repeat(batch_size,1,query_num).to(query_nocs.device)
            # distance=torch.abs(query_nocs_bin_theta.reshape(batch_size,1,query_num).repeat(1,FLAGS.bin_size,1)-bins)
            # smooth_target_theta=inv_std*torch.cos((distance)/(FLAGS.bin_size)*2*math.pi)
            # smooth_target_theta=nn.functional.softmax(smooth_target_theta,dim=1).float()
            # loss_nocs_theta=self.SoftCross(pred_nocs_bin[:,:,:,2],smooth_target_theta)
            loss_nocs_theta=self.loss_bin_fun(pred_nocs_bin[:,:,:,2],query_nocs_bin[:,:,2])
            loss_nocs=self.loss_bin_fun(pred_nocs_bin[:,:,:,:2],query_nocs_bin[:,:,:2])

        elif sym[0][0]==1:
            loss_nocs=self.loss_bin_fun(pred_nocs_bin[:,:,:,:2],query_nocs_bin[:,:,:2])
            denom=torch.ones_like(query_nocs_bin[:,:,:2]).sum()
        else:
            loss_nocs=self.loss_bin_fun(pred_nocs_bin,query_nocs_bin)
        loss_nocs=loss_nocs.sum()/denom*FLAGS.interpo_reg_w
        if base>2:
            loss_nocs_theta=loss_nocs_theta.sum()/denom*FLAGS.interpo_theta_w
        else:
            loss_nocs_theta=0
        interpo_loss={'Nocs':loss_nocs,'Nocs_theta':loss_nocs_theta}
        # show_open3d(pc_center[0].cpu().detach().numpy(),query_points_center[0].cpu().detach().numpy())
        loss_dict = {}
        loss_dict['fsnet_loss'] = fsnet_loss
        loss_dict['interpo_loss'] = interpo_loss
        return loss_dict




    def data_augment(self, PC, gt_R, gt_t, gt_s, mean_shape, sym, aug_bb, aug_rt_t, aug_rt_r,
                         model_point, nocs_scale, obj_ids):
        # augmentation
        bs = PC.shape[0]
        for i in range(bs):
            obj_id = int(obj_ids[i])
            prop_bb = torch.rand(1)
            if prop_bb < FLAGS.aug_bb_pro:
                #  R, t, s, s_x=(0.9, 1.1), s_y=(0.9, 1.1), s_z=(0.9, 1.1), sym=None
                PC_new, gt_s_new = cat_defor_3D_bb(PC[i, ...], gt_R[i, ...],
                                               gt_t[i, ...], gt_s[i, ...] + mean_shape[i, ...],
                                               sym=sym[i, ...], aug_bb=aug_bb[i, ...])
                gt_s_new = gt_s_new - mean_shape[i, ...]
                PC[i, ...] = PC_new
                gt_s[i, ...] = gt_s_new

            prop_rt = torch.rand(1)
            if prop_rt < FLAGS.aug_rt_pro:
                PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC[i, ...], gt_R[i, ...],
                                                         gt_t[i, ...], aug_rt_t[i, ...], aug_rt_r[i, ...])
                PC[i, ...] = PC_new
                gt_R[i, ...] = gt_R_new
                gt_t[i, ...] = gt_t_new.view(-1)

            prop_pc = torch.rand(1)
            r=0.00005
            PC[i, ...]+=torch.clamp(r*torch.randn(PC[i, ...].shape[0], 3).to(PC.device), -r, r)
        return PC, gt_R, gt_t, gt_s


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