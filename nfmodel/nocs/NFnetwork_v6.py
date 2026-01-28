import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net_v6 import GCN3D_segR,Rot_red,Rot_green,MyQNet,MyQNet_sampling,Pose_Ts,Point_center_res_cate, \
    VADLogVar,Decoder
curdir=os.path.dirname(os.path.realpath(__file__))
qnet_config_file=os.path.join(curdir,'qnet.yaml')
from network.point_sample.pc_sample import *
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from nfmodel.uti_tool import *
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from losses.nf_loss import *
from nnutils.torch_util import *
import torch.optim as optim
from nnutils.torch_pso import *

def KLD(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    KLD = torch.mean(KLD)
    return KLD



class NFPose(nn.Module):
    def __init__(self):
        super(NFPose, self).__init__()
        self.qnet=MyQNet(qnet_config_file)
        self.qnet_sampling=MyQNet_sampling(qnet_config_file)
        self.rot_green = Rot_green(F=FLAGS.feat_c_R,k=FLAGS.R_c)
        self.rot_red = Rot_red(F=FLAGS.feat_c_R,k=FLAGS.R_c)

        self.backbone1=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        if FLAGS.two_back:
            self.backbone2=GCN3D_segR(support_num= FLAGS.gcn_sup_num, neighbor_num= FLAGS.gcn_n_num)
        if FLAGS.feat_for_ts:
            self.ts=Pose_Ts(F=FLAGS.feat_c_ts,k=FLAGS.Ts_c)
        else:
            self.ts=Point_center_res_cate()
        self.loss_fs_net = fs_net_loss()
        self.loss_consistency=consistency_loss()
        self.loss_inter=inter_loss()
        self.loss_coord=nn.SmoothL1Loss(beta=0.5,reduction='mean')
        # self.loss_coord=nn.MSELoss(reduction='mean')
        self.loss_coord_sym=nn.SmoothL1Loss(beta=0.5,reduction='none')
        self.loss_bin_fun=nn.CrossEntropyLoss(reduce=False)



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

        pad_nocs=pad_points
        max_real_scale=torch.max(real_scale,dim=-1)[0]*FLAGS.scale_ratio
        pad_nocs=pad_nocs*(max_real_scale.reshape(-1,1,1))/(real_scale.reshape(-1,1,3))

        query_nocs=sdf_points
        query_num=query_nocs.shape[1]

        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        bs = PC.shape[0]

        recon,point_fea,global_fea,feature_dict,feature_dict_detach= self.backbone1(pc_center)






        query_nocs_deform=query_nocs
        pad_nocs_deform=pad_nocs


        gt_pad_camera=torch.bmm((pad_nocs.detach()*real_scale.reshape(-1,1,3)),gt_R.permute(0,2,1))+gt_t.reshape(-1,1,3)-center.reshape(-1,1,3).detach()

        # show_open3d(gt_pad_camera[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())

        pad_bin_first,pad_bin_value_first=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size//10,pad_nocs)
        pad_bin_first_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size//10,pad_nocs_deform)
        pad_bin_second,pad_bin_value_second=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size,pad_nocs)
        pad_bin_second_deform,_=self.to_bin(cat_name,sym[0][0],FLAGS.bin_size,pad_nocs_deform)

        pred_coord=self.qnet(gt_pad_camera,feature_dict)['coord']
        Loss_nocs_fun=nn.SmoothL1Loss(size_average=False,reduce=False)
        if sym[0][0]==1:
            pad_nocs_r=pad_nocs.clone()
            pad_nocs_r[:,:,0]=torch.norm(pad_nocs_r[:,:,(0,2)],dim=-1)
            pad_nocs_r[:,:,2]=0
            loss_nocs_original=Loss_nocs_fun(pred_coord[:,:,:2],pad_nocs_r[:,:,:2]).mean(-1)
        else:
            loss_nocs_original=Loss_nocs_fun(pred_coord,pad_nocs).mean(-1)
        if FLAGS.use_sampling_train:

            pred_binary=self.qnet_sampling(gt_pad_camera,feature_dict)['binary']
            pred_mask=F.gumbel_softmax(pred_binary,hard=True,dim=-1)[:,:,0]
            pred_mask_sum=torch.sum(pred_mask,dim=-1)
            pred_mask_ratio=(pred_mask_sum/pred_mask.shape[1]).mean()

            pad_nocs_dis=torch.exp(-torch.norm(pad_nocs.unsqueeze(1)-pad_nocs.unsqueeze(2),dim=-1)**2)

            diag=torch.ones_like(pad_nocs_dis[0])
            diag=diag-torch.diag(torch.diag(diag))
            pad_nocs_dis=pad_nocs_dis*diag.unsqueeze(0)
            pad_nocs_dis=pad_nocs_dis*pred_mask.unsqueeze(1)
            pad_nocs_dis=torch.log(torch.sum(pad_nocs_dis,dim=-1)/pred_mask_sum[:,None].detach())
            loss_dis=(torch.sum(pad_nocs_dis*pred_mask,dim=-1)/pred_mask_sum.detach()).mean()



            loss_nocs_normal=loss_nocs_original.mean(-1).mean(-1).detach()
            loss_nocs=(torch.sum(loss_nocs_original*(pred_mask.detach()),dim=-1)/(pred_mask_sum.detach())).mean()
            loss_sampling=-(torch.sum((loss_nocs_original.detach())*pred_mask,dim=-1)/pred_mask_sum.detach()).mean()
            loss_ratio=(pred_mask_ratio-FLAGS.mask_ratio)**2
        else:
            loss_nocs=loss_nocs_original.mean(-1).mean(-1)
            loss_nocs_normal=0
            loss_sampling=0
            loss_ratio=0
            loss_dis=0


        if FLAGS.use_fsnet:
            name_fs_list=['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular', 'Tran', 'Size']
            pred_fsnet_list=self.forward_fsnet(point_fea,pc_center,center)
            gt_green_v, gt_red_v = get_gt_v(gt_R)
            gt_fsnet_list = {
                'Rot1': gt_green_v,
                'Rot2': gt_red_v,
                'Tran': gt_t,
                'Size': gt_s,
            }
            fsnet_loss=self.loss_fs_net(name_fs_list,pred_fsnet_list,gt_fsnet_list,sym)
        else:
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
        loss_dict['interpo_loss']={'Nocs':loss_nocs,'normal':loss_nocs_normal,'sampling':loss_sampling,
                                   'ratio':loss_ratio*0.1,'dis':loss_dis*0.1}
        loss_dict['fsnet_loss'] = fsnet_loss
        return loss_dict




    def to_bin(self,cat_name,sym,bin_size,pad_nocs):
        ratio_x=ratio_dict[cat_name][0]
        ratio_y=ratio_dict[cat_name][1]
        ratio_z=ratio_dict[cat_name][2]
        pad_nocs_r=pad_nocs.clone()
        if sym==1:
            x_bin_resolution=FLAGS.pad_radius/bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
            x_start=0
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=0
            z_bin_resolution=0
            pad_nocs_r[:,:,0]=torch.norm(pad_nocs_r[:,:,(0,2)],dim=-1)
            pad_nocs_r[:,:,2]=0
            pad_nocs_bin=torch.zeros_like(pad_nocs_r).long()
            pad_nocs_bin[:,:,0]=torch.clamp(((pad_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,1]=torch.clamp(((pad_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
        else:
            x_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_x
            y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
            z_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_z
            x_start=(-FLAGS.pad_radius)*ratio_x
            y_start=(-FLAGS.pad_radius)*ratio_y
            z_start=(-FLAGS.pad_radius)*ratio_z
            pad_nocs_bin=torch.zeros_like(pad_nocs_r).long()
            pad_nocs_bin[:,:,0]=torch.clamp(((pad_nocs_r[:,:,0]-x_start)/x_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,1]=torch.clamp(((pad_nocs_r[:,:,1]-y_start)/y_bin_resolution),0,bin_size-1).long()
            pad_nocs_bin[:,:,2]=torch.clamp(((pad_nocs_r[:,:,2]-z_start)/z_bin_resolution),0,bin_size-1).long()
        pad_bin_value = torch.zeros((3,bin_size)).to(pad_nocs_r.device)
        pad_bin_value[0]=x_start+torch.arange(bin_size)*x_bin_resolution
        pad_bin_value[1]=y_start+torch.arange(bin_size)*y_bin_resolution
        pad_bin_value[2]=z_start+torch.arange(bin_size)*z_bin_resolution
        return pad_nocs_bin,pad_bin_value


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

