import numpy as np
import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts
from network.point_sample.pc_sample import PC_sample
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc
from nfmodel.uti_tool import *
from nfmodel.cat.CatNFnetwork import NFPose
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from EQNet.eqnet.ops.knn.knn_utils import knn_query
from EQNet.eqnet.ops.grouping.grouping_utils import grouping_operation
from easydict import EasyDict
class multi_NFPose(nn.Module):
    def __init__(self,cat_names):
        super(multi_NFPose, self).__init__()
        self.per_networks=nn.ModuleDict()
        self.cat_names=cat_names
        for cat_name in cat_names:
            cat_model_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
            cat_model=NFPose()
            cat_model.load_state_dict(torch.load(cat_model_path))
            self.per_networks[cat_name]=cat_model

    def forward(self, depth, cat_id_0base, camK, gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None,def_mask=None, model_point=None, nocs_scale=None,rgb=None,gt_mask=None):
        # FLAGS.sample_method = 'basic'
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None
        PC = PC_sample(def_mask, depth, camK, gt_2D)

        PC = PC.detach()
        # show_open3d(PC[0].cpu().detach().numpy(),PC[0].cpu().detach().numpy())
        obj_num=PC.shape[0]
        res_list=[]
        scale_list=[]
        for i in range(obj_num):
            cat_name=self.cat_names[cat_id_0base[i]]
            res,scale=self.per_infer(cat_name,cat_id_0base[i].item(),sym[i],PC[i],mean_shape[i])
            res_list.append(res)
            scale_list.append(scale)
        return torch.stack(res_list,dim=0),torch.stack(scale_list,dim=0)

    def per_infer(self,cat_name,cat_id0,sym,pc,mean_shape):
        # show_open3d(pc.detach().cpu().numpy(),pc.detach().cpu().numpy())
        cat_model=self.per_networks[cat_name]
        PC=pc.unsqueeze(0)

        # r=0.00005
        # PC[0, ...]+=torch.clamp(r*torch.randn(PC[0, ...].shape[0], 3).to(PC.device), -r, r)

        mean_shape=mean_shape.detach().cpu().numpy()

        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center

        show_points=pc_center[0].detach().cpu().numpy()

        recon,point_fea,global_fea,feature_dict= cat_model.backbone(pc_center)

        p_green_R=cat_model.rot_green(point_fea.permute(0,2,1))
        p_red_R=cat_model.rot_red(point_fea.permute(0,2,1))
        feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
        T, s = cat_model.ts(feat_for_ts.permute(0, 2, 1))


        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)

        Pred_T = T  # bs x 3  # bs x 3
        Pred_s = s  # this s is



        p_green_R =p_green_R[0].detach().cpu().numpy()
        p_red_R=p_red_R[0].detach().cpu().numpy()
        Pred_T=Pred_T[0].detach().cpu().numpy()
        Pred_s=Pred_s[0].detach().cpu().numpy()+mean_shape




        if sym[0] < 1 :
            num_cor=3
            cor0 = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
        else:
            num_cor=2
            cor0 = np.array([[0, 0, 0], [0, 1, 0]])
        cor0= cor0/np.linalg.norm(cor0)
        pred_axis = np.zeros((num_cor,3))
        pred_axis[1,:]=p_green_R
        if num_cor==3:
            pred_axis[2,:]=p_red_R
        pose=gettrans(cor0.reshape((num_cor, 3)), pred_axis.reshape((num_cor, 1, 3)))
        fake_rotation = pose[0][0:3, 0:3]

        if FLAGS.only_backbone:
            res = torch.eye(4, dtype=torch.float).to(center.device)
            res[:3,:3]=torch.from_numpy(fake_rotation)
            res[:3,3]=torch.from_numpy(Pred_T).to(center.device)+center.reshape(3)
            no_sym_s=torch.from_numpy(Pred_s).to(center.device)
            return res,no_sym_s

        if FLAGS.use_mean_init:
            Pred_T = np.zeros_like(Pred_T)
            Pred_s=mean_shape


        fake_grid = grids['fake_grid'].numpy()
        boxsize = FLAGS.pad_radius*2
        fake_grid = boxsize * (fake_grid)
        fake_grid_scaled=fake_grid*mean_shape.max()
        fake_query_np=fake_grid_scaled @ fake_rotation.T + Pred_T
        # show_open3d(pc_center[0].detach().cpu().numpy(),fake_query_np)
        fake_query=torch.from_numpy(fake_query_np).cuda().unsqueeze(0).float()




        new_query_num=500
        box_size = FLAGS.query_radius*2
        points_uniform = np.random.rand(new_query_num, 3)-0.5
        # points_uniform=grids['new_grid'].numpy()

        points_uniform = box_size * points_uniform
        new_query_nocs=torch.from_numpy(points_uniform).float().unsqueeze(0).cuda().detach()
        new_query_num=new_query_nocs.shape[1]
        fake_query=fake_query.detach()

        cat_id=cat_id0+1
        ratio_x=cat_property[cat_id]['ratio'][0]
        ratio_y=cat_property[cat_id]['ratio'][1]
        ratio_z=cat_property[cat_id]['ratio'][2]
        base=cat_property[cat_id]['base']
        new_query_nocs_euler=new_query_nocs.clone()


        def get_bin(input_nocs):
            new_query_nocs=input_nocs.clone()
            if sym[2]==1:
                y_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_y
                new_query_nocs[:,:,1]=torch.abs(new_query_nocs[:,:,1])
                y_start=0
            else:
                y_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_y
                y_start=(-FLAGS.pad_radius)*ratio_y

            if sym[0]==1 or base>2:
                if sym[4]==1:
                    r_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_x
                else:
                    r_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_z
                r_start=0
                if sym[0]==1:
                    new_query_nocs_r=torch.norm(new_query_nocs[:,:,(0,2)],dim=-1)
                    new_query_nocs[:,:,0]=new_query_nocs_r
                    new_query_nocs[:,:,2]=0
                    new_query_nocs_bin=torch.zeros_like(new_query_nocs)
                    new_query_nocs_bin[:,:,0]=torch.clamp(((new_query_nocs[:,:,0]-r_start)/r_bin_resolution),0,FLAGS.bin_size-1)
                    new_query_nocs_bin[:,:,1]=torch.clamp(((new_query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1)

                else:
                    theta_start=0
                    new_query_nocs_r=torch.norm(new_query_nocs[:,:,(0,2)],dim=-1)
                    whole_range=(2*math.pi/base)

                    new_query_nocs_theta=torch.atan2(new_query_nocs[:,:,2],new_query_nocs[:,:,0])
                    new_query_nocs_theta=(new_query_nocs_theta+2*math.pi)%(2*math.pi)%whole_range
                    if sym[1] ==1 or sym[3]==1:
                        half_range=(math.pi)/base
                        query_nocs_theta=torch.abs(new_query_nocs_theta-half_range)
                        theta_bin_resolution=half_range/FLAGS.bin_size
                    else:
                        new_query_nocs_theta=new_query_nocs_theta
                        theta_bin_resolution=whole_range/FLAGS.bin_size
                    new_query_nocs[:,:,0]=new_query_nocs_r
                    new_query_nocs[:,:,2]=new_query_nocs_theta
                    new_query_nocs_bin=torch.zeros_like(new_query_nocs)
                    new_query_nocs_bin[:,:,0]=torch.clamp(((new_query_nocs[:,:,0]-r_start)/r_bin_resolution),0,FLAGS.bin_size-1)
                    new_query_nocs_bin[:,:,1]=torch.clamp(((new_query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1)
                    new_query_nocs_bin[:,:,2]=torch.clamp(((new_query_nocs[:,:,2]-theta_start)/theta_bin_resolution),0,FLAGS.bin_size-1)
            else:
                if sym[1]==1:
                    x_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_x
                    new_query_nocs[:,:,0]=torch.abs(new_query_nocs[:,:,0])
                    x_start=0
                else:
                    x_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_x
                    x_start=(-FLAGS.pad_radius)*ratio_x
                if sym[3]==1:
                    z_bin_resolution=FLAGS.pad_radius/FLAGS.bin_size*ratio_z
                    new_query_nocs[:,:,2]=torch.abs(new_query_nocs[:,:,2])
                    z_start=0
                else:
                    z_bin_resolution=2*FLAGS.pad_radius/FLAGS.bin_size*ratio_z
                    z_start=(-FLAGS.pad_radius)*ratio_z
                new_query_nocs_bin=torch.zeros_like(new_query_nocs)
                new_query_nocs_bin[:,:,0]=torch.clamp(((new_query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1)
                new_query_nocs_bin[:,:,1]=torch.clamp(((new_query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1)
                new_query_nocs_bin[:,:,2]=torch.clamp(((new_query_nocs[:,:,2]-z_start)/z_bin_resolution),0,FLAGS.bin_size-1)
            return new_query_nocs_bin

        new_query_nocs_bin_float=get_bin(new_query_nocs)
        new_query_nocs_bin=new_query_nocs_bin_float.long()
        pred_fake_nocs_bin=cat_model.qnet(fake_query,feature_dict).reshape(1,-1,3,FLAGS.bin_size).detach()


        # dis_numpy=nn.functional.softmax(pred_fake_nocs_bin,dim=-1).cpu().numpy()
        # vis_index=2500
        # plt_y=dis_numpy[0,vis_index,2,:]
        # plt_x=np.arange(FLAGS.bin_size)
        # plt.plot(plt_x,plt_y)
        # plt.savefig('tmp.png')
        # plt.close()




        lr=0.001
        if sym[4]==1:
            sym_s=Pred_s[:2]
        else:
            sym_s=Pred_s
        new_Rvec=torch.from_numpy(cv2.Rodrigues(fake_rotation)[0][:,0]).float().cuda().requires_grad_()
        new_T=torch.from_numpy(Pred_T).float().cuda().requires_grad_()
        new_scale=torch.from_numpy(sym_s).float().cuda().requires_grad_()
        opt = torch.optim.Adam([{'params':new_scale},{'params':new_Rvec},{'params':new_T}], lr=lr)


        show_dict=EasyDict()
        show_dict.show_new_query=None
        show_dict.ori_show_new_query=None
        m=torch.nn.Softmax(dim=-1)
        fake_dis_log=m(pred_fake_nocs_bin)


        #
        # def visual_neighbor(query_point,support_points,num_neighber):
        #     query_cnt=query_point.new_zeros(1).int()
        #     query_cnt[0]=1
        #     support_cnt=support_points.new_zeros(1).int()
        #     support_cnt[0]=support_points.shape[0]
        #
        #     index_pair = knn_query(
        #         num_neighber,
        #         support_points, support_cnt,
        #         query_point, query_cnt).int()
        #     neighbor_pos=grouping_operation(
        #         support_points, support_cnt, index_pair, query_cnt).permute(0,2,1)
        #     show_open3d(query_point.cpu().numpy(),support_points.cpu().numpy(),neighbor_pos[0].cpu().numpy())
        #     return
        #
        # support0=feature_dict['conv0']['pos']
        # support2=feature_dict['conv2']['pos']
        # support4=feature_dict['conv4']['pos']
        # visual_nocs_point=np.array([0.0,0.1,0])
        # visual_nocs_point=torch.from_numpy(visual_nocs_point).cuda().float().unsqueeze(0)
        # Rt=fake_rotation
        # res=center.detach().cpu().numpy().reshape(3)
        # support0_nocs=torch.from_numpy(((support0[0].cpu().numpy()) @ Rt)).cuda().float()/(mean_shape.max())
        # support2_nocs=torch.from_numpy(((support2[0].cpu().numpy()) @ Rt)).cuda().float()/(mean_shape.max())
        # support4_nocs=torch.from_numpy(((support4[0].cpu().numpy()) @ Rt)).cuda().float()/(mean_shape.max())
        # visual_neighbor(visual_nocs_point,support0_nocs,16)
        # visual_neighbor(visual_nocs_point,support2_nocs,16)
        # visual_neighbor(visual_nocs_point,support4_nocs,16)
        # print('save query!!')

        def objective(new_scale,new_R,new_T):
            flag=False
            if type(new_scale) is np.ndarray:
                flag=True
                new_scale=torch.from_numpy(new_scale).float().cuda()
            if type(new_R) is np.ndarray:
                flag=True
                new_R=torch.from_numpy(new_R).float().cuda()
            if type(new_T) is np.ndarray:
                flag=True
                new_T=torch.from_numpy(new_T).float().cuda()
            if sym[4]==1:
                tmp_new_scale=torch.zeros(3,device=new_scale.device)
                tmp_new_scale[:2]=new_scale
                tmp_new_scale[2]=new_scale[0]
                new_scale=tmp_new_scale
            # if FLAGS.use_scene:
            #     fake_query_nocs = (fake_query-new_T) @ new_R / new_scale
            #     fake_query_nocs_bin_float=get_bin(fake_query_nocs)
            #     fake_query_nocs_bin_floor=fake_query_nocs_bin_float.floor().long()
            #     fake_query_nocs_bin_ceil=fake_query_nocs_bin_float.ceil().long()
            #     w1=fake_query_nocs_bin_float-fake_query_nocs_bin_floor
            #     w2=fake_query_nocs_bin_ceil-fake_query_nocs_bin_float
            #     scene_dis_floor=torch.gather(fake_dis_log,-1,fake_query_nocs_bin_floor.unsqueeze(-1)).squeeze(-1)
            #     scene_dis_ceil=torch.gather(fake_dis_log,-1,fake_query_nocs_bin_ceil.unsqueeze(-1)).squeeze(-1)
            #     scene_dis=w1*scene_dis_floor+w2*scene_dis_ceil
            #     scene_dis_log=torch.log(scene_dis+1e-5)[0]
            #     if sym[0]==1:
            #         scene_dis_log=(scene_dis_log[:,0]+scene_dis_log[:,1])
            #     else:
            #         scene_dis_log=(scene_dis_log[:,0]+scene_dis_log[:,1]+scene_dis_log[:,2])
            #     scene_score=-scene_dis_log.mean()
            new_query=(new_query_nocs_euler*new_scale) @ new_R.T + new_T
            if show_dict.ori_show_new_query is None:
                show_dict.ori_show_new_query=new_query[0].detach().cpu().numpy().copy()
            show_dict.show_new_query=new_query[0].detach().cpu().numpy().copy()

            # if count[0]%5==0:
            #     show_open3d(show_dict.show_new_query,show_points,fake_query_np)
            query_cnt=new_query.new_zeros(1).int()
            query_cnt[0]=new_query.shape[1]
            fake_cnt=fake_query.new_zeros(1).int()
            fake_cnt[0]=fake_query.shape[1]

            index_pair = knn_query(
                8,
                fake_query[0], fake_cnt,
                new_query[0], query_cnt).int()
            neighbor_pos=grouping_operation(
                fake_query[0], fake_cnt, index_pair, query_cnt).permute(0,2,1)

            neighbor_dis_log=grouping_operation(
                fake_dis_log.reshape(fake_cnt,3*FLAGS.bin_size), fake_cnt, index_pair, query_cnt).permute(0,2,1)

            weight=1/(torch.norm((new_query[0].unsqueeze(1)-neighbor_pos),dim=-1)+1e-5)
            denomin=torch.sum(weight,dim=-1).unsqueeze(-1)
            weight=weight/(denomin+1e-5)

            # weight=m(-torch.norm((new_query[0].unsqueeze(1)-neighbor_pos),dim=-1))
            query_dis_log=torch.sum(neighbor_dis_log*weight.unsqueeze(-1),dim=1).reshape(1,query_cnt,3,FLAGS.bin_size)
            query_dis_log=torch.log(query_dis_log+1e-5)

            pred_prob=torch.gather(query_dis_log,-1,new_query_nocs_bin.unsqueeze(-1))[0,:,:,0]
            if sym[0]==1:
                pred_prob=(pred_prob[:,0]+pred_prob[:,1])
            else:
                # pred_prob=(pred_prob[:,0]+pred_prob[:,1]+pred_prob[:,2])
                pred_prob=(pred_prob[:,0]+pred_prob[:,1]+pred_prob[:,2])
            score=-pred_prob.mean()
            if torch.isnan(score):
                print('nan')
            if flag:
                return score.item()
            else:
                return score

        torch.autograd.set_detect_anomaly(False)
        with torch.enable_grad():
            for i in range(50):

                cur_scale=new_scale
                cur_T=new_T
                cur_R=Rodrigues.apply(new_Rvec)
                cur_loss=objective(cur_scale,cur_R,cur_T)

                opt.zero_grad()
                cur_loss.backward()
                # print(new_scale.grad)
                # torch.nn.utils.clip_grad_norm_([new_Rvec,new_scale,new_T], 1)
                # print(new_scale.grad)
                if i<10:
                    opt.param_groups[0]['lr'] = lr*0.1
                    opt.param_groups[1]['lr'] = lr*0.1
                    opt.param_groups[2]['lr'] = lr*0.1
                else:
                    opt.param_groups[0]['lr'] = lr
                    opt.param_groups[1]['lr'] = lr
                    opt.param_groups[2]['lr'] = lr
                opt.step()


        res = torch.eye(4, dtype=torch.float).to(new_scale.device)


        res[:3,:3]=cur_R
        res[:3,3]=cur_T+center.reshape(3)
        if sym[4]==1:
            no_sym_s=torch.zeros(3,device=new_scale.device)
            no_sym_s[:2]=cur_scale
            no_sym_s[2]=cur_scale[0]
        else:
            no_sym_s=cur_scale


        show_open3d(show_dict.ori_show_new_query,show_points)
        show_open3d(show_dict.show_new_query,show_points)
        return res,no_sym_s