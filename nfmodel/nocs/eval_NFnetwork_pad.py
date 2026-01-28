import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from nfmodel.part_net import GCN3D_segR,Rot_red,Rot_green,MyQNet,Pose_Ts
from network.point_sample.pc_sample import PC_sample
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc
from nfmodel.uti_tool import *
from nfmodel.nocs.NFnetwork import NFPose
from tools.training_utils import get_gt_v
from losses.fs_net_loss import fs_net_loss
from EQNet.eqnet.ops.knn.knn_utils import knn_query
from EQNet.eqnet.ops.grouping.grouping_utils import grouping_operation
from easydict import EasyDict
import torch.optim as optim
from tools.torch_utils.solver.ranger2020 import Ranger

class multi_NFPose(nn.Module):
    def __init__(self,cat_names):
        super(multi_NFPose, self).__init__()
        self.per_networks=nn.ModuleDict()
        self.cat_names=cat_names
        for cat_name in cat_names:
            cat_model_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
            cat_model=NFPose()
            cat_model.load_state_dict(torch.load(cat_model_path))
            cat_model.eval()
            self.per_networks[cat_name]=cat_model

    def forward(self, depth, cat_id_0base, camK, gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None,def_mask=None, model_point=None, nocs_scale=None,rgb=None,gt_mask=None):
        # FLAGS.sample_method = 'basic'
        bs = depth.shape[0]
        H, W = depth.shape[2], depth.shape[3]
        sketch = torch.rand([bs, 6, H, W], device=depth.device)
        obj_mask = None
        PC = PC_sample(def_mask, depth, camK, gt_2D)

        PC = PC.detach()
        obj_num=PC.shape[0]
        res_list=[]
        scale_list=[]
        for i in range(obj_num):
            if FLAGS.per_obj is not '':
                cat_name=FLAGS.per_obj
            else:
                cat_name=self.cat_names[cat_id_0base[i]]
            res,scale=self.per_infer(cat_name,sym[i],PC[i],mean_shape[i],gt_R[i],gt_t[i],gt_s[i])
            res_list.append(res)
            scale_list.append(scale)
        return torch.stack(res_list,dim=0),torch.stack(scale_list,dim=0)

    def per_infer(self,cat_name,sym,pc,mean_shape,gt_R,gt_T,gt_s):
        # show_open3d(pc.detach().cpu().numpy(),pc.detach().cpu().numpy())
        cat_model=self.per_networks[cat_name]
        PC=pc.unsqueeze(0)
        mean_shape=mean_shape.detach().cpu().numpy()

        center=PC.mean(dim=1,keepdim=True)
        pc_center=PC-center
        show_points=pc_center[0].detach().cpu().numpy()
        gt_T=gt_T-center.squeeze()
        pc_num=PC.shape[1]
        recon,point_fea,global_fea,feature_dict= cat_model.backbone(pc_center)
        # point_fea=torch.cat([point_fea,global_fea.unsqueeze(1).repeat(1,pc_num,1)],dim=-1)
        p_green_R=cat_model.rot_green(point_fea.permute(0,2,1))
        p_red_R=cat_model.rot_red(point_fea.permute(0,2,1))
        if FLAGS.feat_for_ts:
            feat_for_ts = torch.cat([point_fea, pc_center], dim=2)
            T, s = cat_model.ts(feat_for_ts.permute(0, 2, 1))
        else:
            feat_for_ts = pc_center
            objs=torch.zeros_like(feat_for_ts[:,0,0])
            T, s = cat_model.ts(feat_for_ts.permute(0, 2, 1),objs)


        p_green_R = p_green_R / (torch.norm(p_green_R, dim=1, keepdim=True) + 1e-6)
        p_red_R = p_red_R / (torch.norm(p_red_R, dim=1, keepdim=True) + 1e-6)
        Pred_T = T  # bs x 3
        Pred_s = s  # this s is


        p_green_R =p_green_R[0].detach().cpu().numpy()
        p_red_R=p_red_R[0].detach().cpu().numpy()
        Pred_T=Pred_T[0].detach().cpu().numpy()
        if FLAGS.res_scale:
            Pred_s=Pred_s[0].detach().cpu().numpy()+mean_shape
        else:
            Pred_s=Pred_s[0].detach().cpu().numpy()


        if FLAGS.use_mean_init:
            Pred_T = np.zeros_like(Pred_T)
            Pred_s=mean_shape

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

        fake_grid = grids['fake_grid'].numpy()
        boxsize = FLAGS.fake_radius*2
        fake_grid = boxsize * (fake_grid)
        fake_grid_scaled=fake_grid*mean_shape.max()
        fake_nocs=torch.from_numpy(fake_grid).cuda().unsqueeze(0).float()
        fake_query_np=fake_grid_scaled @ fake_rotation.T + Pred_T
        # fake_grid_scaled=fake_grid*(pre.cpu().numpy()).max()
        # fake_query_np=fake_grid_scaled @ (gt_R.cpu().numpy()).T + (gt_T.cpu().numpy())
        # show_open3d(pc_center[0].detach().cpu().numpy(),fake_query_np)
        fake_query=torch.from_numpy(fake_query_np).cuda().unsqueeze(0).float()


        new_query_num=FLAGS.new_grid_num
        box_size = FLAGS.query_radius*2
        points_uniform = np.random.rand(new_query_num, 3)
        points_uniform = box_size * (points_uniform - 0.5)
        new_query_nocs=torch.from_numpy(points_uniform).float().unsqueeze(0).cuda().detach()
        new_query_num=new_query_nocs.shape[1]





        ratio_x=ratio_dict[cat_name][0]
        ratio_y=ratio_dict[cat_name][1]
        ratio_z=ratio_dict[cat_name][2]
        if sym[0]==1:
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

        bin_value = torch.zeros((3,FLAGS.bin_size)).to(fake_query.device)
        bin_value[0]=x_start+torch.arange(FLAGS.bin_size)*x_bin_resolution
        bin_value[1]=y_start+torch.arange(FLAGS.bin_size)*y_bin_resolution
        bin_value[2]=z_start+torch.arange(FLAGS.bin_size)*z_bin_resolution


        pred_fake_nocs_bin=cat_model.qnet(fake_query,feature_dict).reshape(1,-1,3,FLAGS.bin_size).detach()
        fake_bin=torch.max(pred_fake_nocs_bin,dim=-1)[1]
        pred_fake_nocs=fake_bin.clone().float()
        pred_fake_nocs[:,:,0]=fake_bin[:,:,0]*x_bin_resolution+x_start
        pred_fake_nocs[:,:,1]=fake_bin[:,:,1]*y_bin_resolution+y_start
        pred_fake_nocs[:,:,2]=fake_bin[:,:,2]*z_bin_resolution+z_start





        show_dict=EasyDict()
        show_dict.show_new_query=None
        show_dict.ori_show_new_query=None
        m=torch.nn.LogSoftmax(dim=-1)
        fake_dis_log=m(pred_fake_nocs_bin).detach()

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
            if sym[0]==1:
                tmp_new_scale=torch.zeros(3,device=new_scale.device)
                tmp_new_scale[:2]=new_scale
                tmp_new_scale[2]=new_scale[0]
                new_scale=tmp_new_scale

            new_query=(new_query_nocs*new_scale) @ new_R.T + new_T
            if show_dict.ori_show_new_query is None:
                show_dict.ori_show_new_query=new_query[0].detach().cpu().numpy().copy()
            show_dict.show_new_query=new_query[0].detach().cpu().numpy().copy()

            cal_fake_nocs=torch.bmm(fake_query-new_T.reshape(-1,1,3),new_R.reshape(-1,3,3))/new_scale.reshape(-1,1,3)
            if FLAGS.use_prob:
                if sym[0]==1:
                    cal_fake_nocs_r=cal_fake_nocs
                    cal_fake_nocs_r[:,:,0]=torch.norm(cal_fake_nocs[:,:,(0,2)],dim=-1)
                    distance=cal_fake_nocs_r.unsqueeze(-1)-bin_value
                else:
                    distance=cal_fake_nocs.unsqueeze(-1)-bin_value
                gamma=100
                smooth_target=nn.functional.softmax(-distance**2*gamma,dim=-1)

                pad_log_prob=torch.sum(fake_dis_log*smooth_target,dim=-1)
                if sym[0]==1:
                    log_prob=-pad_log_prob[:,:,:2].sum(-1).mean()
                else:
                    log_prob=-pad_log_prob.sum(-1).mean()
            else:
                log_prob=0

            if FLAGS.use_scene:
                if sym[0]==0:
                    distance=torch.norm(pred_fake_nocs - cal_fake_nocs, dim=-1)
                else:
                    distance=torch.norm(pred_fake_nocs[:,:,:2] - cal_fake_nocs[:,:,:2], dim=-1)
                scene_score=-torch.exp(-distance**2).mean()

            score=log_prob+scene_score
            if torch.isnan(score):
                print('nan')
            if flag:
                return score.item()
            else:
                return score


        lr=FLAGS.refine_lr
        if sym[0]==1:
            sym_s=Pred_s[:2]
        else:
            sym_s=Pred_s
        new_Rvec=torch.from_numpy(cv2.Rodrigues(fake_rotation)[0][:,0]).float().cuda().requires_grad_()
        new_T=torch.from_numpy(Pred_T).float().cuda().requires_grad_()
        new_scale=torch.from_numpy(sym_s).float().cuda().requires_grad_()


        if FLAGS.optimizer=='adam':
            params=[{'params':new_scale},{'params':new_Rvec},{'params':new_T}]
            opt = torch.optim.Adam(params, lr=lr)
            # opt=Ranger(params=params, lr=lr, weight_decay=0.1)
            # torch.autograd.set_detect_anomaly(False)
            with torch.enable_grad():
                cur_scale=new_scale
                cur_T=new_T
                # tmp_Rvec=torch.stack([new_Rvec,new_Rvec],dim=0)
                # R, jac = cv2.Rodrigues(tmp_Rvec.detach().cpu().numpy())
                cur_R=Rodrigues.apply(new_Rvec)
                for i in range(FLAGS.refine_step):
                    cur_scale=new_scale
                    cur_T=new_T
                    cur_R=Rodrigues.apply(new_Rvec)
                    cur_loss=objective(cur_scale,cur_R,cur_T)

                    opt.zero_grad()
                    cur_loss.backward()
                    # print(new_scale.grad)
                    # torch.nn.utils.clip_grad_norm_([new_Rvec,new_scale,new_T], 10)
                    # print(new_scale.grad)
                    if i<10:
                        opt.param_groups[0]['lr'] = lr
                        opt.param_groups[1]['lr'] = lr
                        opt.param_groups[2]['lr'] = lr
                    else:
                        opt.param_groups[0]['lr'] = lr
                        opt.param_groups[1]['lr'] = lr
                        opt.param_groups[2]['lr'] = lr
                    opt.step()
        elif FLAGS.optimizer=='newton':
            opt=optim.LBFGS([new_scale,new_Rvec,new_T],lr=FLAGS.refine_lr,max_eval=FLAGS.refine_step,
                            line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                cur_scale=new_scale
                cur_T=new_T
                cur_R=Rodrigues.apply(new_Rvec)
                cur_loss=objective(cur_scale,cur_R,cur_T)
                # print(cur_loss)
                cur_loss.backward()
                return cur_loss
            # for i in range(FLAGS.refine_step):
            #     opt.step(closure)
            opt.step(closure)
            cur_scale=new_scale
            cur_T=new_T
            cur_R=Rodrigues.apply(new_Rvec)


        res = torch.eye(4, dtype=torch.float).to(new_scale.device)
        res[:3,:3]=cur_R
        res[:3,3]=cur_T+center.reshape(3)

        if sym[0]==1:
            no_sym_s=torch.zeros(3,device=new_scale.device)
            no_sym_s[:2]=cur_scale
            no_sym_s[2]=cur_scale[0]
        else:
            no_sym_s=cur_scale

        # show_open3d(show_dict.ori_show_new_query,show_points)
        # show_open3d(show_dict.show_new_query,show_points)
        return res,no_sym_s