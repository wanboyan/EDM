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
        fake_query_np=fake_grid_scaled @ fake_rotation.T + Pred_T
        # fake_grid_scaled=fake_grid*(pre.cpu().numpy()).max()
        # fake_query_np=fake_grid_scaled @ (gt_R.cpu().numpy()).T + (gt_T.cpu().numpy())
        # show_open3d(pc_center[0].detach().cpu().numpy(),fake_query_np)
        fake_query=torch.from_numpy(fake_query_np).cuda().unsqueeze(0).float()

        def visual_neighbor(query_point,support_points,num_neighber):
            query_cnt=query_point.new_zeros(1).int()
            query_cnt[0]=1
            support_cnt=support_points.new_zeros(1).int()
            support_cnt[0]=support_points.shape[0]

            index_pair = knn_query(
                num_neighber,
                support_points, support_cnt,
                query_point, query_cnt).int()
            neighbor_pos=grouping_operation(
                support_points, support_cnt, index_pair, query_cnt).permute(0,2,1)
            show_open3d(query_point.cpu().numpy(),support_points.cpu().numpy(),neighbor_pos[0].cpu().numpy())
            return


        # support0=feature_dict['conv0']['pos']
        # support2=feature_dict['conv2']['pos']
        # support4=feature_dict['conv4']['pos']
        # visual_nocs_point=np.array([0.0,0.1,0])
        # visual_nocs_point=torch.from_numpy(visual_nocs_point).cuda().float().unsqueeze(0)
        # Rt=fake_rotation
        # res=center.detach().cpu().numpy().reshape(3)
        # support0_nocs=torch.from_numpy(((support0[0].cpu().numpy()) @ Rt)).cuda().float()/(mean_shape.max())
        # support2_nocs=torch.from_numpy(((support2[0].cpu().numpy()) @ Rt)).cuda().float()/(mean_shape.max())
        # support4_nocs=torch.from_numpy(((support4[0].cpu().numpy()) @ Rt)).cuda().float()/(mean_shape.m
        # visual_neighbor(visual_nocs_point,support4_nocs,16)


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

        def get_bin(new_query_nocs):
            if sym[0]==1:
                new_query_r=torch.norm(new_query_nocs[:,:,(0,2)],dim=-1)
                new_query_nocs_bin=torch.zeros_like(new_query_nocs).long()
                new_query_nocs_bin[:,:,0]=torch.clamp(((new_query_r-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
                new_query_nocs_bin[:,:,1]=torch.clamp(((new_query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
            else:
                new_query_nocs_bin=torch.zeros_like(new_query_nocs).long()
                new_query_nocs_bin[:,:,0]=torch.clamp(((new_query_nocs[:,:,0]-x_start)/x_bin_resolution),0,FLAGS.bin_size-1).long()
                new_query_nocs_bin[:,:,1]=torch.clamp(((new_query_nocs[:,:,1]-y_start)/y_bin_resolution),0,FLAGS.bin_size-1).long()
                new_query_nocs_bin[:,:,2]=torch.clamp(((new_query_nocs[:,:,2]-z_start)/z_bin_resolution),0,FLAGS.bin_size-1).long()
            return new_query_nocs_bin

        support0=feature_dict['conv0']['pos'][0]
        support2=feature_dict['conv2']['pos'][0]
        support4=feature_dict['conv4']['pos'][0]
        def cal_weight(query_nocs,gt_R,gt_T,gt_s,supports,axis=0,level=0):
            query_points=(query_nocs*gt_s) @ gt_R.T + gt_T
            neighbor=get_neighbor(query_points[0],supports,16)
            dis_weight_1=torch.norm(query_points[0].unsqueeze(1)-neighbor,dim=-1)
            dis_weight_1=dis_weight_1/torch.sum(dis_weight_1,dim=-1,keepdim=True)
            change=torch.zeros(3,device=supports.device)
            change[axis]=0.01*(2**level)
            query_change=((query_nocs+change)*gt_s) @ gt_R.T + gt_T
            dis_weight_2=torch.norm(query_change[0].unsqueeze(1)-neighbor,dim=-1)
            dis_weight_2=dis_weight_2/torch.sum(dis_weight_2,dim=-1,keepdim=True)
            weight_change=torch.abs(dis_weight_1-dis_weight_2).sum(-1)
            return weight_change
        if FLAGS.use_weight:
            x_weight=cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support0,0)+cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support2,0,2)\
                     +cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support4,0,4)
            y_weight=cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support0,1)+cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support2,1,2)\
                     +cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support4,1,4)
            z_weight=cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support0,2)+cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support2,2,2)\
                     +cal_weight(new_query_nocs,gt_R,gt_T,gt_s,support4,2,4)
            x_weight=x_weight
            x_weight=x_weight/(x_weight.sum())
            y_weight=y_weight
            y_weight=y_weight/(y_weight.sum())
            z_weight=z_weight
            z_weight=z_weight/(z_weight.sum())
        if FLAGS.debug:
            from sklearn.manifold import TSNE
            gt_s_max=gt_s.max()
            x, y = np.mgrid[-0.6:0.6:40j,
                   -0.6:0.6:40j]
            split=x.shape[0]
            x=torch.from_numpy(x).float().cuda()
            y=torch.from_numpy(y).float().cuda()
            x=x*gt_s_max/gt_s[0]
            y=y*gt_s_max/gt_s[1]
            z=torch.zeros_like(x)
            plane_query_nocs=torch.stack([x,y,z],dim=-1).reshape(1,-1,3)
            # gt_T=T[0]
            # gt_R=torch.from_numpy(fake_rotation).cuda()
            plane_query=(plane_query_nocs*gt_s) @ gt_R.T + gt_T
            surf_nocs=(pc_center-gt_T) @ gt_R / gt_s
            surf_nocs_xy=surf_nocs[torch.abs(surf_nocs[:,:,2])<0.1]
            # show_open3d(surf_nocs_xy.detach().cpu().numpy(),plane_query_nocs[0].detach().cpu().numpy())
            # show_open3d(fake_query[0].detach().cpu().numpy(),plane_query[0].detach().cpu().numpy())
            pred_plane_nocs_bin,feat=cat_model.qnet(plane_query,feature_dict,debug=True)
            tsne = TSNE(n_components=1, init='pca', random_state=0)


            plane_query=plane_query.requires_grad_()



            x_weight=cal_weight(plane_query_nocs,gt_R,gt_T,gt_s,support0,0)+cal_weight(plane_query_nocs,gt_R,gt_T,gt_s,support2,0,2)\
                     +cal_weight(plane_query_nocs,gt_R,gt_T,gt_s,support4,0,4)
            y_weight=cal_weight(plane_query_nocs,gt_R,gt_T,gt_s,support0,1)+cal_weight(plane_query_nocs,gt_R,gt_T,gt_s,support2,1,2)+\
                     cal_weight(plane_query_nocs,gt_R,gt_T,gt_s,support4,1,4)
            pred_plane_nocs_bin=pred_plane_nocs_bin.reshape(1,-1,3,FLAGS.bin_size).permute(0,3,1,2).detach()

            m=torch.nn.Softmax(dim=1)
            dis=m(pred_plane_nocs_bin)
            log_dis=torch.log(dis)
            h=-torch.sum(log_dis*dis,dim=1)
            gt_plane_nocs_bin=get_bin(plane_query_nocs)
            cross=nn.CrossEntropyLoss(reduce=False)
            pred_plane_nocs_bin_max=pred_plane_nocs_bin.max(1)[1]
            pred_plane_nocs_x=(pred_plane_nocs_bin_max[:,:,0]*x_bin_resolution+x_start)
            pred_plane_nocs_y=(pred_plane_nocs_bin_max[:,:,1]*y_bin_resolution+y_start)
            pred_plane_nocs_z=(pred_plane_nocs_bin_max[:,:,2]*z_bin_resolution+z_start)
            if sym[0]==1:
                plane_query_nocs_r=torch.abs(plane_query_nocs[:,:,0])
                dis_x=torch.abs(pred_plane_nocs_x-plane_query_nocs_r)
                dis_y=torch.abs(pred_plane_nocs_y-plane_query_nocs[:,:,1])
                loss=cross(pred_plane_nocs_bin[:,:,:,:2],gt_plane_nocs_bin[:,:,:2])
                loss=loss.reshape(split,split,2)
            else:
                dis_x=torch.abs(pred_plane_nocs_x-plane_query_nocs[:,:,0])
                dis_y=torch.abs(pred_plane_nocs_y-plane_query_nocs[:,:,1])
                loss=cross(pred_plane_nocs_bin,gt_plane_nocs_bin)
                loss=loss.reshape(split,split,3)
            surf_nocs_xy_np=surf_nocs_xy.detach().cpu().numpy()
            x_weight_np=x_weight.detach().cpu().numpy().reshape(split,split)
            y_weight_np=y_weight.detach().cpu().numpy().reshape(split,split)
            feat_np=tsne.fit_transform(feat[0].detach().cpu().numpy()).reshape(split,split)
            h_x_np=h[:,:,0].detach().cpu().numpy().reshape(split,split)
            h_y_np=h[:,:,1].detach().cpu().numpy().reshape(split,split)
            x_np=x.detach().cpu().numpy()
            y_np=y.detach().cpu().numpy()
            fig, ax = plt.subplots()
            # plt.axis("equal")
            dis_x_np=dis_x.reshape(split,split).detach().cpu().numpy()
            dis_y_np=dis_y.reshape(split,split).detach().cpu().numpy()
            loss_x_np=loss[:,:,0].detach().cpu().numpy()
            loss_y_np=loss[:,:,1].detach().cpu().numpy()
            x_min=x_np.min()
            x_max=x_np.max()
            y_min=y_np.min()
            y_max=y_np.max()
            loss_x_min, loss_x_max =loss_x_np.min(), loss_x_np.max()
            loss_y_min, loss_y_max =loss_y_np.min(), loss_y_np.max()
            # c = plt.pcolormesh(x_np, y_np, loss_x_np, cmap ='Greens', vmin = loss_x_min, vmax = loss_x_max)
            # c = plt.pcolormesh(x_np, y_np, loss_y_np, cmap ='Greens', vmin = loss_y_min, vmax = loss_y_max)
            # c = plt.pcolormesh(x_np, y_np, dis_x_np, cmap ='Greens', vmin = dis_x_np.min(), vmax = dis_x_np.max())
            c = plt.pcolormesh(x_np, y_np, dis_y_np, cmap ='Greens', vmin = dis_y_np.min(), vmax = dis_y_np.max())
            # c = plt.pcolormesh(x_np, y_np, h_x_np, cmap ='Greens', vmin = h_x_np.min(), vmax = h_x_np.max())
            # c = plt.pcolormesh(x_np, y_np, h_y_np, cmap ='Greens', vmin = h_y_np.min(), vmax = h_y_np.max())
            # c = plt.pcolormesh(x_np, y_np, x_weight_np, cmap ='Greens', vmin = x_weight_np.min(), vmax = x_weight_np.max())
            # c = plt.pcolormesh(x_np, y_np, y_weight_np, cmap ='Greens', vmin = y_weight_np.min(), vmax = y_weight_np.max())

            cm=plt.cm.get_cmap('rainbow')
            # c = plt.pcolormesh(x_np, y_np, feat_np, cmap =cm, vmin = feat_np.min(), vmax = feat_np.max())
            ax.set_xlim(x_min,x_max)
            ax.set_ylim(y_min,y_max)
            plt.colorbar(c)
            plt.scatter(surf_nocs_xy_np[:,0], surf_nocs_xy_np[:,1],color='black')
            plt.show()


        pred_fake_nocs_bin=cat_model.qnet(fake_query,feature_dict).reshape(1,-1,3,FLAGS.bin_size).detach()
        new_query_nocs_bin=get_bin(new_query_nocs)



        show_dict=EasyDict()
        show_dict.show_new_query=None
        show_dict.ori_show_new_query=None
        m=torch.nn.Softmax(dim=-1)
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

            if FLAGS.use_scene:
                fake_query_nocs = (fake_query-new_T) @ new_R / new_scale

                if sym[0]==1:
                    fake_query_nocs_r=torch.norm(fake_query_nocs[:,:,(0,2)],dim=-1)
                    fake_query_nocs[:,:,0]=fake_query_nocs_r
                pred_fake_nocs_bin_max=pred_fake_nocs_bin.max(-1)[1]
                pred_fake_nocs_x=(pred_fake_nocs_bin_max[:,:,0]*x_bin_resolution+x_start)
                pred_fake_nocs_y=(pred_fake_nocs_bin_max[:,:,1]*y_bin_resolution+y_start)
                pred_fake_nocs_z=(pred_fake_nocs_bin_max[:,:,2]*z_bin_resolution+z_start)

                scene_index_y=torch.abs(pred_fake_nocs_y)<FLAGS.query_radius
                if sym[0]==1:
                    scene_index_x=pred_fake_nocs_x<FLAGS.query_radius
                    scene_index_z=torch.ones_like(scene_index_x)
                else:
                    scene_index_x=torch.abs(pred_fake_nocs_x)<FLAGS.query_radius
                    scene_index_z=torch.abs(pred_fake_nocs_z)<FLAGS.query_radius
                if FLAGS.use_scene_index:
                    scene_index=torch.logical_and(scene_index_x,scene_index_y)
                    scene_index=torch.logical_and(scene_index,scene_index_z).squeeze()
                else:
                    scene_index=torch.ones_like(scene_index_x).squeeze()
                pred_fake_nocs=torch.stack([pred_fake_nocs_x,pred_fake_nocs_y,pred_fake_nocs_z],dim=-1)
                scene_pred_fake_nocs=pred_fake_nocs[:,scene_index]
                scene_fake_query_nocs=fake_query_nocs[:,scene_index]
                if sym[0]==0:
                    scene_score=torch.mean(torch.norm(scene_pred_fake_nocs - scene_fake_query_nocs, dim=-1))
                else:
                    scene_score=torch.mean(torch.norm(scene_pred_fake_nocs[:,:,:2] - scene_fake_query_nocs[:,:,:2], dim=-1))
            else:
                scene_score=0

            if FLAGS.use_prob:
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
                if FLAGS.use_weight:
                    if sym[0]==1:
                        pred_prob=(pred_prob[:,0]*x_weight).sum()+(pred_prob[:,1]*y_weight).sum()
                    else:
                        pred_prob=(pred_prob[:,0]*x_weight).sum()+(pred_prob[:,1]*y_weight).sum()+(pred_prob[:,2]*z_weight).sum()
                else:
                    if sym[0]==1:
                        pred_prob=(pred_prob[:,0]+pred_prob[:,1])
                    else:
                        pred_prob=(pred_prob[:,0]+pred_prob[:,1]+pred_prob[:,2])
                prob_score=-pred_prob.mean()
            else:
                prob_score=0
            score=prob_score+scene_score
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