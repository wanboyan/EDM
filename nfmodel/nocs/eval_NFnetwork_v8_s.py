import gc

import cv2
import nbformat.reader
import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from network.point_sample.pc_sample import PC_sample
from nfmodel.uti_tool import *
from nfmodel.nocs.NFnetwork_v8_s import PIPS_s
from nfmodel.nocs.NFnetwork_v7_pips_t import PIPS_T
from tools.training_utils import get_gt_v
from nnutils.torch_pso import  *
from network.point_sample.pc_sample import farthest_sample
from easydict import EasyDict
import torch.optim as optim
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from pytorch3d.transforms import so3_log_map,so3_exponential_map
from tools.rot_utils import get_rot_mat_y_first


def cal_rot(p_green_R,p_red_R,sym):




    p_green_R =p_green_R[0].detach().cpu().numpy()
    p_red_R=p_red_R[0].detach().cpu().numpy()


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
    Rotation = pose[0][0:3, 0:3]
    return Rotation


class multi_NFPose(nn.Module):
    def __init__(self,cat_names):
        super(multi_NFPose, self).__init__()
        self.per_networks=nn.ModuleDict()
        self.per_pips_t=nn.ModuleDict()
        self.cat_names=cat_names
        self.regular_grid=make_regular_grid(11)[0].cuda()
        for cat_name in cat_names:
            if cat_name in ['bottle','bowl','can']:
                cat_model_path=os.path.join(FLAGS.resume_dir,'sym',FLAGS.resume_model_name)
            else:
                cat_model_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
            cat_model=PIPS_s().eval()
            state_dict=torch.load(cat_model_path)
            if FLAGS.stage==2:
                partail_keys=[k for k in state_dict.keys() if 'qnet' in k or 'backbone1'in k or 'ts'in k
                              or 'loss_coord' in k]
            else:
                partail_keys=[k for k in state_dict.keys() if 'qnet' in k or 'backbone1'in k or 'ts'in k
                              or 'loss_coord' in k]
            partail_dicts={k:state_dict[k] for k in partail_keys}
            cat_model.load_state_dict(partail_dicts,strict=False)
            # cat_model.load_state_dict(torch.load(cat_model_path),strict=False)
            self.per_networks[cat_name]=cat_model

    def forward(self, depth, cat_id_0base, camK, gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None,def_mask=None,
                model_points=None, nocs_scale=None,rgb=None,gt_mask=None,
                coefficients=None,control_points=None, std_models=None,sota_R=None,sota_t=None,sota_s=None,
                picture_index=None,rgb_whole=None):


        if FLAGS.pic_save:
            if not os.path.exists(FLAGS.pic_save_dir):
                os.mkdir(FLAGS.pic_save_dir)
            FLAGS.cur_eval_index=picture_index
            rgb_path=os.path.join(FLAGS.pic_save_dir,str(FLAGS.cur_eval_index)+'_rgb.png')
            # plt.imshow(rgb_whole[:,:,(2,1,0)])
            cv2.imwrite(rgb_path,rgb_whole )


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
            res,scale=self.per_infer(cat_name,sym[i],PC[i],mean_shape[i],gt_R[i],gt_t[i],gt_s[i],sota_R[i],sota_t[i],sota_s[i],
                                     coefficients[i],control_points[i],std_models[i],rgb_whole,model_points[i],i)
            res_list.append(res)
            scale_list.append(scale)
        return torch.stack(res_list,dim=0),torch.stack(scale_list,dim=0)

    def per_infer(self,cat_name,sym,pc,mean_shape,gt_R=None,gt_T=None,gt_s=None,sota_R=None,sota_t=None,sota_s=None,
                  coefficients=None,control_points=None,std_model=None,rgb_whole=None,model_point=None,obj_index=None):
        # show_open3d(pc.detach().cpu().numpy(),pc.detach().cpu().numpy())
        choose=farthest_sample(model_point.cuda(),1024)
        model_point=model_point[choose]
        cat_model=self.per_networks[cat_name]
        PC=pc.unsqueeze(0)
        real_shape=gt_s
        nocs_scale=torch.norm(real_shape,dim=-1)
        mean_shape=mean_shape.detach().cpu().numpy()

        center=PC.mean(dim=1,keepdim=True)

        gt_T=gt_T-center.squeeze()
        pc_num=PC.shape[1]
        pc_center=PC-center

        # pc_center=pc_center@ gt_R.T+gt_T

        # regular_grid=make_regular_grid(5)
        # show_open3d(canonical_pc[0].detach().cpu().numpy(),regular_grid.detach().cpu().numpy())
        #
        with torch.no_grad():
            feature_dict= cat_model.backbone1(pc_center)

        if FLAGS.scale_invariant:
            pred_scale=nocs_scale
        else:
            pred_scale=torch.ones_like(nocs_scale)


        grid_R=gt_R.cpu().numpy()
        grid_T=gt_T.cpu().numpy()
        grid_s=nocs_scale.cpu().numpy()

        if sym[0]==1:

            gt_green_v, gt_red_v = get_gt_v(gt_R[None,:,:])
            my_x=torch.zeros_like(gt_green_v)
            my_x[:,-1]=1.0
            my_z=torch.cross(my_x,gt_green_v,dim=-1)
            my_z=F.normalize(my_z,dim=-1)
            my_x=torch.cross(gt_green_v,my_z,dim=-1)
            my_x=F.normalize(my_x,dim=-1)
            my_R=torch.cat([my_x[:,:,None],gt_green_v[:,:,None,],my_z[:,:,None,]],dim=2)
            gt_R=my_R
            gt_R=gt_R[0]
        canonical_pc=torch.bmm(pc_center-gt_T.reshape(-1,1,3),gt_R.reshape(-1,3,3))/(nocs_scale.reshape(-1,1,1))
        fake_grid = grids['fake_grid'].numpy()
        boxsize = 1
        fake_grid = boxsize * (fake_grid)

        # feat_for_ts = pc_center
        # T,s= cat_model.ts(feat_for_ts)

        use_sota=True
        if use_sota:
            pred_R=sota_R.cpu().numpy()
            pred_t=sota_t.cpu().numpy()-center[0,0].cpu().numpy()
            pred_s=sota_s.cpu().numpy()
        else:
            pred_t=T[0].detach().cpu().numpy()
            pred_s=s[0].detach().cpu().numpy()

        use_fake_pose=False
        if use_fake_pose:
            # fake_grid=torch.from_numpy(fake_grid)
            # # fake_grid=torch.cat([torch.tensor([[0,0,0]]),fake_grid],dim=0).numpy()
            # fake_grid=fake_grid.numpy()
            #
            # fake_grid_scaled=fake_grid*grid_s
            # fake_query_np=fake_grid_scaled @ grid_rotation.T + grid_T
            #
            # fake_query_norm_np=fake_grid @ grid_rotation.T+ grid_T
            # fake_query_norm=torch.from_numpy(fake_query_norm_np).unsqueeze(0).float().cuda()
            # fake_query=torch.from_numpy(fake_query_np).unsqueeze(0).float().cuda()
            # fake_nocs=torch.from_numpy(fake_grid).unsqueeze(0).float().cuda()
            # fake_num=fake_query.shape[1]
            points_uniform = np.random.rand(1000, 3)
            points_uniform = 1.2 * (points_uniform - 0.5)
            points_r=np.linalg.norm(points_uniform,axis=-1)
            points_uniform=points_uniform[points_r<0.6]
            points_uniform=points_uniform*grid_s+grid_T

            points_uniform=torch.from_numpy(points_uniform).float()
            fake_query=points_uniform.unsqueeze(0).float().cuda()
            fake_num=fake_query.shape[1]
        else:
            points_uniform = np.random.rand(1500, 3)
            points_uniform = 1.2 * (points_uniform - 0.5)
            points_r=np.linalg.norm(points_uniform,axis=-1)
            # if sym[0]==0:
            points_uniform=points_uniform[points_r<0.6]
            # else:
            #     points_uniform= np.matmul(pred_R[None,:,:] , points_uniform[:,:,None])[:,:,0]

            points_uniform=points_uniform*np.linalg.norm(pred_s,axis=-1)+pred_t

            points_uniform=torch.from_numpy(points_uniform).float()
            fake_query=points_uniform.unsqueeze(0).float().cuda()
            fake_num=fake_query.shape[1]


        with torch.no_grad():
            pred_dict=cat_model.qnet(fake_query,feature_dict,pred_scale.detach())
        # FLAGS.stage=0

        if FLAGS.stage==0:
            pred_camera=pred_dict['coord']
            pred_log_stds=pred_dict['log_stds'].detach()
            inv_stds=torch.exp(-pred_log_stds.reshape(-1,3))
            inv_stds_mean=torch.mean(inv_stds,dim=-1,keepdim=True)
            inv_stds_mean=inv_stds_mean.reshape(1,-1,1)
            choose_by_var=torch.topk(inv_stds_mean[:,:,0],k=fake_num//2,dim=-1,largest=True)[1]
            fake_query_choose=torch.gather(fake_query,1,choose_by_var[:,:,None].repeat(1,1,3))
            pred_camera_choose=torch.gather(pred_camera,1,choose_by_var[:,:,None].repeat(1,1,3))
            pred_camera=pred_camera_choose
            fake_query=fake_query_choose
            # show_open3d(pc_center[0].detach().cpu().numpy(),fake_query[0].cpu().numpy())
            noise_Rs=[]
            opti_bs=32
            for i in range(opti_bs):
                x=torch.Tensor(1)
                x.uniform_(-180,180)
                y=torch.Tensor(1)
                y.uniform_(-180,180)
                z=torch.Tensor(1)
                z.uniform_(-180,180)
                delta_r1 = get_rotation_torch(x, y, z)
                noise_Rs.append(delta_r1)
            noise_Rs=torch.stack(noise_Rs,dim=0).float()
            init_R=noise_Rs.numpy()
            init_s=np.repeat(np.linalg.norm(pred_s,axis=-1),opti_bs,0)
            init_t=np.repeat(pred_t[None,:],opti_bs,0)
            Rotation, Translation=optimize_pose2(init_R,init_t,init_s,pred_camera.detach(),fake_query,sym)

        elif FLAGS.stage==1:
            if sym[0]==0 :

                pred_camera=pred_dict['coord']

                # z_inv=pred_dict['z_inv'].detach()
                # z_so3=pred_dict['z_so3'].detach()
                pred_log_stds=pred_dict['log_stds'].detach()
                pred_log_stds_sum=pred_log_stds.sum(-1)
                pred_stds=torch.exp(pred_log_stds_sum)


                choose_by_var=torch.topk(pred_stds,k=fake_num//2,dim=-1,largest=False)[1]
                # z_so3_choose=torch.gather(z_so3,1,choose_by_var[:,:,None,None].repeat(1,1,z_inv.shape[2],3))
                fake_query_choose=torch.gather(fake_query,1,choose_by_var[:,:,None].repeat(1,1,3))
                pred_camera_choose=torch.gather(pred_camera,1,choose_by_var[:,:,None].repeat(1,1,3))



                if FLAGS.use_stable_train==0:
                    # point_fea=z_so3_choose
                    pred_camera=pred_camera_choose
                    fake_query=fake_query_choose
                else:
                    pass
                    # z_inv_choose=torch.gather(z_inv,1,choose_by_var[:,:,None].repeat(1,1,z_inv.shape[2]))
                    # pred_mask=cat_model.weight_model.generate(z_inv_choose).unsqueeze(-1)
                    # stable_z_so3=z_so3_choose*pred_mask.unsqueeze(-1)
                    # point_fea=stable_z_so3
                    # pred_camera=pred_camera_choose*pred_mask
                    # pred_camera=pred_camera[pred_mask[:,:,0]!=0].unsqueeze(0)
                    # fake_query=fake_query_choose*pred_mask
                    # fake_query=fake_query[pred_mask[:,:,0]!=0].unsqueeze(0)

                # fake_query_center=fake_query-torch.mean(fake_query,dim=1,keepdim=True)
                # pred_camera_center=pred_camera-torch.mean(pred_camera,dim=1,keepdim=True)
                # cal_scale=(torch.norm(  fake_query_center,dim=-1).sum())/(torch.norm(pred_camera_center,dim=-1).sum())
                # cal_tran=fake_query.mean(1)-(pred_camera*cal_scale).mean(1)
                # Translation=cal_tran[0].detach().cpu().numpy()    a
                # Rotation=gt_R.cpu().numpy()
                # show_open3d(pc_center[0].detach().cpu().numpy(),fake_query[0].cpu().numpy())
                Scale, Rotation, Translation, OutTransform,inlier_index=estimateSimilarityTransform(pred_camera[0].detach().cpu().numpy(),
                                                                                                    fake_query[0].detach().cpu().numpy())
            elif sym[0]==1:
                if FLAGS.qnet_version=='v7':
                    pred_nocs=pred_dict['coord'].detach()
                    pred_nocs[:,:,0]=0
                    pred_nocs[:,:,2]=0

                    pred_c_nocs=pred_dict['c_coord'].detach()

                    pred_log_stds=pred_dict['log_stds'].detach()[:,:,:1]


                    pred_c_log_stds=pred_dict['c_log_stds']

                    pred_log_stds_sum=pred_log_stds.mean(-1)+pred_c_log_stds.mean(-1)
                    pred_stds=torch.exp(pred_log_stds_sum)
                    choose_by_var=torch.topk(pred_stds,k=fake_num//2,dim=-1,largest=False)[1]
                    # z_so3_choose=torch.gather(z_so3,1,choose_by_var[:,:,None,None].repeat(1,1,z_inv.shape[2],3))
                    fake_query_choose=torch.gather(fake_query,1,choose_by_var[:,:,None].repeat(1,1,3))
                    pred_nocs_choose=torch.gather(pred_nocs,1,choose_by_var[:,:,None].repeat(1,1,3))
                    pred_c_nocs_choose=torch.gather(pred_c_nocs,1,choose_by_var[:,:,None].repeat(1,1,3))


                    if FLAGS.use_stable_train==0:
                        pred_c_nocs=pred_c_nocs_choose
                        pred_nocs=pred_nocs_choose
                        fake_query=fake_query_choose
                    # show_open3d(pc_center[0].detach().cpu().numpy(),fake_query[0].cpu().numpy())
                    # Scale, pred_R, _
                    # =estimateSymmetryTransform(pred_nocs[0].detach().cpu().numpy(),fake_query[0].detach().cpu().numpy())
                    _, Rotation, _, OutTransform,inlier_index=estimateSimilarityTransform(pred_nocs[0].detach().cpu().numpy(),
                                                                                                            pred_c_nocs[0].detach().cpu().numpy())
                    Translation=pred_t
                else:
                    pred_nocs=pred_dict['coord'].detach()


                    pred_log_stds=pred_dict['log_stds'].detach()[:,:,:2]
                    pred_log_stds_sum=pred_log_stds.sum(-1)
                    pred_stds=torch.exp(pred_log_stds_sum)
                    choose_by_var=torch.topk(pred_stds,k=fake_num//2,dim=-1,largest=False)[1]
                    fake_query_choose=torch.gather(fake_query,1,choose_by_var[:,:,None].repeat(1,1,3))
                    pred_nocs_choose=torch.gather(pred_nocs,1,choose_by_var[:,:,None].repeat(1,1,3))
                    if FLAGS.use_stable_train==0:
                        pred_nocs=pred_nocs_choose
                        fake_query=fake_query_choose
                    # show_open3d(pc_center[0].detach().cpu().numpy(),fake_query[0].cpu().numpy())
                    pred_nocs=pred_nocs[:,:,:2]
                    # Scale, Rotation, Translation=estimateSymmetryTransform(pred_nocs[0].detach().cpu().numpy(),fake_query[0].detach().cpu().numpy())

                    Rotation,Translation,Scale = optimize_pose(pred_R,pred_t,np.array([np.linalg.norm(pred_s,axis=-1)],dtype=np.float32),pred_nocs,fake_query)
                    # Rotation=grid_R
                    # Rotation=pred_R
                    # Translation=pred_t


        if FLAGS.stage==2:

            pred_coord=pred_dict['coord'].detach()
            z_inv=pred_dict['z_inv'].detach()
            pred_log_stds=pred_dict['log_stds'].detach()
            pred_rot_1=pred_dict['rot_vec_1'].detach()
            pred_rot_2=pred_dict['rot_vec_2'].detach()
            pred_var_R=get_rot_mat_y_first(pred_rot_1,pred_rot_2).reshape(-1,3,3)

            inv_stds=torch.exp(-pred_log_stds.reshape(-1,3))
            inv_stds_ma=torch.diag_embed(inv_stds)
            inv_sigma=torch.bmm(inv_stds_ma,pred_var_R.permute(0,2,1)).reshape(1,-1,3,3)
            inv_sigma=inv_sigma/cat_model.loss_coord.mean_inv_std
            fake_nocs=((fake_query[0]-gt_T) @ gt_R)/nocs_scale
            pc_nocs=((pc_center[0]-gt_T) @ gt_R)/nocs_scale


            model_transformed=((model_point.cuda()) @ (gt_R.T))+gt_T

            fake_error=torch.norm(pred_coord-fake_nocs,dim=-1)/0.3

            from evaluation.utils.eval_utils import draw_single_results

            result_dict={}



            query_color=show_error_2(pc_nocs.cpu().numpy(),fake_nocs.cpu().numpy(),
                         error=fake_error[0].cpu().numpy(),
                         outpath=os.path.join(FLAGS.pic_save_dir,str(FLAGS.cur_eval_index)+f'_error_1_{obj_index}.png'))

            result_dict['query_color']=query_color
            result_dict['query_points']=fake_nocs.cpu().numpy()
            result_dict['input_points']=pc_nocs.cpu().numpy()
            result_dict['query_error']=fake_error.cpu().numpy()



            pred_s=gt_s.cpu().numpy()
            real_intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)

            pred_rt_1=get_noise_rt(gt_R.cpu(),(gt_T+center).cpu(),16,0.016).numpy()
            draw_single_results(rgb_whole.copy(),FLAGS.pic_save_dir,f'pred_pose_1_{obj_index}',FLAGS.cur_eval_index,real_intrinsics,
                                pred_rt_1[None,:,:],pred_s[None,:],[6],color=(0,0,255),)



            neighbors=get_neighbor(fake_query[0],model_transformed,1)
            neighbors_distance=torch.norm(neighbors-fake_query,dim=-1)
            fake_query_choose=fake_query[neighbors_distance<0.04]
            inv_sigma_choose=inv_sigma[neighbors_distance<0.04]
            fake_nocs_choose=fake_nocs.unsqueeze(0)[neighbors_distance<0.04]
            fake_error_choose=fake_error[neighbors_distance<0.04]
            result_dict['choose_idx']=(neighbors_distance<0.04).nonzero()[:,1].cpu().numpy()

            show_error_2(pc_nocs.cpu().numpy(),fake_nocs_choose.cpu().numpy(),
                         error=fake_error_choose.cpu().numpy(),
                         outpath=os.path.join(FLAGS.pic_save_dir,str(FLAGS.cur_eval_index)+f'_error_2_{obj_index}.png'))

            pred_rt_2=get_noise_rt(gt_R.cpu(),(gt_T+center).cpu(),8,0.008).numpy()
            draw_single_results(rgb_whole.copy(),FLAGS.pic_save_dir,f'pred_pose_2_{obj_index}',FLAGS.cur_eval_index,real_intrinsics,
                                pred_rt_2[None,:,:],pred_s[None,:],[6],color=(0,0,255),)


            # show_open3d(fake_query_choose.cpu().numpy(),pc_center[0].cpu().numpy())

            sample_idx=pick_stable(fake_nocs_choose,inv_sigma_choose)
            fake_nocs_stable=fake_nocs_choose[sample_idx]
            fake_error_stable=fake_error_choose[sample_idx]
            result_dict['stable_idx']=sample_idx.cpu().numpy()
            np.savez(os.path.join(FLAGS.pic_save_dir,str(FLAGS.cur_eval_index)+f'_{obj_index}.npz'),
                     **result_dict)

            pred_rt_3=get_noise_rt(gt_R.cpu(),(gt_T+center).cpu(),4,0.004).numpy()
            draw_single_results(rgb_whole.copy(),FLAGS.pic_save_dir,f'pred_pose_2_{obj_index}',FLAGS.cur_eval_index,real_intrinsics,
                                pred_rt_3[None,:,:],pred_s[None,:],[6],color=(0,0,255),)
            show_error_2(pc_nocs.cpu().numpy(),fake_nocs_stable.cpu().numpy(),
                         error=fake_error_stable.cpu().numpy(),
                         outpath=os.path.join(FLAGS.pic_save_dir,str(FLAGS.cur_eval_index)+f'_error_3_{obj_index}.png'))

            # show_open3d(fake_query_stable.cpu().numpy(),pc_center[0].cpu().numpy())



            pred_camera=pred_dict['coord']

            z_inv=pred_dict['z_inv'].detach()
            pred_log_stds=pred_dict['log_stds'].detach()


            inv_stds=torch.exp(-pred_log_stds.reshape(-1,3))
            inv_stds_mean=torch.mean(inv_stds,dim=-1,keepdim=True)/cat_model.loss_coord.mean_inv_std
            inv_stds_mean=inv_stds_mean.reshape(1,-1,1)






        # fake_grid_transformed=((fake_grid*pred_s) @ (Rotation.T))+Translation
        # show_open3d(pc_center[0].detach().cpu().numpy(),fake_grid_transformed)


        vis_error=False
        if vis_error:
            if sym[0]==1:
                draw_sym_map(feature_dict,cat_model,canonical_pc,pc_center,gt_R,gt_T,nocs_scale)
            else:
                draw_map(feature_dict,cat_model,canonical_pc,pc_center,gt_R,gt_T,nocs_scale)


        res = torch.eye(4, dtype=torch.float)
        try:
            res[:3,:3]=torch.from_numpy(Rotation)
            res[:3,3]=torch.from_numpy(Translation)+center.reshape(3).cpu()

        except:
            print('error !!!!!')
            pass
        # cur_s=s[0].detach().cpu()
        cur_s=torch.from_numpy(pred_s)
        # cur_s=real_shape


        return res,cur_s





def get_neighbor(query_point,support_points,num_neighber):

    from EQNet.eqnet.ops.knn.knn_utils import knn_query
    from EQNet.eqnet.ops.grouping.grouping_utils import grouping_operation
    query_cnt=query_point.new_zeros(1).int()
    query_cnt[0]=query_point.shape[0]
    support_cnt=support_points.new_zeros(1).int()
    support_cnt[0]=support_points.shape[0]

    index_pair = knn_query(
        num_neighber,
        support_points, support_cnt,
        query_point, query_cnt).int()
    neighbor_pos=grouping_operation(
        support_points, support_cnt, index_pair, query_cnt).permute(0,2,1).squeeze(1)

    return neighbor_pos

def optimize_pose(init_r,init_t,init_s,source,target):
    source=source.detach().cpu()
    target=target.detach().cpu()
    new_Rvec=torch.from_numpy(cv2.Rodrigues(init_r)[0][:,0]).float().requires_grad_()
    new_T=torch.from_numpy(init_t.astype(np.float32)).float().requires_grad_()
    new_scale=torch.from_numpy(init_s.astype(np.float32)).float().requires_grad_()

    params=[{'params':new_Rvec},{'params':new_T},{'params':new_scale}]
    opt_1 = torch.optim.Adam(params, lr=0.001)
    opt_2=torch.optim.LBFGS([new_Rvec,new_T,new_scale],lr=1,max_iter=100,line_search_fn="strong_wolfe")
    step=50

    def objective():
        cur_R=Rodrigues.apply(new_Rvec)
        cur_t=new_T
        cur_s=new_scale

        cal_fake_nocs=torch.bmm(target-cur_t.reshape(-1,1,3),cur_R.reshape(1,3,3))/cur_s.reshape(-1,1,1)
        cal_fake_nocs_r=torch.norm(cal_fake_nocs[:,:,(0,2)],dim=-1)
        cal_fake_nocs[:,:,0]=cal_fake_nocs_r
        distance=torch.norm(cal_fake_nocs[:,:,:2] - source[:,:,:2], dim=-1)
        scene_score=distance.mean()
        # print(scene_score)
        return scene_score

    def closure_1():
        opt_1.zero_grad()
        cur_loss=objective()
        cur_loss.backward(retain_graph=True)
        return cur_loss
    def closure2():
        opt_2.zero_grad()
        cur_loss=objective()
        cur_loss.backward(retain_graph=True)
        return cur_loss
    for i in range(step):
        closure_1()
        opt_1.step()

    # opt_2.step(closure2)

    cur_R=Rodrigues.apply(new_Rvec).detach().numpy()
    cur_t=new_T.detach().numpy()
    cur_s=new_scale.detach().numpy()
    return cur_R,cur_t,cur_s



def optimize_pose2(init_r,init_t,init_s,source,target,sym):
    source=source.detach().cpu()
    target=target.detach().cpu()


    new_Rvec=so3_log_map(torch.from_numpy(init_r)).float().requires_grad_()
    new_T=torch.from_numpy(init_t.astype(np.float32)).float().requires_grad_()
    new_s=torch.from_numpy(init_s.astype(np.float32)).float().requires_grad_()

    params=[{'params':new_Rvec},{'params':new_T},{'params':new_s}]
    opt_1 = torch.optim.Adam(params, lr=0.01)
    step=100

    def objective():
        cur_R=so3_exponential_map(new_Rvec)
        cur_t=new_T
        cal_fake_nocs=torch.bmm(target-cur_t.reshape(-1,1,3),cur_R.reshape(-1,3,3))/new_s.reshape(-1,1,1)
        if sym[0]==1:
            cal_fake_nocs_r=torch.norm(cal_fake_nocs[:,:,(0,2)],dim=-1)
            cal_fake_nocs[:,:,0]=cal_fake_nocs_r
            distance=torch.norm(cal_fake_nocs[:,:,:2] - source[:,:,:2], dim=-1)
        else:
            cal_fake_nocs_c=cal_fake_nocs
            distance=torch.norm(cal_fake_nocs_c - source, dim=-1)
        scene_score=distance.mean(-1)
        # print(scene_score)
        return scene_score

    def closure_1():
        opt_1.zero_grad()
        cur_loss=objective().sum()
        cur_loss.backward(retain_graph=True)
        # print(cur_loss)
        return cur_loss

    for i in range(step):
        closure_1()
        opt_1.step()

    batch_loss=objective()
    min_index=torch.argmin(batch_loss)
    cur_R=so3_exponential_map(new_Rvec[min_index].unsqueeze(0)).squeeze(0).detach().numpy()
    cur_t=new_T[min_index].detach().numpy()
    return cur_R,cur_t

