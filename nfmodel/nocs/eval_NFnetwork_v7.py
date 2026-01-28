import gc

import cv2
import torch
import torch.nn as nn
import absl.flags as flags
import os
FLAGS = flags.FLAGS
from network.point_sample.pc_sample import PC_sample
from nfmodel.uti_tool import *
from nfmodel.nocs.NFnetwork_v7 import NFPose
from nfmodel.nocs.NFnetwork_v7_pips_t import PIPS_T

from nnutils.torch_pso import  *
from network.point_sample.pc_sample import farthest_sample
from easydict import EasyDict
import torch.optim as optim
from datasets.data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc,get_rotation_torch
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
)



class multi_NFPose(nn.Module):
    def __init__(self,cat_names):
        super(multi_NFPose, self).__init__()
        self.per_networks=nn.ModuleDict()
        self.per_pips_t=nn.ModuleDict()
        self.cat_names=cat_names
        self.regular_grid=make_regular_grid(FLAGS.regular_grid_spilt)[0].cuda()
        for cat_name in cat_names:
            cat_model_path=os.path.join(FLAGS.resume_dir,cat_name,FLAGS.resume_model_name)
            cat_model=NFPose()
            cat_model.load_state_dict(torch.load(cat_model_path),strict=True)

            pips_t_path=os.path.join(FLAGS.resume_pips_t_dir,cat_name,FLAGS.resume_pips_t_model_name)
            pips_t=PIPS_T()
            pips_t.load_state_dict(torch.load(pips_t_path))
            self.per_networks[cat_name]=cat_model
            self.per_pips_t[cat_name]=pips_t

    def forward(self, depth, cat_id_0base, camK, gt_R, gt_t, gt_s, mean_shape, gt_2D=None, sym=None,def_mask=None,
                model_points=None, nocs_scale=None,rgb=None,gt_mask=None,noises=None,
                coefficients=None,control_points=None, std_models=None,
                picture_index=None,rgb_whole=None):
        if FLAGS.pic_save:
            if not os.path.exists(FLAGS.pic_save_dir):
                os.mkdir(FLAGS.pic_save_dir)
            FLAGS.cur_eval_index=picture_index
            rgb_path=os.path.join(FLAGS.pic_save_dir,str(FLAGS.cur_eval_index)+'_rgb.png')
            plt.imshow(rgb_whole[:,:,(2,1,0)])
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
            res,scale=self.per_infer(cat_name,sym[i],PC[i],mean_shape[i],gt_R[i],gt_t[i],gt_s[i],
                                     coefficients[i],control_points[i],std_models[i],rgb_whole,model_points[i],noises[i])
            res_list.append(res)
            scale_list.append(scale)
        return torch.stack(res_list,dim=0),torch.stack(scale_list,dim=0)

    def per_infer(self,cat_name,sym,pc,mean_shape,gt_R=None,gt_T=None,gt_s=None,
                  coefficients=None,control_points=None,std_model=None,
                  rgb_whole=None,model_point=None,noise=None):
        # show_open3d(pc.detach().cpu().numpy(),pc.detach().cpu().numpy())
        choose=farthest_sample(model_point.cuda(),1024)
        model_point=model_point[choose]
        cat_model=self.per_networks[cat_name]
        pips_t=self.per_pips_t[cat_name]
        if FLAGS.use_noise:
            pc=torch.einsum('ij,pj->pi',noise,pc)
        PC=pc.unsqueeze(0)
        real_shape=gt_s
        nocs_scale=torch.norm(real_shape,dim=-1)
        mean_shape=mean_shape.detach().cpu().numpy()

        center=PC.mean(dim=1,keepdim=True)

        gt_T=gt_T-center.squeeze()
        pc_num=PC.shape[1]
        pc_center=PC-center

        # pc_center=pc_center@ gt_R.T+gt_T
        canonical_pc=torch.bmm(pc_center-gt_T.reshape(-1,1,3),gt_R.reshape(-1,3,3))/(nocs_scale.reshape(-1,1,1))
        # regular_grid=make_regular_grid(5)
        # show_open3d(canonical_pc[0].detach().cpu().numpy(),regular_grid.detach().cpu().numpy())
        #
        if FLAGS.backbone=='neuron':
            feature_dict,pred_scale,_= cat_model.backbone1(pc_center)
        else:
            feature_dict= cat_model.backbone1(pc_center)
        pred_scale=nocs_scale




        a=0
        delta_t1 = torch.rand(3)
        delta_t1 = delta_t1.uniform_(-0.00, 0.00)

        x=torch.Tensor(1)
        x.uniform_(-a,a)
        y=torch.Tensor(1)
        y.uniform_(-a,a)
        z=torch.Tensor(1)
        z.uniform_(-a,a)
        delta_r1 = get_rotation_torch(x, y, z)
        init_R=gt_R.cpu().numpy()@ delta_r1.numpy()
        #
        grid_rotation=init_R
        grid_T=gt_T.cpu().numpy()+delta_t1.numpy()
        grid_s=nocs_scale.cpu().numpy()





        fake_grid = grids['fake_grid'].numpy()
        boxsize = 1
        fake_grid = boxsize * (fake_grid)
        fake_grid_scaled=fake_grid*grid_s
        fake_query_np=fake_grid_scaled @ grid_rotation.T + grid_T

        fake_query=torch.from_numpy(fake_query_np).unsqueeze(0).float().cuda()
        fake_nocs=torch.from_numpy(fake_grid).unsqueeze(0).float().cuda()
        fake_num=fake_query.shape[1]



        if FLAGS.use_pick:
            regular_grid=self.regular_grid.unsqueeze(0)
            logits=pips_t.occnet(canonical_pc,regular_grid)
            weights=F.softmax(logits,dim=1)

            samples=torch.einsum('bps,bpi->bsi',weights,regular_grid)
            show_open3d(samples[0].detach().cpu().numpy(),canonical_pc[0].detach().cpu().numpy())
            pick_index=torch.topk(logits,FLAGS.gt_topk,1,largest=False)[1]
            for i in range(1):
                r=math.floor(FLAGS.gt_topk*(FLAGS.pick_ratio))
                random_index=torch.randperm(regular_grid.shape[1])[:FLAGS.gt_topk-r]
                pick_index[i,r:,0]=random_index


            regular_grid_pick=torch.gather(regular_grid,1,pick_index.expand(-1,-1,3))

            regular_grid_camera_pick=torch.bmm((regular_grid_pick*nocs_scale.reshape(-1,1,1)),gt_R.reshape(1,3,3).permute(0,2,1))+gt_T.reshape(-1,1,3)
            pred_dict=cat_model.qnet(regular_grid_camera_pick,feature_dict,pred_scale.detach())
            pred_coord=pred_dict['coord']
            # show_open3d(regular_grid_camera_pick[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
            Scale, Rotation, Translation, OutTransform,inlier_index=estimateSimilarityTransform(pred_coord[0].detach().cpu().numpy(),regular_grid_camera_pick[0].detach().cpu().numpy())
            # Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(np.transpose(pred_coord[0].detach().cpu().numpy()),
            #                                                                        np.transpose(regular_grid_camera_pick[0].detach().cpu().numpy()))

        else:
            pred_dict=cat_model.qnet(fake_query,feature_dict,pred_scale.detach())
            # show_open3d(fake_query[0].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
            pred_coord=pred_dict['coord'][0]
            pred_log_stds=pred_dict['log_stds'][0]
            pred_quat=pred_dict['quat'][0]
            pred_mask=torch.exp(torch.sum(pred_log_stds,dim=-1))

            pred_var_R=quaternion_to_matrix(pred_quat.reshape(-1,4)).contiguous()
            inv_stds=torch.diag_embed(torch.exp(-pred_log_stds.reshape(-1,3)))
            inv_sigma=torch.bmm(inv_stds,pred_var_R.permute(0,2,1))
            # r=torch.topk(pred_mask,100,largest=False)[1]
            #
            # fake_query=fake_query[0]
            # fake_nocs=fake_nocs[0]
            #
            # pred_coord=pred_coord[r]
            # fake_query=fake_query[r]
            # inv_sigma=inv_sigma[r]
            #
            # pred_coord=pred_coord
            # query_num=fake_query.shape[0]
            # pool_num=500
            # set_num=50
            # pool_query=torch.zeros(pool_num,set_num,3).cuda()
            # pool_coord=torch.zeros(pool_num,set_num,3).cuda()
            # pool_pred_coord=torch.zeros(pool_num,set_num,3).cuda()
            # pool_inv_sigma=torch.zeros(pool_num,set_num,3,3).cuda()
            # for i in range(pool_num):
            #     sample_idx = torch.randperm(query_num)[:set_num]
            #     # sample_idx = torch.tensor([2]*set_num).long()
            #     pool_query[i]=fake_query[sample_idx]
            #     pool_coord[i]=fake_nocs[sample_idx]
            #     pool_pred_coord[i]=pred_coord[sample_idx]
            #     pool_inv_sigma[i]=inv_sigma[sample_idx]
            #
            # # sample_idx_1=(torch.abs(fake_nocs[:,1]+0.0714)<0.001).nonzero()[:,0]
            # # pool_query[0]=fake_query[sample_idx_1]
            # # pool_coord[0]=fake_nocs[sample_idx_1]
            # # pool_pred_coord[0]=pred_coord[sample_idx_1]
            # # pool_inv_sigma[0]=inv_sigma[sample_idx_1]
            # #
            # # sample_idx_2=torch.randperm(query_num)[:set_num]
            # # pool_query[1]=fake_query[sample_idx_2]
            # # pool_coord[1]=fake_nocs[sample_idx_2]
            # # pool_pred_coord[1]=pred_coord[sample_idx_2]
            # # pool_inv_sigma[1]=inv_sigma[sample_idx_2]
            #
            #
            #
            # p=pool_coord
            # p_center=torch.mean(p,dim=1,keepdim=True)
            # p=p-p_center
            # p_norm=torch.norm(p,dim=-1,keepdim=True)
            # p_norm_mean=torch.mean(p_norm,dim=1,keepdim=True)
            # p=p/p_norm_mean
            # FT=torch.zeros(pool_num,set_num,3,6)
            # h=hat(p)
            # c=torch.einsum('psij,psjk->psik',-pool_inv_sigma,h)
            # FT[:,:,:,:3]=c
            # FT[:,:,:,3:]=pool_inv_sigma
            # C=torch.einsum('psij,psik->pjk',FT,FT)
            # values=torch.linalg.eigvalsh(C)
            # conds=values[:,0]/values[:,-1]
            # max_index=torch.max(conds,dim=0)[1]
            # min_index=torch.min(conds,dim=0)[1]
            # Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(np.transpose(pool_pred_coord[max_index].detach().cpu().numpy()),
            #                                                                        np.transpose(pool_query[max_index].detach().cpu().numpy()))
            # # fake_grid_transformed=((fake_grid*real_shape.cpu().numpy()) @ Rotation.T)+Translation
            # # show_open3d(pc_center[0].detach().cpu().numpy(),fake_grid_transformed)
            # show_open3d(pool_query[min_index].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
            # show_open3d(pool_query[max_index].detach().cpu().numpy(),pc_center[0].detach().cpu().numpy())
            # print(1)
            # use_topk=True
            # if use_topk:
            # Scale, Rotation, Translation, OutTransform,inlier_index=estimateSimilarityTransform(pred_coord[r].detach().cpu().numpy(),fake_query[0][r].detach().cpu().numpy())
            # else:
            Scale, Rotation, Translation, OutTransform,inlier_index=estimateSimilarityTransform(pred_coord.detach().cpu().numpy(),fake_query[0].detach().cpu().numpy())


        if FLAGS.verbose:
            vis_mask=False
            pred_mask_min=1e-6
            pred_mask_max=20e-6
            if vis_mask:

                pred_mask_np=pred_mask[0,:].detach().cpu().numpy()

                pred_mask_min=pred_mask_np.min()
                pred_mask_max=pred_mask_np.max()


                mask_norm=(pred_mask_np-pred_mask_min)/(pred_mask_max-pred_mask_min)
                cm=plt.cm.get_cmap('jet')
                mask_color=cm(mask_norm)[:,:3]
                show_open3d(pc_center[0].detach().cpu().numpy(),fake_query_np,color_2=mask_color)

                fake_grid_transformed=((fake_grid*real_shape.cpu().numpy()) @ Rotation.T)+Translation
                show_open3d(pc_center[0].detach().cpu().numpy(),fake_grid_transformed)

            vis_error=True
            if vis_error:
                from scipy.stats import multivariate_normal
                x_np, y_np = np.mgrid[-0.6:0.6:40j,
                           -0.6:0.6:40j]
                x_grid_np, y_grid_np = np.mgrid[-0.6:0.6:19j,
                             -0.6:0.6:19j]
                split=x_np.shape[0]
                grid_split=x_grid_np.shape[0]
                x=torch.from_numpy(x_np).float().cuda()
                y=torch.from_numpy(y_np).float().cuda()
                x_grid=torch.from_numpy(x_grid_np).float().cuda()
                y_grid=torch.from_numpy(y_grid_np).float().cuda()
                z=torch.zeros_like(x)
                z_grid=torch.zeros_like(x_grid)
                plane_query_nocs=torch.stack([x,y,z],dim=-1).reshape(1,-1,3)
                grid_query_nocs=torch.stack([x_grid,y_grid,z_grid],dim=-1).reshape(1,-1,3)
                # grid_var=pips_t.occnet(canonical_pc,grid_query_nocs)

                plane_query=(plane_query_nocs*nocs_scale) @ gt_R.T + gt_T
                plane_dict=cat_model.qnet(plane_query,feature_dict,pred_scale)
                plane_coord=plane_dict['coord']
                plane_log_stds=plane_dict['log_stds']
                plane_quat=plane_dict['quat']

                plane_var_R=quaternion_to_matrix(plane_quat.reshape(-1,4)).contiguous()
                stds=torch.diag_embed(torch.exp(plane_log_stds.reshape(-1,3)))
                sigma=torch.bmm(stds,plane_var_R.permute(0,2,1))
                cov=torch.bmm(sigma.permute(0,2,1),sigma)
                cov2d=cov[:,:2,:2].detach().cpu().numpy()

                index1=(10,20)
                index2=(25,20)

                index1_flat=index1[0]*40+index1[1]
                index2_flat=index2[0]*40+index2[1]
                mean1=[x_np[index1[0]][0],y_np[0][index1[1]]]
                mean2=[x_np[index2[0]][0],y_np[0][index2[1]]]

                pos = np.vstack((x_np.flatten(), y_np.flatten())).T

                rv1 = multivariate_normal(mean1, cov2d[index1_flat])
                density1 = rv1.pdf(pos)
                rv2 = multivariate_normal(mean2, cov2d[index2_flat])
                density2 = rv2.pdf(pos)

                density1=density1.reshape(40,40)
                density2=density2.reshape(40,40)








                plane_mask=torch.exp(torch.sum(plane_log_stds,dim=-1))
                plane_mask_np=plane_mask.reshape(split,split).detach().cpu().numpy()

                # grid_var_np=grid_var.reshape(grid_split,grid_split).detach().cpu().numpy()

                x_np=x.detach().cpu().numpy()
                y_np=y.detach().cpu().numpy()

                surf_nocs=(pc_center-gt_T) @ gt_R / nocs_scale
                surf_nocs_xy=surf_nocs[torch.abs(surf_nocs[:,:,2])<0.1]
                surf_nocs_xy_np=surf_nocs_xy.detach().cpu().numpy()
                error=torch.norm(plane_coord-plane_query_nocs,dim=-1)
                error_np=error.reshape(split,split).detach().cpu().numpy()
                # c = plt.pcolormesh(x_np, y_np, plane_mask_np, cmap ='Greens', vmin = pred_mask_min, vmax = pred_mask_max)
                fig=plt.figure(figsize=(30,4))
                ax0=fig.add_subplot(141,projection='3d')
                ax1=fig.add_subplot(142)
                plt.xlim(-0.6,0.6)
                ax2=fig.add_subplot(143)
                plt.xlim(-0.6,0.6)
                ax3=fig.add_subplot(144)
                plt.xlim(-0.6,0.6)
                pc1=ax1.pcolormesh(x_np, y_np, plane_mask_np, cmap ='Greens', vmin = pred_mask_min, vmax = pred_mask_max)
                # pc2=ax2.pcolormesh(x_grid_np, y_grid_np, grid_var_np, cmap ='Greens', vmin = 0, vmax = 1)
                pc3=ax3.pcolormesh(x_np, y_np, error_np, cmap ='Reds', vmin = 0, vmax = 0.3)
                ax1.scatter(surf_nocs_xy_np[:,0], surf_nocs_xy_np[:,1],color='black')
                ax2.scatter(surf_nocs_xy_np[:,0], surf_nocs_xy_np[:,1],color='black')
                ax3.scatter(surf_nocs_xy_np[:,0], surf_nocs_xy_np[:,1],color='black')
                import mpl_toolkits.mplot3d.axes3d as p3



                ax0.set_box_aspect([1,1,1])
                limit=0.5
                ax0.set(xlim3d=(-limit, limit), xlabel='X')
                ax0.set(ylim3d=(-limit, limit), ylabel='Y')
                ax0.set(zlim3d=(-limit, limit), zlabel='Z')
                azim=170
                dist=90
                elev=31

                pc=canonical_pc[0].cpu().numpy()
                pc_x=pc[:,0]
                pc_y=pc[:,2]
                pc_z=pc[:,1]
                ax0.scatter(pc_x, pc_y, pc_z,s=0.6)


                ax1.contour(x_np, y_np, density1, levels=[0.05], colors='k')
                ax1.contour(x_np, y_np, density2, levels=[0.05], colors='k')

                fig.colorbar(pc1,ax=ax1)
                # fig.colorbar(pc2,ax=ax2)
                fig.colorbar(pc3,ax=ax3)
                plt.show()
                plt.close()


        res = torch.eye(4, dtype=torch.float)
        try:
            res[:3,:3]=torch.from_numpy(Rotation)
            res[:3,3]=torch.from_numpy(Translation)+center.reshape(3).cpu()
        except:
            pass
        cur_s=real_shape


        return res,cur_s

def to_value(cat_name,sym,bin_size,fake_bin):
    ratio_x=ratio_dict[cat_name][0]
    ratio_y=ratio_dict[cat_name][1]
    ratio_z=ratio_dict[cat_name][2]
    fake_nocs=fake_bin.clone().float()
    if sym==1:
        x_bin_resolution=FLAGS.pad_radius/bin_size*ratio_x
        y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
        x_start=0
        y_start=(-FLAGS.pad_radius)*ratio_y
        z_start=0
        z_bin_resolution=0
    else:
        x_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_x
        y_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_y
        z_bin_resolution=2*FLAGS.pad_radius/bin_size*ratio_z
        x_start=(-FLAGS.pad_radius)*ratio_x
        y_start=(-FLAGS.pad_radius)*ratio_y
        z_start=(-FLAGS.pad_radius)*ratio_z
    fake_nocs[:,:,0]=fake_bin[:,:,0]*x_bin_resolution+x_start
    fake_nocs[:,:,1]=fake_bin[:,:,1]*y_bin_resolution+y_start
    fake_nocs[:,:,2]=fake_bin[:,:,2]*z_bin_resolution+z_start
    return fake_nocs

def to_bin(cat_name,sym,bin_size,pad_nocs):
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


def scale2to3(new_scale):
    if new_scale.shape[-1]==2:
        tmp_new_scale=torch.zeros_like(new_scale)
        tmp_new_scale[:2]=new_scale[:2]
        tmp_new_scale=torch.cat([tmp_new_scale,new_scale[0:1]],dim=-1)
    else:
        tmp_new_scale=new_scale
    return tmp_new_scale
