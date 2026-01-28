import os

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import absl.flags as flags
FLAGS = flags.FLAGS
from pytorch3d.transforms import so3_log_map,so3_exponential_map
from evaluation.utils.eval_utils import *
from nfmodel.uti_tool import *
from network.point_sample.pc_sample import *
from datasets.data_augmentation import *
# from nfmodel.torch_util import *
def optimize_pose(init_r,init_t,source,target,nocs_scale,sym_trans):
    source=source.detach().cpu()
    target=target.detach().cpu()
    sym_trans=sym_trans.cpu()
    # new_Rvec=torch.from_numpy(cv2.Rodrigues(init_r[0])[0][:,0]).float().requires_grad_()
    new_Rvec=so3_log_map(torch.from_numpy(init_r)).float().requires_grad_()
    new_T=torch.from_numpy(init_t.astype(np.float32)).float().requires_grad_()

    params=[{'params':new_Rvec},{'params':new_T}]
    opt_1 = torch.optim.Adam(params, lr=0.01)
    step=100

    def objective():
        cur_R=so3_exponential_map(new_Rvec)
        cur_t=new_T
        cal_fake_nocs=torch.bmm(target-cur_t.reshape(-1,1,3),cur_R.reshape(-1,3,3))/nocs_scale
        if len(sym_trans)>1:
            cal_fake_nocs_c=torch.zeros_like(cal_fake_nocs)
            cal_fake_nocs_c[:,:,0]=torch.abs(cal_fake_nocs[:,:,0])
            cal_fake_nocs_c[:,:,1]=torch.abs(cal_fake_nocs[:,:,1])
            cal_fake_nocs_c[:,:,2]=cal_fake_nocs[:,:,2]
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







def eval_fun(network_dict,test_dataset,model_size_dict,model_points_dict,mesh_dict,sym_trans_dict):
    results=[]
    preds=[]
    bs=1

    for data in tqdm(test_dataset):

        sym=[0]

        mask=data['roi_mask'].unsqueeze(0).cuda()
        depth=data['roi_depth'].unsqueeze(0).cuda()
        coord=data['roi_coord_2d'].unsqueeze(0).cuda()
        scene_id=data['scene_id']
        im_id=data['img_id']

        # if im_id not in [33,125,178]:
        #     continue

        obj_id=data['obj_id']
        mesh=mesh_dict[obj_id]

        if data['invalid']==1:
            res = torch.eye(4, dtype=torch.float)
            pred = dict(scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        score=1,
                        R=res[:3,:3].numpy(),
                        t=(res[:3,3].numpy())*1000,
                        time=0)

            preds.append(pred)
            print('skip invalid')
            continue




        model_size=model_size_dict[obj_id]/1000.0
        model_size=model_size[None,:].repeat(bs,1)
        model_points=model_points_dict[obj_id]/1000.0
        model_points=model_points.numpy()
        nocs_scale=torch.norm(model_size,dim=-1)
        sym_trans=sym_trans_dict[obj_id]
        network=network_dict[obj_id]



        camK=data['cam_K'].unsqueeze(0).cuda()
        PC = PC_sample(mask, depth, camK, coord)
        points_defor = torch.randn(PC.shape).cuda()
        PC = PC + torch.clip(points_defor * 0.001,-0.002,0.002)
        bs=PC.shape[0]



        gt_R=data['rotation'].cuda()
        gt_T=data['translation'].cuda()/1000
        center=PC.mean(dim=1,keepdim=True)
        gt_T=gt_T-center.squeeze()


        pc_num=PC.shape[1]
        pc_center=PC-center
        eval_start_time = time.time()
        with torch.no_grad():
            feature_dict= network.backbone1(pc_center)
        pred_scale=torch.tensor([[1.0]]).cuda()
        feat_for_ts = pc_center
        # T,s= network.ts(feat_for_ts)
        #

        # pred_t=T[0].detach().cpu().numpy()
        pred_t=np.zeros_like(gt_T.cpu().numpy())
        # pred_t=gt_T.detach().cpu().numpy()

        points_uniform = np.random.rand(1500, 3)
        points_uniform = 1.2 * (points_uniform - 0.5)
        points_r=np.linalg.norm(points_uniform,axis=-1)
        # if sym[0]==0:
        points_uniform=points_uniform[points_r<0.6]
        points_uniform=points_uniform*nocs_scale.cpu().numpy()+pred_t
        points_uniform=torch.from_numpy(points_uniform).float()
        fake_query=points_uniform.unsqueeze(0).float().cuda()

        # show_open3d(fake_query[0].cpu().numpy(),pc_center[0].cpu().numpy())
        fake_num=fake_query.shape[1]
        with torch.no_grad():
            pred_dict=network.qnet(fake_query,feature_dict,pred_scale.detach())

        if FLAGS.stage==1:
            pred_camera=pred_dict['coord']
            pred_log_stds=pred_dict['log_stds'].detach()
            # inv_stds=torch.exp(-pred_log_stds.reshape(-1,3))
            # inv_stds_mean=torch.mean(inv_stds,dim=-1,keepdim=True)/network.loss_coord.mean_inv_std
            # inv_stds_mean=inv_stds_mean.reshape(1,-1,1)
            # # choose_by_var=(inv_stds_mean>0.5).nonzero()[:,1].reshape(1,-1)
            # #
            # choose_by_var=torch.topk(inv_stds_mean[:,:,0],k=fake_num//2,dim=-1,largest=True)[1]


            pred_log_stds_sum=pred_log_stds.sum(-1)
            pred_stds=torch.exp(pred_log_stds_sum)
            choose_by_var=torch.topk(pred_stds,k=fake_num//2,dim=-1,largest=False)[1]


            fake_query_choose=torch.gather(fake_query,1,choose_by_var[:,:,None].repeat(1,1,3))
            pred_camera_choose=torch.gather(pred_camera,1,choose_by_var[:,:,None].repeat(1,1,3))
            pred_camera=pred_camera_choose
            fake_query=fake_query_choose
            # show_open3d(fake_query[0].cpu().numpy(),pc_center[0].cpu().numpy())

            do_opti=True
            # if len(sym_trans)>1:
            #     do_opti=True
            # else:
            #     Scale, Rotation, Translation, OutTransform,inlier_index=estimateSimilarityTransform(
            #         pred_camera[0].detach().cpu().numpy(),fake_query[0].detach().cpu().numpy())
            #     if Scale==None:
            #         do_opti=True
            if do_opti:
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

                # init_R=gt_R.cpu().numpy()
                init_R=noise_Rs.numpy()
                init_t=np.repeat(pred_t[None,:],opti_bs,0)
                # init_t=gt_T.cpu().numpy()

                # show_open3d(pred_camera[0].detach().cpu().numpy(),cal_gt_nocs[0].detach().cpu().numpy())




                Rotation, Translation=optimize_pose(init_R,init_t,pred_camera.detach(),fake_query,nocs_scale,sym_trans)

            # model_transformed=((model_points) @ (Rotation.T))+Translation
            # show_open3d(pc_center[0].detach().cpu().numpy(),model_transformed)

        # Rotation=gt_R.cpu().numpy()
        # Translation=pred_t



        elif FLAGS.stage==2:
            rgb=data['rgb']
            if not os.path.exists(FLAGS.pic_save_dir):
                os.makedirs(FLAGS.pic_save_dir)
            rgb_path=os.path.join(FLAGS.pic_save_dir,str(im_id)+'_rgb.png')
            cv2.imwrite(rgb_path,rgb )

            pred_coord=pred_dict['coord'].detach()
            z_inv=pred_dict['z_inv'].detach()
            pred_log_stds=pred_dict['log_stds'].detach()
            pred_rot_1=pred_dict['rot_vec_1'].detach()
            pred_rot_2=pred_dict['rot_vec_2'].detach()
            from tools.rot_utils import get_rot_mat_y_first
            pred_var_R=get_rot_mat_y_first(pred_rot_1,pred_rot_2).reshape(-1,3,3)
            inv_stds=torch.exp(-pred_log_stds.reshape(-1,3))
            inv_stds_ma=torch.diag_embed(inv_stds)
            inv_sigma=torch.bmm(inv_stds_ma,pred_var_R.permute(0,2,1)).reshape(1,-1,3,3)
            inv_sigma=inv_sigma/network.loss_coord.mean_inv_std
            nocs_scale=nocs_scale.cuda()
            fake_nocs=((fake_query[0]-gt_T) @ gt_R)/nocs_scale
            pc_nocs=((pc_center[0]-gt_T) @ gt_R)/nocs_scale
            fake_error=torch.norm(pred_coord-fake_nocs,dim=-1)/0.3
            model_points=torch.from_numpy(model_points).float()
            model_transformed=((model_points.cuda()) @ (gt_R.T))+gt_T
            # show_open3d(model_transformed.cpu().numpy(),pc_center[0].cpu().numpy())
            # show_open3d(mesh.vertices,pc_center[0].cpu().numpy())



            intrinsic=camK[0].cpu().numpy()


            pred_rt=get_noise_rt(gt_R.cpu(),(gt_T+center.squeeze()).cpu(),16,0.04).numpy()

            # draw_pose_mesh(rgb.copy(),mesh,intrinsic,pred_rt,)

            draw_pose(rgb.copy(),model_points.cpu().numpy(),intrinsic,pred_rt,
                      outpath=os.path.join(FLAGS.pic_save_dir,str(im_id)+f'_pred_pose_1_{obj_id}.jpg'))


            query_color=show_error_3(pc_nocs.cpu().numpy(),fake_nocs.cpu().numpy(),
                                     error=fake_error[0].cpu().numpy(),
                                     outpath=os.path.join(FLAGS.pic_save_dir,str(im_id)+f'_error_1_{obj_id}.jpg'))

            result_dict={}
            result_dict['query_color']=query_color
            result_dict['query_points']=fake_nocs.cpu().numpy()
            result_dict['input_points']=pc_nocs.cpu().numpy()
            result_dict['query_error']=fake_error.cpu().numpy()

            neighbors=get_neighbor(fake_query[0],model_transformed,1)[:,0,:]
            neighbors_distance=torch.norm(neighbors-fake_query,dim=-1)
            fake_query_choose=fake_query[neighbors_distance<0.06]
            inv_sigma_choose=inv_sigma[neighbors_distance<0.06]
            fake_nocs_choose=fake_nocs.unsqueeze(0)[neighbors_distance<0.06]
            fake_error_choose=fake_error[neighbors_distance<0.06]
            result_dict['choose_idx']=(neighbors_distance<0.06).nonzero()[:,1].cpu().numpy()



            show_error_3(pc_nocs.cpu().numpy(),fake_nocs_choose.cpu().numpy(),
                         error=fake_error_choose.cpu().numpy(),
                         outpath=os.path.join(FLAGS.pic_save_dir,str(im_id)+f'_error_2_{obj_id}.jpg'))

            pred_rt=get_noise_rt(gt_R.cpu(),(gt_T+center.squeeze()).cpu(),8,0.02).numpy()
            draw_pose(rgb.copy(),model_points.cpu().numpy(),intrinsic,pred_rt,
                      outpath=os.path.join(FLAGS.pic_save_dir,str(im_id)+f'_pred_pose_2_{obj_id}.jpg'))

            sample_idx=pick_stable(fake_nocs_choose,inv_sigma_choose)
            fake_nocs_stable=fake_nocs_choose[sample_idx]
            fake_error_stable=fake_error_choose[sample_idx]

            result_dict['stable_idx']=sample_idx.cpu().numpy()
            np.savez(os.path.join(FLAGS.pic_save_dir,str(im_id)+f'_{obj_id}.npz'),
                     **result_dict)

            show_error_3(pc_nocs.cpu().numpy(),fake_nocs_stable.cpu().numpy(),
                         error=fake_error_stable.cpu().numpy(),
                         outpath=os.path.join(FLAGS.pic_save_dir,str(im_id)+f'_error_3_{obj_id}.jpg'))

            pred_rt=get_noise_rt(gt_R.cpu(),(gt_T+center.squeeze()).cpu(),4,0.01).numpy()
            draw_pose(rgb.copy(),model_points.cpu().numpy(),intrinsic,pred_rt,
                      outpath=os.path.join(FLAGS.pic_save_dir,str(im_id)+f'_pred_pose_3_{obj_id}.jpg'))



        eval_end_time = time.time()
        eval_time = eval_end_time - eval_start_time
        #
        # gt_R=data['rotation']
        # gt_T=data['translation']



        res = torch.eye(4, dtype=torch.float)
        try:
            res[:3,:3]=torch.from_numpy(Rotation)
            res[:3,3]=torch.from_numpy(Translation)



        except:
            print('error !!')
            pass

        pred = dict(scene_id=scene_id,
                        im_id=im_id,
                        obj_id=obj_id,
                        score=1,
                        R=res[:3,:3].numpy(),
                        t=(res[:3,3].numpy()+center.squeeze().cpu().numpy())*1000,
                        time=eval_time)

        preds.append(pred)


        regress_r_diff=modelnet_r_diff(gt_R.cpu(),res[:3,:3],sym[0]).item()
        regress_t_diff=torch.norm(res[:3,3] - gt_T.cpu(), dim=-1).item()
        result={}
        result['regress_r_error']=regress_r_diff
        result['regress_t_error']=regress_t_diff
        results.append(result)
    return results,preds



def save_bop_results(path, results, version='bop19'):
    """Saves 6D object pose estimates to a file.
    :param path: Path to the output file.
    :param results: Dictionary with pose estimates.
    :param version: Version of the results.
    """
    # See docs/bop_challenge_2019.md for details.
    if version == 'bop19':
        lines = ['scene_id,im_id,obj_id,score,R,t,time']
        for res in results:
            if 'time' in res:
                run_time = res['time']
            else:
                run_time = -1

            lines.append('{scene_id},{im_id},{obj_id},{score},{R},{t},{time}'.format(
                scene_id=res['scene_id'],
                im_id=res['im_id'],
                obj_id=res['obj_id'],
                score=res['score'],
                R=' '.join(map(str, res['R'].flatten().tolist())),
                t=' '.join(map(str, res['t'].flatten().tolist())),
                time=run_time))

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

    else:
        raise ValueError('Unknown version of BOP results.')




def draw_pose(image,model, camera_intrinsics, transform_matrix, outpath=None):


    # 读取真实的RGB图像
    image_height, image_width = image.shape[:2]
    vertices=model
    transform_3d=transform_coordinates_3d(vertices.transpose(), transform_matrix)
    proj_2d= calculate_2d_projections(transform_3d, camera_intrinsics)

    scale_factor=4
    original_shape = image.shape
    high_res_shape = (original_shape[0] * scale_factor, original_shape[1] * scale_factor)
    high_res_image = np.zeros((high_res_shape[0], high_res_shape[1], 3), dtype=np.uint8)
    mask = np.zeros((high_res_shape[0], high_res_shape[1]), dtype=np.uint8)
    for point in proj_2d:
        high_res_point = (np.array(point) * scale_factor).astype(int)
        cv2.circle(high_res_image, tuple(high_res_point), 1, (0, 255, 0), 3)
        cv2.circle(mask, tuple(high_res_point),1, 255, 3)
    resized_image = cv2.resize(high_res_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

    mask_indices = resized_mask > 0  # 获取需要融合的像素位置
    final_image = image.copy()
    final_image[mask_indices] = resized_image[mask_indices]
    # final_image = cv2.addWeighted(image, 0.5, resized_image, 0.5, 0)
    cv2.imwrite(outpath, final_image)
    return


def draw_pose_mesh(image,mesh, camera_intrinsics, transform_matrix, alpha=0.5):
    import cv2
    import numpy as np
    import pyrender
    import trimesh

    # 读取真实的RGB图像
    image_height, image_width = image.shape[:2]

    # 为每个顶点生成随机颜色
    # random_colors = np.random.rand(len(mesh.vertices), 4)  # 生成RGBA颜色
    # random_colors[:, 3] = 1.0  # 设置完全不透明
    # mesh.visual.vertex_colors = random_colors

    # 创建场景
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh, pose=transform_matrix)

    fx = camera_intrinsics[0, 0]/1000  # x轴焦距
    fy = camera_intrinsics[1, 1]/1000
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # 图像尺寸
    width = 640
    height = 480

    # 计算垂直视场（FOV）
    fov_y = 2 * np.arctan(height / (2 * fy))

    # 计算长宽比（Aspect Ratio）
    aspect_ratio = width / height

    # 创建PerspectiveCamera
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=aspect_ratio)


    # 根据相机内参创建相机
    # camera = pyrender.IntrinsicsCamera(fx=camera_intrinsics[0, 0], fy=camera_intrinsics[1, 1],
    #                                    cx=camera_intrinsics[0, 2], cy=camera_intrinsics[1, 2])
    scene.add(camera, pose=np.eye(4))

    # 创建渲染器
    r = pyrender.OffscreenRenderer(image_width, image_height)
    color, depth = r.render(scene)

    # 将3D模型渲染为半透明并叠加到RGB图像上
    for i in range(3):  # 对于每个颜色通道
        image[:, :, i] = (1 - alpha) * image[:, :, i] + alpha * color[:, :, i]



    fig,ax=plt.subplots()
    im=ax.imshow(image)
    plt.show()
    # 保存或显示结果
    # cv2.imshow('Rendered Image', image)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




def eval_train(network_dict,test_dataset_list,model_size_dict,model_points_dict,mesh_dict,sym_trans_dict,logger):
    for k,network in network_dict.items():
        network.eval()
    for test_dataset in test_dataset_list:
        cur_subset=' '
        results,preds=eval_fun(network_dict,test_dataset,model_size_dict,model_points_dict,mesh_dict,sym_trans_dict)
        track_dict={'regress_r_error':[],
                    'regress_t_error':[],
                    'regress_5deg':[],
                    'regress_5cm': [],
                    'regress_5deg5cm':[],
                    'regress_10deg':[],
                    'regress_10deg5cm':[],
                    }
        for result in results:
            for key in ['regress_r_error', 'regress_t_error']:
                track_dict[key].append(result[key])
            regress_5deg = float(result['regress_r_error'] <= 5.0)
            regress_10deg = float(result['regress_r_error'] <= 10.0)
            regress_cm = float(result['regress_t_error'] <= 0.05)
            regress_5degcm = regress_5deg * regress_cm
            regress_10degcm = regress_10deg * regress_cm
            track_dict['regress_5deg'].append(regress_5deg)
            track_dict['regress_10deg'].append(regress_10deg)
            track_dict['regress_5cm'].append(regress_cm)
            track_dict['regress_5deg5cm'].append(regress_5degcm)
            track_dict['regress_10deg5cm'].append(regress_10degcm)

        logger.info('{0} regress: mean_r: {1:f}, median_r: {2:f} '
                    'mean_t: {3:f} , median_t : {4:f}, 5deg: {5:f} 10deg: {6:f} '
                    '5cm: {7:f} 5deg5cm: {8:f} 10deg5cm: {9:f}'.format(cur_subset,
            np.mean(np.array(track_dict['regress_r_error'])),
            np.median(np.array(track_dict['regress_r_error'])),
            np.mean(np.array(track_dict['regress_t_error'])),
            np.median(np.array(track_dict['regress_t_error'])),
            sum(track_dict['regress_5deg'])/len(track_dict['regress_5deg']),
            sum(track_dict['regress_10deg'])/len(track_dict['regress_10deg']),
            sum(track_dict['regress_5cm'])/len(track_dict['regress_5cm']),
            sum(track_dict['regress_5deg5cm'])/len(track_dict['regress_5deg5cm']),
            sum(track_dict['regress_10deg5cm'])/len(track_dict['regress_10deg5cm']),
            ))
        csv_path=os.path.join(FLAGS.eval_out,'pips_lmo-test.csv')
        save_bop_results(csv_path,preds)
    network.train()