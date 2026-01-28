import numpy as np

import torch
import absl.flags as flags
FLAGS = flags.FLAGS
from pytorch3d.ops import sample_farthest_points
from nfmodel.uti_tool import *

def eval_fun(network,test_dataset):
    results=[]
    for data in tqdm(test_dataset):
        sym=data['sym']
        cloud=data['cloud'].cuda()
        model_point=data['model_point']
        gt_R=data['rotation'].cuda()
        gt_T=data['translation'].cuda()
        gt_s=data['scale'].cuda()
        model_id=data['instance']
        img_id=data['img_id']

        # cat2img_list={'02691156':[468,658,1163,1172],
        #               '03001627':[10,469,660,1160,1175],
        #               '04256520':[0,475,653,655,1162]
        # }
        #
        # img_list=cat2img_list[FLAGS.per_obj]
        # if img_id not in img_list:
        #     continue
        # print(img_id)
        cam_K=data['cam_k']
        with torch.no_grad():
            PC, _ = sample_farthest_points(cloud.unsqueeze(0), K=FLAGS.random_points)

        # show_open3d(PC[0].cpu().numpy(),PC[0].cpu().numpy())

        center=PC.mean(dim=1,keepdim=True)
        gt_T=gt_T-center.squeeze()
        pc_num=PC.shape[1]
        pc_center=PC-center
        with torch.no_grad():
            feature_dict= network.backbone1(pc_center)
        pred_scale=torch.tensor([[1.0]]).cuda()
        feat_for_ts = pc_center
        T,s= network.ts(feat_for_ts)

        pred_t=T[0].detach().cpu().numpy()

        points_uniform = np.random.rand(1500, 3)
        points_uniform = 1.2 * (points_uniform - 0.5)
        points_r=np.linalg.norm(points_uniform,axis=-1)
        # if sym[0]==0:
        points_uniform=points_uniform[points_r<0.6]
        points_uniform=points_uniform+pred_t
        points_uniform=torch.from_numpy(points_uniform).float()
        fake_query=points_uniform.unsqueeze(0).float().cuda()
        fake_num=fake_query.shape[1]
        with torch.no_grad():
            pred_dict=network.qnet(fake_query,feature_dict,pred_scale.detach())
        # FLAGS.stage=1
        if FLAGS.stage==1:

            pred_camera=pred_dict['coord']
            pred_log_stds=pred_dict['log_stds'].detach()
            pred_log_stds_sum=pred_log_stds.sum(-1)
            pred_stds=torch.exp(pred_log_stds_sum)
            choose_by_var=torch.topk(pred_stds,k=fake_num//2,dim=-1,largest=False)[1]
            # z_so3_choose=torch.gather(z_so3,1,choose_by_var[:,:,None,None].repeat(1,1,z_inv.shape[2],3))
            fake_query_choose=torch.gather(fake_query,1,choose_by_var[:,:,None].repeat(1,1,3))
            pred_camera_choose=torch.gather(pred_camera,1,choose_by_var[:,:,None].repeat(1,1,3))
            pred_camera=pred_camera_choose
            fake_query=fake_query_choose


            Scale, Rotation, Translation, OutTransform,inlier_index=estimateSimilarityTransform(
                pred_camera[0].detach().cpu().numpy(),fake_query[0].detach().cpu().numpy())
        elif FLAGS.stage==2:
            rgb=data['rgb'].numpy()

            if not os.path.exists(os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj)):
                os.makedirs(os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj))
            rgb_path=os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+'_rgb.png')
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
            nocs_scale=1
            fake_nocs=((fake_query[0]-gt_T) @ gt_R)/nocs_scale
            pc_nocs=((pc_center[0]-gt_T) @ gt_R)/nocs_scale
            fake_error=torch.norm(pred_coord-fake_nocs,dim=-1)/0.3
            model_transformed=((model_point.cuda()) @ (gt_R.T))+gt_T

            intrinsic=cam_K
            pred_rt=get_noise_rt(gt_R.cpu(),(gt_T+center.squeeze()).cpu(),16,0.04).numpy()
            draw_pose(rgb.copy(),model_point.cpu().numpy(),intrinsic,pred_rt,
                      outpath=os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+f'_pred_pose_1_{model_id}.jpg'))

            query_color=show_error_3(pc_nocs.cpu().numpy(),fake_nocs.cpu().numpy(),
                                     error=fake_error[0].cpu().numpy(),
                                     outpath=os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+f'_error_1_{model_id}.jpg'))
            result_dict={}
            result_dict['query_color']=query_color
            result_dict['query_points']=fake_nocs.cpu().numpy()
            result_dict['input_points']=pc_nocs.cpu().numpy()
            result_dict['query_error']=fake_error.cpu().numpy()

            neighbors=get_neighbor(fake_query[0],model_transformed,1)[:,0,:]
            neighbors_distance=torch.norm(neighbors-fake_query,dim=-1)

            fake_query_choose=fake_query[neighbors_distance<0.2]
            inv_sigma_choose=inv_sigma[neighbors_distance<0.2]
            fake_nocs_choose=fake_nocs.unsqueeze(0)[neighbors_distance<0.2]
            fake_error_choose=fake_error[neighbors_distance<0.2]
            result_dict['choose_idx']=(neighbors_distance<0.2).nonzero()[:,1].cpu().numpy()

            show_error_3(pc_nocs.cpu().numpy(),fake_nocs_choose.cpu().numpy(),
                         error=fake_error_choose.cpu().numpy(),
                         outpath=os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+f'_error_2_{model_id}.jpg'))

            pred_rt=get_noise_rt(gt_R.cpu(),(gt_T+center.squeeze()).cpu(),8,0.02).numpy()
            draw_pose(rgb.copy(),model_point.cpu().numpy(),intrinsic,pred_rt,
                      outpath=os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+f'_pred_pose_2_{model_id}.jpg'))

            sample_idx=pick_stable(fake_nocs_choose,inv_sigma_choose)
            fake_nocs_stable=fake_nocs_choose[sample_idx]
            fake_error_stable=fake_error_choose[sample_idx]

            result_dict['stable_idx']=sample_idx.cpu().numpy()
            np.savez(os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+f'_{model_id}.npz'),
                     **result_dict)

            show_error_3(pc_nocs.cpu().numpy(),fake_nocs_stable.cpu().numpy(),
                         error=fake_error_stable.cpu().numpy(),
                         outpath=os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+f'_error_3_{model_id}.jpg'))

            pred_rt=get_noise_rt(gt_R.cpu(),(gt_T+center.squeeze()).cpu(),4,0.01).numpy()
            draw_pose(rgb.copy(),model_point.cpu().numpy(),intrinsic,pred_rt,
                      outpath=os.path.join(FLAGS.pic_save_dir,FLAGS.per_obj,str(img_id)+f'_pred_pose_3_{model_id}.jpg'))


            # show_open3d(pc_center[0].detach().cpu().numpy(),model_transformed.cpu().numpy())

        # model_transformed=((model_point) @ (Rotation.T))+Translation
        # show_open3d(pc_center[0].detach().cpu().numpy(),model_transformed.cpu().numpy())

        res = torch.eye(4, dtype=torch.float).cuda()
        try:
            res[:3,:3]=torch.from_numpy(Rotation).cuda()
            res[:3,3]=torch.from_numpy(Translation).cuda()
        except:
            pass
        regress_r_diff=modelnet_r_diff(gt_R,res[:3,:3],sym[0]).item()
        regress_t_diff=torch.norm(res[:3,3] - gt_T, dim=-1).item()
        result={}
        result['regress_r_error']=regress_r_diff
        result['regress_t_error']=regress_t_diff
        result['gt_R']=gt_R.reshape(9).tolist()
        result['gt_T']=(gt_T+center.squeeze()).reshape(3).tolist()
        result['pred_R']=res[:3,:3].reshape(9).tolist()
        result['pred_T']=(res[:3,3]+center.squeeze()).reshape(3).tolist()
        result['img_id']=img_id
        result['model_id']=model_id
        results.append(result)
    return results




def eval_train(network,test_dataset_list,logger):
    network = network.eval()
    for test_dataset in test_dataset_list:
        cur_subset=test_dataset.subset
        results=eval_fun(network,test_dataset)
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
    network.train()

def eval_test(network,test_dataset):
    network = network.eval()

    results=eval_fun(network,test_dataset)
    out_dir=FLAGS.eval_out
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_path=os.path.join(out_dir,'pred_results.txt')
    import json
    with open(out_path,'w') as f:
        json.dump(results,f)

def eval_test_noise(network,test_dataset_list):
    network = network.eval()

    for test_dataset in test_dataset_list:
        cur_subset=test_dataset.subset
        results=eval_fun(network,test_dataset)
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

        print('{0} regress: mean_r: {1:f}, median_r: {2:f} '
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

from evaluation.utils.eval_utils import *
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
        cv2.circle(high_res_image, tuple(high_res_point), 1, (0, 255, 0), 9)
        cv2.circle(mask, tuple(high_res_point),1, 255, 9)
    resized_image = cv2.resize(high_res_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

    mask_indices = resized_mask > 0  # 获取需要融合的像素位置
    final_image = image.copy()
    final_image[mask_indices] = resized_image[mask_indices]
    # final_image = cv2.addWeighted(image, 0.5, resized_image, 0.5, 0)
    cv2.imwrite(outpath, final_image)
    return

