import os

import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import absl.flags as flags
FLAGS = flags.FLAGS
import _pickle as cPickle
from pytorch3d.transforms import so3_log_map,so3_exponential_map
from evaluation.utils.eval_utils import *
from nfmodel.uti_tool import *
from network.point_sample.pc_sample import *
from datasets.data_augmentation import *
# from nfmodel.torch_util import *
import json
import open3d as o3d
from absl import app


def save_result(cur_out_dir,img_dict,vae):
    trans_scale_dict={}
    var_max=-1
    obj_max=-1
    for obj_index in img_dict.keys():
        var=img_dict[obj_index]['var']
        if var>var_max:
            var_max=var
            obj_max=obj_index
    for obj_index in img_dict.keys():
        z_so3=img_dict[obj_index]['z_so3']
        z_inv=img_dict[obj_index]['z_inv']
        trans_pred=img_dict[obj_index]['trans']
        scale_pred=img_dict[obj_index]['scale']
        var=img_dict[obj_index]['var']

        pick_var=False
        if pick_var:
            if obj_index!=obj_max:
                z_so3=z_so3[0:1]
                z_inv=z_inv[0:1]
        else:
            z_so3=z_so3[0:1]
            z_inv=z_inv[0:1]

        for j in range(len(z_so3)):
            embedding = {
                "z_so3": z_so3[j:j+1],
                "z_inv": z_inv[j:j+1],
            }
            latent_path_equi=os.path.join(cur_out_dir,f'{obj_index}_{j}.npy')
            latent_path_inv=os.path.join(cur_out_dir,f'{obj_index}_{j}_inv.npy')
            np.save(latent_path_equi,z_so3[j:j+1].detach().cpu().numpy())
            np.save(latent_path_inv,z_inv[j:j+1].detach().cpu().numpy())

            vertices,faces=vae.generate_mesh_2(embedding,0.01)

            j_scale=scale_pred[j].item()
            j_trans=trans_pred[j].detach().cpu().numpy()

            trans_scale_dict[f'{obj_index}_{j}']={'trans':j_trans.tolist(),'scale':j_scale,'var':var}
            vertices_trans=vertices*j_scale
            vertices_trans+=j_trans

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)

            # Save mesh to PLY file

            out_path=os.path.join(cur_out_dir,f'{obj_index}_{j}.obj')
            o3d.io.write_triangle_mesh(out_path, mesh)


    with open(os.path.join(cur_out_dir,'trans_scale.txt'), "w") as f:
        json.dump(trans_scale_dict, f)




def eval_fun(network,vae,test_dataset):
    results=[]
    preds=[]
    bs=1
    cur_key=None
    cur_dict={}
    cur_out_dir=None
    network.diffusion_model.model.cond_dropout=False

    result_dir='/data_sata/pack/result/mid_con_v2'
    test_subset_dict=json.load(open('/data_sata/pack/test_sim.json','r'))


    for data in tqdm(test_dataset):
        sym=[0]
        mask=data['roi_mask'].unsqueeze(0).cuda()
        depth=data['roi_depth'].unsqueeze(0).cuda()
        coord=data['roi_coord_2d'].unsqueeze(0).cuda()
        latent_code=data['latent_code'].cuda()
        scene_id=data['scene_id']
        im_id=data['img_id']
        obj_id=data['obj_id']
        obj_index=data['obj_index']
        rgb=data['roi_img']

        key=f'{scene_id}_{im_id}'
        # if not key in test_subset_dict.keys():
        #     continue
        if im_id <100:
            pass
        else:
            continue

        if key!=cur_key:
            if cur_key!=None:
                save_result(cur_out_dir,cur_dict,vae)
            cur_key=key
            cur_dict={}
            cur_out_dir=os.path.join(result_dir,f'{scene_id}_{im_id}')
            if not os.path.exists(cur_out_dir):
                os.makedirs(cur_out_dir)

        import matplotlib.pyplot as plt
        plt.imshow(rgb.numpy().transpose(1, 2, 0)[:,:,(2,1,0)])
        plt.savefig(os.path.join(cur_out_dir,f'{obj_index}.jpg'))


        scale=data['scale']


        camK=data['cam_K'].unsqueeze(0).cuda()
        PC = PC_sample(mask, depth, camK, coord)
        # points_defor = torch.randn(PC.shape).cuda()
        # PC = PC + torch.clip(points_defor * 0.001,-0.002,0.002)
        bs=PC.shape[0]



        gt_R=data['rotation'].cuda()
        gt_T=data['translation'].cuda()/1000

        gt_latent_code=torch.einsum('ij,qj->qi',gt_R,latent_code)
        center=PC.mean(dim=1,keepdim=True)
        # gt_T=gt_T-center.squeeze()

        pc_num=PC.shape[1]
        pc_center=PC-center


        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(pc_center[0].detach().cpu().numpy())


        eval_start_time = time.time()
        with torch.no_grad():
            final_fea,final_point,_= network.backbone1(pc_center)
            latent_pred,trans_pred,scale_pred,x_list,beta_list=network.diffusion_model.generate_from_cond(final_fea,final_point,20)

        # show_open3d(pc_center[0].cpu().numpy(),pc_center[0].cpu().numpy())
        # latent_pred=gt_latent_code[None,:,:]
        if x_list!=None:
            x_0_list=x_list[-1]
            x_0_mean=x_0_list.mean(0)
            x_0_var_list=[]
            for step in range(len(x_list)-1,len(x_list)):
                cur_var=torch.mean((x_list[step]-x_0_mean)**2)/beta_list[step]
                x_0_var_list.append(cur_var)
            x_0_var_list=torch.stack(x_0_var_list)
            x_0_var=torch.mean(x_0_var_list)
        z_so3,z_inv=vae.network.network_dict['encoder'].get_inv(latent_pred)

        # for j in range(1):
        #     embedding = {
        #         "z_so3": z_so3[j:j+1],
        #         "z_inv": z_inv[j:j+1],
        #     }
        #     vertices,faces=vae.generate_mesh_2(embedding,0.01)
        #     vertices_trans=vertices*(scale_pred[0].item())
        #     vertices_trans+=trans_pred[0].detach().cpu().numpy()
        #     mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()
        #     mesh_trans=o3d.geometry.TriangleMesh.create_coordinate_frame()
        #     mesh_trans.vertices=o3d.utility.Vector3dVector(vertices_trans)
        #     mesh.vertices=o3d.utility.Vector3dVector(vertices+0.5)
        #     mesh.triangles=o3d.utility.Vector3iVector(faces)
        #     mesh_trans.triangles=o3d.utility.Vector3iVector(faces)
        #     mesh.compute_vertex_normals()
        #     mesh_trans.compute_vertex_normals()
        #     frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        #     frame.scale(0.2, center=(0,0,0))
        #     o3d.visualization.draw_geometries([mesh,mesh_trans,frame,pcd],mesh_show_back_face=True)




        cur_dict[f'{obj_index}']={'trans':trans_pred+center[0,0,:],'scale':scale_pred,'var':x_0_var.item(),'z_so3':z_so3,'z_inv':z_inv}
    return




def eval_fun_nocs(network,vae,test_dataset):

    small_img_list=json.load(open('/data_sata/pack/result/small_img_list.txt'))
    small_img_keys=[]
    for img_path in small_img_list:
        scene_id=int(img_path.split('/')[-2].split('_')[1])
        im_id=int(img_path.split('/')[-1])
        small_img_keys.append((scene_id,im_id))
    result_dir='/data_sata/pack/result/mid_con_v1_nocs_mug_sample_10_mean'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    network.diffusion_model.model.cond_dropout=False
    for data in tqdm(test_dataset):
        data_dict,detection_dict,_=data

        scene_id=int(detection_dict['image_path'].split('/')[-2].split('_')[1])
        im_id=int(detection_dict['image_path'].split('/')[-1])
        if (scene_id,im_id) not in small_img_keys:
            continue
        cur_out_dir=os.path.join(result_dir,f'{scene_id}_{im_id}')
        import pickle
        with open(os.path.join(cur_out_dir,'detection_dict.pkl'), 'wb') as file:
            pickle.dump(detection_dict, file)


        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        valid_num=len(data_dict['roi_mask'])
        trans_scale_dict={}
        for inst_index in range(valid_num):
            mask=data_dict['roi_mask'][inst_index].unsqueeze(0).cuda()
            depth=data_dict['roi_depth'][inst_index].unsqueeze(0).cuda()
            coord=data_dict['roi_coord_2d'][inst_index].unsqueeze(0).cuda()
            gt_R=data_dict['gt_Rs'][inst_index]
            gt_T=data_dict['gt_Ts'][inst_index]
            gt_scale=data_dict['gt_scales'][inst_index]
            model_id=data_dict['model_names'][0]
            cloud=data_dict['model_points'][0]


            camK=data_dict['cam_K'][inst_index].unsqueeze(0).cuda()
            PC = PC_sample(mask, depth, camK, coord)
            # points_defor = torch.randn(PC.shape).cuda()
            # PC = PC + torch.clip(points_defor * 0.001,-0.002,0.002)
            bs=PC.shape[0]





            center=PC.mean(dim=1,keepdim=True)
            # gt_T=gt_T-center.squeeze()
            cam_cloud=torch.einsum('ij, pj->pi',gt_R,cloud)*torch.norm(gt_scale)+gt_T-center.squeeze().cpu()
            pc_num=PC.shape[1]
            pc_center=PC-center

            pcd=o3d.geometry.PointCloud()
            pcd.points=o3d.utility.Vector3dVector(pc_center[0].detach().cpu().numpy())


            eval_start_time = time.time()
            with torch.no_grad():
                final_fea,final_point,_= network.backbone1(pc_center,return_fuse=True)
                latent_pred,trans_pred,scale_pred,x_list,beta_list=network.diffusion_model.generate_from_cond(final_fea,final_point,5)

            # show_open3d(cam_cloud.cpu().numpy(),pc_center[0].cpu().numpy())
            # latent_pred=gt_latent_code[None,:,:]
            if x_list!=None:
                x_0_list=x_list[-1]
                x_0_mean=x_0_list.mean(0)
                x_0_var_list=[]
                for step in range(len(x_list)-1,len(x_list)):
                    cur_var=torch.mean((x_list[step]-x_0_mean)**2)/beta_list[step]
                    x_0_var_list.append(cur_var)
                x_0_var_list=torch.stack(x_0_var_list)
                x_0_var=torch.mean(x_0_var_list)

            z_so3,z_inv=vae.network.network_dict['encoder'].get_inv(latent_pred)

            z_so3=z_so3[0:1]
            z_inv=z_inv[0:1]
            for j in range(len(z_so3)):
                embedding = {
                    "z_so3": z_so3[j:j+1],
                    "z_inv": z_inv[j:j+1],
                }
                vertices,faces,_=vae.generate_mesh_2(embedding)
                vertices_trans=vertices*(scale_pred[j].item())
                vertices_trans+=trans_pred[j].detach().cpu().numpy()
                mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()
                mesh_trans=o3d.geometry.TriangleMesh.create_coordinate_frame()
                mesh_trans.vertices=o3d.utility.Vector3dVector(vertices_trans)
                mesh.vertices=o3d.utility.Vector3dVector(vertices+0.5)
                mesh.triangles=o3d.utility.Vector3iVector(faces)
                mesh_trans.triangles=o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                mesh_trans.compute_vertex_normals()
                frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
                frame.scale(0.2, center=(0,0,0))
                o3d.visualization.draw_geometries([mesh,mesh_trans,frame,pcd],mesh_show_back_face=True)
                #

                trans_pred=trans_pred+center[0,0]
                vertices,faces,_=vae.generate_mesh_2(embedding)
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)

                # Save mesh to PLY file

                latent_path_equi=os.path.join(cur_out_dir,f'{inst_index}_{j}.npy')
                latent_path_inv=os.path.join(cur_out_dir,f'{inst_index}_{j}_inv.npy')
                np.save(latent_path_equi,z_so3[j:j+1].detach().cpu().numpy())
                np.save(latent_path_inv,z_inv[j:j+1].detach().cpu().numpy())

                out_path=os.path.join(cur_out_dir,f'{inst_index}_{j}.obj')
                o3d.io.write_triangle_mesh(out_path, mesh)

                j_scale=scale_pred[j].item()
                j_trans=trans_pred[j].detach().cpu().numpy()
                trans_scale_dict[f'{inst_index}_{j}']={'pred_trans':j_trans.tolist(),
                                                       'pred_scale':j_scale,
                                                       'var':x_0_var.item(),
                                                       'gt_R':gt_R.reshape(-1).numpy().tolist(),
                                                       'gt_T':gt_T.reshape(-1).numpy().tolist(),
                                                       'gt_scales':gt_scale.reshape(-1).numpy().tolist(),
                                                       'model_id':model_id
                                                       }
        with open(os.path.join(cur_out_dir,'trans_scale.txt'), "w") as f:
            json.dump(trans_scale_dict, f)

    return







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

    # 保存或显示结果
    cv2.imshow('Rendered Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




def my_eval(network,vae,test_dataset):
    network.eval()
    vae.eval()
    # eval_fun(network,vae,test_dataset)
    eval_fun_nocs(network,vae,test_dataset)


