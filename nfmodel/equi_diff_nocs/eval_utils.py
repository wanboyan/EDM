import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import absl.flags as flags
FLAGS = flags.FLAGS
import trimesh
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


def cal_R(source,target):

    P_centered = source
    Q_centered = target

    # Step 2: 计算相关矩阵
    H = np.dot(P_centered.T, Q_centered)

    # Step 3: 奇异值分解
    U, S, Vt = np.linalg.svd(H)

    # Step 4: 计算旋转矩阵
    R = np.dot(Vt.T, U.T)

    # 确保得到一个合适的旋转矩阵（处理反射情况）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    return R








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

            vertices,faces,_=vae.generate_mesh_2(embedding,0.01)

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















def save_result_occ(cur_out_dir,img_dict,vae,can_r_vector):
    trans_scale_dict={}
    var_max=-1
    obj_max=-1
    for obj_index in img_dict.keys():
        var=img_dict[obj_index]['var']
        if var>var_max:
            var_max=var
            obj_max=obj_index
    for obj_index in img_dict.keys():
        latent_pred=img_dict[obj_index]['latent_pred']
        r_pred=img_dict[obj_index]['r_pred']
        trans_pred=img_dict[obj_index]['trans']
        scale_pred=img_dict[obj_index]['scale']
        var=img_dict[obj_index]['var']


        j=0
        embedding=latent_pred[j:j+1]
        vertices,faces=vae.get_mesh_from_latent(embedding)

        j_scale=scale_pred[j].item()
        j_trans=trans_pred[j].detach().cpu().numpy()
        j_r_vector=r_pred[j]
        j_R=cal_R(can_r_vector.numpy(),j_r_vector.detach().cpu().numpy())


        trans_scale_dict[f'{obj_index}_{j}']={'trans':j_trans.tolist(),'scale':j_scale,'var':var,
                                              'rotation':j_R.reshape(-1).tolist()}
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




def eval_fun_occ(network_dict,vae,test_dataset,result_dir):

    cur_key=None
    cur_dict={}
    cur_out_dir=None

    can_r_vector=torch.load(FLAGS.occ_r_path)
    for data in tqdm(test_dataset):
        sym=[0]
        # mask=data['roi_mask'].unsqueeze(0).cuda()
        mask=data['roi_mask_deform'].unsqueeze(0).cuda()
        depth=data['roi_depth'].unsqueeze(0).cuda()
        cls_vector=data['cls_vector'].unsqueeze(0).cuda()
        coord=data['roi_coord_2d'].unsqueeze(0).cuda()
        latent_code=data['latent_code'].cuda()
        scene_id=data['scene_id']
        im_id=data['img_id']
        obj_id=data['obj_id']
        obj_index=data['obj_index']
        rgb=data['roi_img']
        cat_name=data['cat_name']

        if cat_name in ['bottle','can','bowl']:
            network_cat='container'
        else:

            network_cat=cat_name
        network=network_dict[network_cat]



        key=f'{scene_id}_{im_id}'

        # if im_id <100:
        #     pass
        # else:
        #     continue

        if key!=cur_key:
            if cur_key!=None:
                save_result_occ(cur_out_dir,cur_dict,vae,can_r_vector)
            cur_key=key
            cur_dict={}
            cur_out_dir=os.path.join(result_dir,f'{scene_id}_{im_id}')
            if not os.path.exists(cur_out_dir):
                os.makedirs(cur_out_dir)

        # import matplotlib.pyplot as plt
        # plt.imshow(rgb.numpy().transpose(1, 2, 0)[:,:,(2,1,0)])
        # plt.savefig(os.path.join(cur_out_dir,f'{obj_index}.jpg'))


        scale=data['scale']


        camK=data['cam_K'].unsqueeze(0).cuda()
        PC = PC_sample(mask, depth, camK, coord)
        points_defor = torch.randn(PC.shape).cuda()
        PC = PC + points_defor * 0.005
        bs=PC.shape[0]



        gt_R=data['rotation'].cuda()
        gt_T=data['translation'].cuda()/1000

        center=PC.mean(dim=1,keepdim=True)
        # gt_T=gt_T-center.squeeze()

        pc_num=PC.shape[1]
        pc_center=PC-center


        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(pc_center[0].detach().cpu().numpy())


        eval_start_time = time.time()
        with torch.no_grad():
            final_fea,final_point,_= network.backbone1(pc_center,return_fuse=True)

            latent_pred,r_pred,trans_pred,scale_pred,x_list,beta_list=\
                network.diffusion_model.generate_from_cond(final_fea,final_point,cls_vector,1)

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


        # for j in range(1):
        #
        #
        #     r_vector=r_pred[0]
        #     pred_R=cal_R(can_r_vector.numpy(),r_vector.detach().cpu().numpy())
        #
        #     embedding = latent_pred[j:j+1]
        #     vertices,faces=vae.get_mesh_from_latent(embedding)
        #
        #     vertices_trans=np.einsum('ij,pj->pi',pred_R,vertices)
        #     vertices_trans=vertices_trans*(scale_pred[j].item())
        #     # nocs_scale=torch.norm(gt_scale)
        #     # vertices_trans=vertices_trans*(nocs_scale.item())
        #     vertices_trans+=trans_pred[j].detach().cpu().numpy()
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




        cur_dict[f'{obj_index}']={'trans':trans_pred+center[0,0,:],'scale':scale_pred,'var':x_0_var.item(),'latent_pred':latent_pred,
                                  'r_pred':r_pred}
    return



def eval_fun_occ_cam(network_dict,vae,test_dataset,result_dir):

    cur_key=None
    cur_dict={}
    cur_out_dir=None

    can_r_vector=torch.load(FLAGS.occ_r_path)
    for data in tqdm(test_dataset):
        sym=[0]
        mask=data['roi_mask'].unsqueeze(0).cuda()
        # mask=data['roi_mask_deform'].unsqueeze(0).cuda()
        depth=data['roi_depth'].unsqueeze(0).cuda()
        cls_vector=data['cls_vector'].unsqueeze(0).cuda()
        coord=data['roi_coord_2d'].unsqueeze(0).cuda()

        im_id=data['img_id']
        obj_index=data['obj_index']

        cat_name=data['cat_name']

        if cat_name in ['bottle','can','bowl']:
            network_cat='container'
        else:

            network_cat=cat_name
        network=network_dict[network_cat]



        key=im_id



        if key!=cur_key:
            if cur_key!=None:
                save_result_occ(cur_out_dir,cur_dict,vae,can_r_vector)
            cur_key=key
            cur_dict={}
            cur_out_dir=os.path.join(result_dir,f'{im_id}')
            if not os.path.exists(cur_out_dir):
                os.makedirs(cur_out_dir)

        # import matplotlib.pyplot as plt
        # plt.imshow(rgb.numpy().transpose(1, 2, 0)[:,:,(2,1,0)])
        # plt.savefig(os.path.join(cur_out_dir,f'{obj_index}.jpg'))





        camK=data['cam_K'].unsqueeze(0).cuda()
        PC = PC_sample(mask, depth, camK, coord)
        # points_defor = torch.randn(PC.shape).cuda()
        # PC = PC + points_defor * 0.005
        bs=PC.shape[0]





        center=PC.mean(dim=1,keepdim=True)
        # gt_T=gt_T-center.squeeze()

        pc_num=PC.shape[1]
        pc_center=PC-center


        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(pc_center[0].detach().cpu().numpy())


        eval_start_time = time.time()
        with torch.no_grad():
            final_fea,final_point,_= network.backbone1(pc_center,return_fuse=True)

            latent_pred,r_pred,trans_pred,scale_pred,x_list,beta_list= \
                network.diffusion_model.generate_from_cond(final_fea,final_point,cls_vector,5)

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


        # for j in range(1):
        #
        #
        #     r_vector=r_pred[0]
        #     pred_R=cal_R(can_r_vector.numpy(),r_vector.detach().cpu().numpy())
        #
        #     embedding = latent_pred[j:j+1]
        #     vertices,faces=vae.get_mesh_from_latent(embedding)
        #
        #     vertices_trans=np.einsum('ij,pj->pi',pred_R,vertices)
        #     vertices_trans=vertices_trans*(scale_pred[j].item())
        #     # nocs_scale=torch.norm(gt_scale)
        #     # vertices_trans=vertices_trans*(nocs_scale.item())
        #     vertices_trans+=trans_pred[j].detach().cpu().numpy()
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




        cur_dict[f'{obj_index}']={'trans':trans_pred+center[0,0,:],'scale':scale_pred,'var':x_0_var.item(),'latent_pred':latent_pred,
                                  'r_pred':r_pred}
    return










def eval_fun_nocs(network_dict,vae,test_dataset,result_dir):
    id2cat_name = {1: 'bottle', 2: 'bowl',3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
    small_img_list=json.load(open('/data_sata/pack/result/small_img_list_500.txt'))
    small_img_keys=[]
    for img_path in small_img_list:
        scene_id=int(img_path.split('/')[-2].split('_')[1])
        im_id=int(img_path.split('/')[-1])
        small_img_keys.append((scene_id,im_id))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for data in tqdm(test_dataset):
        data_dict,detection_dict,_=data

        scene_id=int(detection_dict['image_path'].split('/')[-2].split('_')[1])
        im_id=int(detection_dict['image_path'].split('/')[-1])
        if (scene_id,im_id) not in small_img_keys:
            continue
        cur_out_dir=os.path.join(result_dir,f'{scene_id}_{im_id}')
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        import pickle
        with open(os.path.join(cur_out_dir,'detection_dict.pkl'), 'wb') as file:
            pickle.dump(detection_dict, file)



        valid_num=len(data_dict['roi_mask'])
        trans_scale_dict={}
        for inst_index in range(valid_num):
            mask=data_dict['roi_mask'][inst_index].unsqueeze(0).cuda()
            depth=data_dict['roi_depth'][inst_index].unsqueeze(0).cuda()
            cls_vector=data_dict['cls_vectors'][inst_index].unsqueeze(0).cuda()
            coord=data_dict['roi_coord_2d'][inst_index].unsqueeze(0).cuda()
            gt_R=data_dict['gt_Rs'][inst_index]
            gt_T=data_dict['gt_Ts'][inst_index]
            cat_id=int(data_dict['cat_id'][inst_index].item())
            cat_name=id2cat_name[cat_id]
            if cat_name in ['bottle','can','bowl']:
                network_cat='container'
            else:
                network_cat=cat_name
            network=network_dict[network_cat]
            gt_scale=data_dict['gt_scales'][inst_index]
            model_id=data_dict['model_names'][inst_index]
            cloud=data_dict['model_points'][inst_index]


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
                latent_pred,trans_pred,scale_pred,x_list,beta_list=network.diffusion_model.generate_from_cond(final_fea,final_point,cls_vector,100)

            eval_end_time = time.time()
            print('diffuison_time',eval_end_time-eval_start_time)
            # show_open3d(cam_cloud.cpu().numpy(),pc_center[0].cpu().numpy())
            # latent_pred=gt_latent_code[None,:,:]
            trans_pred=(gt_T-center.squeeze().cpu())[None,:].repeat(5,1)
            scale_pred=torch.norm(gt_scale).unsqueeze(0).repeat(5,1)
            if x_list!=None:
                x_0_list=x_list[-1]
                x_0_mean=x_0_list.mean(0)
                x_0_var_list=[]
                for step in range(len(x_list)-1,len(x_list)):
                    cur_var=torch.mean((x_list[step]-x_0_mean)**2)/beta_list[step]
                    x_0_var_list.append(cur_var)
                x_0_var_list=torch.stack(x_0_var_list)
                x_0_var=torch.mean(x_0_var_list).item()
            else:
                x_0_var=0

            z_so3,z_inv=vae.network.network_dict['encoder'].get_inv(latent_pred)

            z_so3=z_so3[0:1]
            z_inv=z_inv[0:1]
            for j in range(len(z_so3)):
                embedding = {
                    "z_so3": z_so3[j:j+1],
                    "z_inv": z_inv[j:j+1],
                }
                # vertices,faces,_=vae.generate_mesh_2(embedding,0.02)
                # vertices_trans=vertices*(scale_pred[j].item())
                # vertices_trans+=trans_pred[j].detach().cpu().numpy()
                # mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()
                # mesh_trans=o3d.geometry.TriangleMesh.create_coordinate_frame()
                # mesh_trans.vertices=o3d.utility.Vector3dVector(vertices_trans)
                # mesh.vertices=o3d.utility.Vector3dVector(vertices+0.5)
                # mesh.triangles=o3d.utility.Vector3iVector(faces)
                # mesh_trans.triangles=o3d.utility.Vector3iVector(faces)
                # mesh.compute_vertex_normals()
                # mesh_trans.compute_vertex_normals()
                # frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
                # frame.scale(0.2, center=(0,0,0))
                # o3d.visualization.draw_geometries([mesh,mesh_trans,frame,pcd],mesh_show_back_face=True)


                trans_pred=trans_pred.cpu()+center[0,0].cpu()
                thred=0.01
                for i in range(20):
                    try:
                        eval_start_time = time.time()
                        vertices,faces,_=vae.generate_mesh_2(embedding,thred)
                        eval_end_time = time.time()
                        print('recon_time',eval_end_time-eval_start_time)
                        break
                    except:
                        thred+=0.01
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
                                                       'var':x_0_var,
                                                       'cat_name':cat_name,
                                                       'gt_R':gt_R.reshape(-1).numpy().tolist(),
                                                       'gt_T':gt_T.reshape(-1).numpy().tolist(),
                                                       'gt_scales':gt_scale.reshape(-1).numpy().tolist(),
                                                       'model_id':model_id
                                                       }
        with open(os.path.join(cur_out_dir,'trans_scale.txt'), "w") as f:
            json.dump(trans_scale_dict, f)

    return













def visualize_feature_tsne(x_src, x_tgt, savepath):
    '''
    src, tgt: torch tensor BxAxc_out
    '''
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.cm as cm
    # def get_cmap(n, name='hsv'):
    #     '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    #     RGB color; the keyword argument name must be a standard mpl colormap name.'''
    #     return plt.cm.get_cmap(name, n)
    # k_cm=get_cmap(4)
    cm=plt.cm.get_cmap('GnBu')


    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np



    bdim = x_src.size(0)
    scaler = MinMaxScaler()
    x_src = scaler.fit_transform(x_src.cpu().numpy())
    x_tgt = scaler.fit_transform(x_tgt.cpu().numpy())

    x_src = x_src.reshape(bdim, 16,16)
    x_tgt = x_tgt.reshape(bdim, 16,16)

    data=np.concatenate([x_src,x_tgt],axis=0)

    fig, axes = plt.subplots(nrows=2, ncols=12, figsize=(24, 6))

    #遍历每一个子图和相应的数据
    for ax, grid in zip(axes.flat, data):

        # cax = ax.matshow(grid, cmap=cm,vmin=0.0,vmax=2)
        cax = ax.matshow(grid, cmap=cm)
        ax.axis('off')  # 关闭每个子图的坐标轴

    #添加一个统一的颜色条
    # data=np.random.rand(10,10)
    # fig,ax=plt.subplots()
    # cax=ax.imshow(data,cmap='viridis',vmin=0,vmax=2)
    # fig.colorbar(cax, ax=ax)

    # 调整布局
    plt.tight_layout()
    plt.savefig(savepath)






def eval_fun_nocs_occ(network_dict,vae,test_dataset,result_dir):
    id2cat_name = {1: 'bottle', 2: 'bowl',3: 'camera', 4: 'can', 5: 'laptop', 6: 'mug'}
    small_img_list=json.load(open('/data_sata/pack/result/small_img_list.txt'))
    small_img_keys=[]

    can_r_vector=torch.load(FLAGS.occ_r_path)


    for img_path in small_img_list:
        scene_id=int(img_path.split('/')[-2].split('_')[1])
        im_id=int(img_path.split('/')[-1])
        small_img_keys.append((scene_id,im_id))

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for data in tqdm(test_dataset):
        data_dict,detection_dict,_=data

        scene_id=int(detection_dict['image_path'].split('/')[-2].split('_')[1])
        im_id=int(detection_dict['image_path'].split('/')[-1])
        if (scene_id,im_id) not in small_img_keys:
            continue
        cur_out_dir=os.path.join(result_dir,f'{scene_id}_{im_id}')
        if not os.path.exists(cur_out_dir):
            os.makedirs(cur_out_dir)
        import pickle
        with open(os.path.join(cur_out_dir,'detection_dict.pkl'), 'wb') as file:
            pickle.dump(detection_dict, file)



        valid_num=len(data_dict['roi_mask'])
        trans_scale_dict={}
        for inst_index in range(valid_num):
            mask=data_dict['roi_mask'][inst_index].unsqueeze(0).cuda()
            depth=data_dict['roi_depth'][inst_index].unsqueeze(0).cuda()
            cls_vector=data_dict['cls_vectors'][inst_index].unsqueeze(0).cuda()
            coord=data_dict['roi_coord_2d'][inst_index].unsqueeze(0).cuda()
            gt_R=data_dict['gt_Rs'][inst_index]
            gt_T=data_dict['gt_Ts'][inst_index]
            gt_scale=data_dict['gt_scales'][inst_index]
            model_id=data_dict['model_names'][0]
            cloud=data_dict['model_points'][0]

            cat_id=int(data_dict['cat_id'][inst_index].item())
            cat_name=id2cat_name[cat_id]
            if cat_name in ['bottle','can','bowl']:
                network_cat='container'
            else:
                network_cat=cat_name
            try:
                network=network_dict[network_cat]
            except:
                continue




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
                latent_pred,r_pred,trans_pred,scale_pred,x_list,beta_list=network.diffusion_model.generate_from_cond(final_fea,final_point,cls_vector,5)

            # show_open3d(cam_cloud.cpu().numpy(),pc_center[0].cpu().numpy())

            x_0_var=0

            latent_pred=latent_pred[0:1]
            for j in range(len(latent_pred)):

                r_vector=r_pred[j]
                pred_R=cal_R(can_r_vector.numpy(),r_vector.detach().cpu().numpy())


                embedding = latent_pred[j].unsqueeze(0)
                vertices,faces=vae.get_mesh_from_latent(embedding)

                # vertices_trans=np.einsum('ij,pj->pi',pred_R,vertices)
                # vertices_trans=vertices_trans*(scale_pred[j].item())
                # # nocs_scale=torch.norm(gt_scale)
                # # vertices_trans=vertices_trans*(nocs_scale.item())
                # vertices_trans+=trans_pred[j].detach().cpu().numpy()
                # mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()
                # mesh_trans=o3d.geometry.TriangleMesh.create_coordinate_frame()
                # mesh_trans.vertices=o3d.utility.Vector3dVector(vertices_trans)
                # mesh.vertices=o3d.utility.Vector3dVector(vertices+0.5)
                # mesh.triangles=o3d.utility.Vector3iVector(faces)
                # mesh_trans.triangles=o3d.utility.Vector3iVector(faces)
                # mesh.compute_vertex_normals()
                # mesh_trans.compute_vertex_normals()
                # frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
                # frame.scale(0.2, center=(0,0,0))
                # o3d.visualization.draw_geometries([mesh,mesh_trans,frame,pcd],mesh_show_back_face=True)

                trans_pred=trans_pred+center[0,0]
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)

                out_path=os.path.join(cur_out_dir,f'{inst_index}_{j}.obj')
                o3d.io.write_triangle_mesh(out_path, mesh)
                j_R=pred_R.reshape(-1).tolist()
                j_scale=scale_pred[j].item()
                j_trans=trans_pred[j].detach().cpu().numpy()
                trans_scale_dict[f'{inst_index}_{j}']={'pred_trans':j_trans.tolist(),
                                                       'pred_R':j_R,
                                                       'pred_scale':j_scale,
                                                       'var':x_0_var,
                                                       'cat_name':cat_name,
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




def my_eval(network_dict,vae,test_dataset,result_dir):

    eval_fun_nocs(network_dict,vae,test_dataset,result_dir)
    # eval_fun_nocs_fake(network_dict,vae,test_dataset,result_dir)



def my_eval_occ(network_dict,vae,test_dataset,result_dir):
    vae.eval()
    eval_fun_nocs_occ(network_dict,vae,test_dataset,result_dir)


