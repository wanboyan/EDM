import numpy as np
import torch
import os
import json
from dm_control import mujoco
import mediapy as media
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pytorch3d.transforms import matrix_to_quaternion
from dm_control.mujoco.wrapper.mjbindings import enums
import trimesh
import pyrender
from tqdm import tqdm
import itertools
import torch.multiprocessing as mp
from collections import defaultdict
import trimesh
from metric_utils import *
from lmo_utils import sample_points_from_mesh
from nfmodel.uti_tool import show_open3d
from vsd import *
from nfmodel.uti_tool import modelnet_r_diff
data_dir='/data_sata/pack/test'
result_dir='/data_sata/pack/result/mid_con_v1'
mesh_dir='/data_nvme/pack/container/sim'
cloud_dir='/data_sata/pack/cloud'
ply_dir='/data_sata/pack/ply/all'

latent_dir='/data_sata/pack/latent/latent_49499'

start = 0
end = 1
# test_subset_dict=json.load(open('/data_sata/pack/test_sim.json'))

subfolders=[entry for entry in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir,entry))]


ds_dir=os.path.join(data_dir, 'ycbv/train_pbr')

id2latent={}
id2latent_inv={}
for entry in os.listdir(latent_dir):
    if not entry.endswith('inv.npy'):
        id=entry.split('.')[0]
        latent=np.load(os.path.join(latent_dir,entry))
        id2latent[id]=latent

for entry in os.listdir(latent_dir):
    if entry.endswith('inv.npy'):
        id=entry.split('.')[0]
        latent_inv=np.load(os.path.join(latent_dir,entry))
        id2latent_inv[id]=latent_inv


id2latent={key:id2latent[key] for key in sorted(id2latent)}
id2latent_inv={key:id2latent_inv[key] for key in sorted(id2latent_inv)}

latent_inv_values=np.array(list(id2latent_inv.values()))
latent_inv_keys=list(id2latent_inv.keys())
latent_inv_norms=np.linalg.norm(latent_inv_values,axis=-1)

out_dict={}
add_list=[]
vsd_list=[]
r_diff_list=[]
t_diff_list=[]
for folder_id in range(start, end):
    depth_parent_path = os.path.join(ds_dir, f'{int(folder_id):06d}', 'depth')
    with open(os.path.join(ds_dir, f'{int(folder_id):06d}/scene_gt_info.json')) as f:
        scene_gt_info_dict = json.load(f)

    with open(os.path.join(ds_dir, f'{int(folder_id):06d}/scene_camera.json')) as f:
        scene_camera_dict = json.load(f)

    with open(os.path.join(ds_dir, f'{int(folder_id):06d}/scene_gt.json')) as f:
        scene_gt_dict = json.load(f)

    cam_K=np.array(scene_camera_dict['0']['cam_K'])
    depth_scale=scene_camera_dict['0']['depth_scale']



    # for data_idx in range(len(self.scene_gt_info_dict)):
    for data_id in tqdm(scene_gt_info_dict.keys()):
        depth_path = os.path.join(depth_parent_path, f'{int(data_id):06d}.png')
        depth = cv2.imread(depth_path, -1)*depth_scale
        len_item = len(scene_gt_info_dict[str(data_id)])
        obj2gt={}
        key=f'{folder_id}_{data_id}'
        #
        # if int(data_id) >5:
        #     break

        if key not in subfolders:
            continue
        for obj_idx in range(len_item):
            cur_dict={}
            cur_dict['gt_R']=np.array(scene_gt_dict[str(data_id)][obj_idx]['cam_R_m2c']).reshape(3,3)
            cur_dict['gt_t']=np.array(scene_gt_dict[str(data_id)][obj_idx]['cam_t_m2c'])/1000.0
            cur_dict['gt_scale']=scene_gt_dict[str(data_id)][obj_idx]['scale']
            cur_dict['gt_id']=scene_gt_dict[str(data_id)][obj_idx]['obj_id']
            cur_dict['gt_cloud_path']=os.path.join(cloud_dir,f"{cur_dict['gt_id']}.npy")
            cur_dict['gt_ply_path']=os.path.join(ply_dir,f"{cur_dict['gt_id']}.ply")
            cur_dict['gt_cloud']=np.load(cur_dict['gt_cloud_path'])
            gt_mesh = trimesh.load_mesh(cur_dict['gt_ply_path'])
            gt_render_mesh = pyrender.Mesh.from_trimesh(gt_mesh)
            cur_dict['gt_render_mesh']=gt_render_mesh
            obj2gt[obj_idx]=cur_dict

        obj2sets=defaultdict(lambda :[])

        subfolder=os.path.join(result_dir,key)
        for entry in os.listdir(subfolder):
            if entry.endswith('.obj'):
                obj_index,_=entry.split('.')[0].split('_')
                obj2sets[obj_index].append(entry.split('.')[0])
            if entry.endswith('.txt'):
                trans_scale_dict=json.load(open(os.path.join(subfolder,entry), "r"))

        sets=list(obj2sets.values())
        high_light_index=-1
        for j,set in enumerate(sets):
            if len(set)>1:
                high_light_index=j
        combineations=list(itertools.product(*sets))
        combine=combineations[0]
        for obj_key in combine:
            obj_index=int(obj_key.split('_')[0])
            obj_id=obj2gt[obj_index]['gt_id']
            pred_latent=np.load(os.path.join(subfolder,f"{obj_key}.npy"))
            pred_latent_inv=np.load(os.path.join(subfolder,f"{obj_key}_inv.npy"))
            pred_latent_inv_norm=np.linalg.norm(pred_latent_inv,axis=-1)


            match_scores=np.einsum('n d, d->n',latent_inv_values,pred_latent_inv[0])/latent_inv_norms/pred_latent_inv_norm

            match_index=np.argmax(match_scores)
            match_id=latent_inv_keys[match_index]
            match_id=match_id.split('_')[0]

            # gt_latent=np.load(os.path.join(latent_dir,f'{obj_id}.npy'))
            # gt_latent_inv=np.load(os.path.join(latent_dir,f'{obj_id}_inv.npy'))

            match_latent=id2latent[match_id]

            pred_R=cal_R(match_latent,pred_latent[0])




            mesh_path=os.path.join(subfolder,f"{obj_key}.obj")
            mesh = trimesh.load(mesh_path, validate=True)
            vertice=np.array(mesh.vertices)
            faces=np.array(mesh.faces)
            pred_cloud=sample_points_from_mesh(vertice,faces, 2048, fps=False, ratio=3)


            pred_render_mesh=pyrender.Mesh.from_trimesh(mesh)
            gt_render_mesh=obj2gt[obj_index]['gt_render_mesh']


            pred_scale=trans_scale_dict[obj_key]['scale']
            pred_trans=trans_scale_dict[obj_key]['trans']


            pred_scale_matrix=np.diag([pred_scale,pred_scale,pred_scale,1])
            pred_pose= np.eye(4)
            pred_pose[:3,3]=pred_trans
            pred_pose=pred_pose @ pred_scale_matrix



            pred_cloud=pred_cloud*pred_scale+pred_trans

            gt_cloud=obj2gt[obj_index]['gt_cloud']
            gt_R=obj2gt[obj_index]['gt_R']
            gt_t=obj2gt[obj_index]['gt_t']
            gt_scale=obj2gt[obj_index]['gt_scale']


            gt_scale_matrix=np.diag([gt_scale,gt_scale,gt_scale,1])
            gt_pose= np.eye(4)
            gt_pose[:3,3]=gt_t
            gt_pose[:3,:3]=gt_R
            gt_pose=gt_pose @ gt_scale_matrix

            vsd_error=my_vsd(pred_render_mesh,gt_render_mesh,pred_pose,gt_pose,cam_K,depth)[0]
            vsd_list.append(vsd_error)
            gt_cloud_cam=np.einsum('ij,pj->pi',gt_R,gt_cloud*gt_scale)+gt_t

            add_error=my_adi_err(pred_cloud,gt_cloud_cam)
            add_list.append(add_error)

            r_diff=modelnet_r_diff(torch.from_numpy(gt_R).float(),torch.from_numpy(pred_R).float(),0).item()
            r_diff_list.append(r_diff)



add_list=np.array(add_list)
vsd_list=np.array(vsd_list)
vsd_recall=(vsd_list<=0.3).sum()/len(vsd_list)
auc=compute_auc(add_list,0.05)

r_diff_list=np.array(r_diff_list)
r_recall=(r_diff_list<5).sum()/len(r_diff_list)
print('auc',auc)
print('vsd_recall',vsd_recall)
print('r_recall',r_recall)












