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
import _pickle as cPickle
from tqdm import tqdm
import itertools
import torch.multiprocessing as mp
from collections import defaultdict
import trimesh
from metric_utils import *
from tools.eval_utils import load_depth
from lmo_utils import sample_points_from_mesh
from nfmodel.uti_tool import show_open3d
from vsd import *
from nfmodel.uti_tool import modelnet_r_diff
from datasets.nocs.data_utils import get_nocs_models
from tools.pyTorchChamferDistance.chamfer_distance import ChamferDistance
import random

result_dir='/data_sata/pack/result/shapo_result_0.5'

dataset_dir='/data_nvme/nocs_data'
detection_dir='/data_nvme/nocs_data/segmentation_results'


ply_dir='/data_sata/pack/ply/all'
cloud_dir='/data_sata/pack/cloud'
mesh_dir='/data_nvme/nocs_data/obj_models/real_test'

depth_path_temp='Real/test/scene_{}/{:04d}_depth.png'
detection_dir = '/data_sata/pack/result/stage2_nocs_exp'


with open(os.path.join(dataset_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
    mug_meta = cPickle.load(f)

subfolders=[entry for entry in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir,entry))]

id2cat_name,id2cloud=get_nocs_models(dataset_dir)





cd_loss=ChamferDistance()

id2mesh={}

out_dict={}
add_list=[]
vsd_list=[]
r_diff_list=[]
t_diff_list=[]
cd_list=[]
pred_results=[]
cam_K=np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]], dtype=np.float)

for subfolder in tqdm(subfolders):
    scene_id,im_id=subfolder.split('_')
    scene_id=int(scene_id)
    im_id=int(im_id)
    depth_path=os.path.join(dataset_dir,depth_path_temp.format(scene_id,im_id))
    try:
        depth = load_depth(depth_path)
    except:
        print('no depth',depth_path)

    obj2sets=defaultdict(lambda :[])
    subfolder=os.path.join(result_dir,subfolder)

    cur_detection_dir=os.path.join(detection_dir,f'{scene_id}_{im_id}')
    try:
        detection_dict=cPickle.load(open(os.path.join(cur_detection_dir,'detection_dict.pkl'),'rb'))
    except:
        print(os.path.join(cur_detection_dir,'detection_dict.pkl'),'not exist')
        continue


    pred_dict=cPickle.load(open(os.path.join(subfolder,'pred_dict.pkl'),'rb'))



    combine=[]
    for entry in os.listdir(subfolder):
        if entry.endswith('.obj'):
            obj_index=entry.split('.')[0]
            combine.append(obj_index)

    pred_RT=np.zeros((len(combine),4,4))
    pred_s=np.zeros((len(combine),3))

    match=match_box(pred_dict['pred_bboxes'],detection_dict['gt_bboxes'])
    for obj_index in combine:
        obj_index=int(obj_index)
        match_index=match[obj_index]

        pred_RTs=pred_dict['pred_RTs'][obj_index]

        pred_factor=np.cbrt(np.linalg.det(pred_RTs[:3, :3]))
        pred_R=pred_RTs[:3, :3]/pred_factor
        pred_t=pred_RTs[:3,3]


        gt_RTs=detection_dict['gt_RTs'][match_index]
        gt_scale=detection_dict['gt_scales'][match_index]
        gt_id=detection_dict['model'][match_index]
        gt_cloud_norm=id2cloud[gt_id]

        factor=np.cbrt(np.linalg.det(gt_RTs[:3, :3]))
        gt_R=gt_RTs[:3, :3]/factor
        gt_t=gt_RTs[:3,3]
        gt_scale=gt_scale*factor





        mesh_path=os.path.join(subfolder,f"{obj_index}.obj")
        mesh = trimesh.load(mesh_path)
        vertice=np.array(mesh.vertices)
        faces=np.array(mesh.faces)
        try:
            pred_cloud=sample_points_from_mesh(vertice,faces, 20000, fps=False, ratio=3)
        except:
            print('error mesh')
            continue

        lx = max(pred_cloud[:, 0]) - min(pred_cloud[:, 0])
        ly = max(pred_cloud[:, 1]) - min(pred_cloud[:, 1])
        lz = max(pred_cloud[:, 2]) - min(pred_cloud[:, 2])

        match_bbox=np.array([lx,ly,lz])

        pred_norm=np.linalg.norm(match_bbox)
        pred_cloud_norm=pred_cloud/pred_norm


        pred_render_mesh=pyrender.Mesh.from_trimesh(mesh)


        if gt_id not in id2mesh:

            gt_mesh_path=os.path.join(mesh_dir,gt_id+'.obj')
            gt_mesh = trimesh.load_mesh(gt_mesh_path)
            vertice=np.array(gt_mesh.vertices)
            faces=np.array(gt_mesh.faces)
            gt_mesh_cloud=sample_points_from_mesh(vertice,faces, 20000, fps=False, ratio=3)
            gt_render_mesh = pyrender.Mesh.from_trimesh(gt_mesh)
            id2mesh[gt_id]=gt_render_mesh
        else:
            print('find mesh')
            gt_render_mesh=id2mesh[gt_id]



        # show_open3d(match_cloud,no_re_gt_cloud)
        dist1, dist2=cd_loss(torch.from_numpy(gt_cloud_norm)[None,:,:].float(),torch.from_numpy(pred_cloud_norm)[None,:,:].float())
        cd = (torch.mean(dist1)) + (torch.mean(dist2)).item()
        cd_list.append(cd)






        pred_pose=pred_RTs



        pred_cloud_cam=np.einsum('ij,pj->pi',pred_R,pred_cloud_norm)*pred_factor+pred_t




        gt_mesh_pose= np.eye(4)
        gt_mesh_pose[:3,3]=gt_t
        gt_mesh_pose[:3,:3]=gt_R





        vsd_error=my_vsd(pred_render_mesh,gt_render_mesh,pred_pose,gt_mesh_pose,cam_K,depth)[0]
        # vsd_error=0
        vsd_list.append(vsd_error)
        gt_cloud_cam=np.einsum('ij,pj->pi',gt_R,gt_cloud_norm)*np.linalg.norm(gt_scale)+gt_t

        gt_mesh_cloud_cam=np.einsum('ij,pj->pi',gt_R,gt_mesh_cloud)+gt_t




        # show_open3d(pred_cloud_cam,gt_cloud_cam)
        # show_open3d(pred_match_cloud,gt_cloud_cam)
        # show_open3d(gt_mesh_cloud_cam,gt_cloud_cam)


        add_error=my_adi_err(pred_cloud_cam,gt_cloud_cam)
        add_list.append(add_error)

        r_diff=modelnet_r_diff(torch.from_numpy(gt_R).float(),torch.from_numpy(pred_R).float(),0).item()
        r_diff_list.append(r_diff)

        t_diff=np.linalg.norm(pred_t - gt_t)
        t_diff_list.append(t_diff)





cd_list=np.array(cd_list)
add_list=np.array(add_list)
vsd_list=np.array(vsd_list)
vsd_recall=(vsd_list<=0.3).sum()/len(vsd_list)
auc=compute_auc(add_list,0.05)

r_diff_list=np.array(r_diff_list)
t_diff_list=np.array(t_diff_list)

r_5degree=r_diff_list<=5
t_2cm=t_diff_list<=0.02
r_recall=(r_5degree).sum()/len(r_diff_list)

rt_5d2c_recall=(r_5degree*t_2cm).sum()/len(r_diff_list)
print('auc',auc)
print('vsd_recall',vsd_recall)
print('r_recall',r_recall)
print('rt_5d2c_recall',rt_5d2c_recall)
print('chamfer',cd_list.mean())












