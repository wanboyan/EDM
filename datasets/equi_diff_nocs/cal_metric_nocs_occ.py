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

result_dir='/data_sata/pack/result/stage2_occ_nocs'
pred_result_save_path='/data_sata/pack/result/stage2_occ_nocs.pkl'
dataset_dir='/data_nvme/nocs_data'
detection_dir='/data_nvme/nocs_data/segmentation_results'


ply_dir='/data_sata/pack/ply/all'
cloud_dir='/data_sata/pack/cloud'
mesh_dir='/data_nvme/nocs_data/obj_models/real_test'

depth_path_temp='Real/test/scene_{}/{:04d}_depth.png'
detection_path_temp='REAL275/results_test_scene_{}_{:04d}.pkl'


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

for subfolder in subfolders:
    scene_id,im_id=subfolder.split('_')
    scene_id=int(scene_id)
    im_id=int(im_id)
    depth_path=os.path.join(dataset_dir,depth_path_temp.format(scene_id,im_id))
    try:
        depth = load_depth(depth_path)
    except:
        print(1)

    obj2sets=defaultdict(lambda :[])
    subfolder=os.path.join(result_dir,subfolder)

    detection_path=os.path.join(subfolder,'detection_dict.pkl')
    with open(detection_path, 'rb') as file:
        detection_dict = cPickle.load(file)

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
    pred_RT=np.zeros((len(combine),4,4))
    pred_s=np.zeros((len(combine),3))
    for obj_key in combine:
        obj_index=int(obj_key.split('_')[0])
        pred_R=np.array(trans_scale_dict[obj_key]['pred_R']).reshape(3,3)
        pred_scale=trans_scale_dict[obj_key]['pred_scale']
        pred_trans=trans_scale_dict[obj_key]['pred_trans']
        cat_name=trans_scale_dict[obj_key]['cat_name']






        mesh_path=os.path.join(subfolder,f"{obj_key}.obj")
        mesh = trimesh.load(mesh_path, validate=True)
        vertice=np.array(mesh.vertices)
        faces=np.array(mesh.faces)
        pred_cloud=sample_points_from_mesh(vertice,faces, 20000, fps=False, ratio=3)


        if cat_name=='mug':
            lx = 2 * np.amax(np.abs(pred_cloud[:, 0]))
            ly = 2 * np.amax(np.abs(pred_cloud[:, 1]))
            lz = 2 * np.amax(np.abs(pred_cloud[:, 2]))
        else:
            lx = max(pred_cloud[:, 0]) - min(pred_cloud[:, 0])
            ly = max(pred_cloud[:, 1]) - min(pred_cloud[:, 1])
            lz = max(pred_cloud[:, 2]) - min(pred_cloud[:, 2])

        match_bbox=np.array([lx,ly,lz])






        pred_render_mesh=pyrender.Mesh.from_trimesh(mesh)

        gt_id=trans_scale_dict[obj_key]['model_id']

        if gt_id not in id2mesh:
            gt_mesh_path=os.path.join(mesh_dir,gt_id+'.obj')
            gt_mesh = trimesh.load_mesh(gt_mesh_path)
            vertice=np.array(gt_mesh.vertices)
            faces=np.array(gt_mesh.faces)
            gt_mesh_cloud=sample_points_from_mesh(vertice,faces, 20000, fps=False, ratio=3)
            gt_render_mesh = pyrender.Mesh.from_trimesh(gt_mesh)
            id2mesh[gt_id]=gt_render_mesh
        else:
            gt_render_mesh=id2mesh[gt_id]




        gt_cloud=id2cloud[gt_id]
        gt_R=np.array(trans_scale_dict[obj_key]['gt_R']).reshape(3,3)
        gt_t=np.array(trans_scale_dict[obj_key]['gt_T']).reshape(3)
        gt_scale=np.linalg.norm(np.array(trans_scale_dict[obj_key]['gt_scales']))

        if cat_name=='mug':
            try:
                meta=mug_meta[gt_id]
            except:
                print('wrong','wrong_cat_name_for_mug')
                continue
            t0=meta[0]
            s0=meta[1]

            no_re_gt_cloud=gt_cloud/s0-t0
        else:
            t0=np.array([0,0,0])
            s0=1
            no_re_gt_cloud=gt_cloud
        index=random.sample(list(range(len(no_re_gt_cloud))),2048)
        no_re_gt_cloud=no_re_gt_cloud[index]
        no_re_gt_scale=gt_scale*s0
        no_re_gt_t=gt_t+np.einsum('ij,j->i',gt_R,t0)*no_re_gt_scale


        # show_open3d(match_cloud,no_re_gt_cloud)
        dist1, dist2=cd_loss(torch.from_numpy(no_re_gt_cloud)[None,:,:].float(),torch.from_numpy(pred_cloud)[None,:,:].float())
        cd = (torch.mean(dist1)) + (torch.mean(dist2)).item()
        cd_list.append(cd)

        re_pred_scale=pred_scale/s0
        re_pred_trans=pred_trans-np.einsum('ij,j->i',gt_R,t0)*gt_scale*s0




        # pred_scale_matrix=np.diag([pred_scale,pred_scale,pred_scale,1])
        pred_scale_matrix=np.diag([no_re_gt_scale,no_re_gt_scale,no_re_gt_scale,1])
        pred_pose= np.eye(4)
        pred_pose[:3,:3]=pred_R
        pred_pose[:3,3]=pred_trans
        pred_pose=pred_pose @ pred_scale_matrix



        pred_cloud_cam=np.einsum('ij,pj->pi',pred_R,pred_cloud)*no_re_gt_scale+pred_trans

        match_scale=match_bbox*no_re_gt_scale





        gt_mesh_pose= np.eye(4)
        gt_mesh_pose[:3,3]=no_re_gt_t
        gt_mesh_pose[:3,:3]=gt_R





        vsd_error=my_vsd(pred_render_mesh,gt_render_mesh,pred_pose,gt_mesh_pose,cam_K,depth)[0]
        # vsd_error=0
        vsd_list.append(vsd_error)
        gt_cloud_cam=np.einsum('ij,pj->pi',gt_R,gt_cloud)*gt_scale+gt_t

        gt_mesh_cloud_cam=np.einsum('ij,pj->pi',gt_R,gt_mesh_cloud)+no_re_gt_t

        pred_match_cloud=np.einsum('ij,pj->pi',pred_R,gt_cloud)*re_pred_scale+re_pred_trans

        pred_RT[obj_index,:3,:3]=pred_R
        pred_RT[obj_index,:3,3]=re_pred_trans
        pred_s[obj_index]=match_scale


        # show_open3d(pred_cloud_cam,gt_cloud_cam)
        # show_open3d(pred_match_cloud,gt_cloud_cam)
        # show_open3d(gt_mesh_cloud_cam,gt_cloud_cam)


        add_error=my_adi_err(pred_cloud_cam,gt_cloud_cam)
        add_list.append(add_error)

        r_diff=modelnet_r_diff(torch.from_numpy(gt_R).float(),torch.from_numpy(pred_R).float(),0).item()
        r_diff_list.append(r_diff)

        t_diff=np.linalg.norm(re_pred_trans - gt_t)
        t_diff_list.append(t_diff)

    detection_dict['pred_RTs'] = pred_RT
    detection_dict['pred_scales'] = pred_s
    pred_results.append(detection_dict)


with open(pred_result_save_path, 'wb') as file:
    pickle.dump(pred_results, file)


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












