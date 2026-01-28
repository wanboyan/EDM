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






def do_sim(mesh_dir,obj_id_list,obj_scale_list,obj_R_list,obj_t_list,cam_K,out_path):
    width=640
    height=480
    cam_R=np.eye(3)
    cam_t=np.zeros(3)

    new_cam_R=cam_R.copy()
    new_cam_R[1]=-cam_R[1]
    new_cam_R[2]=cam_R[2]
    ori_cam_R=cam_R
    cam_R=new_cam_R

    new_cam_t=cam_t.copy()
    new_cam_t[1]=-cam_t[1]
    new_cam_t[2]=cam_t[2]
    ori_cam_t=cam_t
    # cam_t=new_cam_t



    cam_R_T=cam_R.transpose()
    ori_cam_R_T=ori_cam_R.transpose()
    cam_K=np.array(cam_K).reshape(3,3)

    cam_pos=-np.einsum('ij,j->i', cam_R_T,cam_t)
    focal=cam_K[1,1]
    fovy=2*np.rad2deg(np.arctan2(1.0,focal/(height/2.0)))
    cam_x,cam_y,cam_z=cam_pos[0],cam_pos[1],cam_pos[2]
    x_axes_x,x_axes_y,x_axes_z=cam_R_T[0][0],cam_R_T[1][0],cam_R_T[2][0]
    y_axes_x,y_axes_y,y_axes_z=cam_R_T[0][1],cam_R_T[1][1],cam_R_T[2][1]

    OBJ_ASSET=""""""
    OBJ_QUAT_BODY=""""""
    for i,(model_id,scale) in enumerate(zip(obj_id_list,obj_scale_list)):
        OBJ_ASSET += f"""<mesh name="mesh_{i}" file="{model_id}.obj" scale="{scale} {scale} {scale}"/>\n"""
        OBJ_QUAT_BODY += f""" <body>
                            <joint name="joint_{i}" type="free" stiffness="0.1" damping="0.99" frictionloss="0.99"/>
                            <geom name="obj_{i}" mesh="mesh_{i}" rgba=".51 .6 .7 1" mass="100" />
                             </body>"""
    xml = f"""
    <mujoco model="pack">
      <option timestep=".001">
        <flag energy="enable"/>
        </option>
        <option integrator="RK4"/>
        <compiler angle="radian" meshdir="{mesh_dir}"/>
        <default class="collision">
            
            <geom contype="1" conaffinity="1" type="mesh" friction="1.0"  />
        </default>
        
        <asset>
            {OBJ_ASSET}
        </asset>
        <asset>
            <texture type="skybox" builtin="gradient" rgb1="0.6 0.6 0.6" rgb2="0 0 0" width="512" height="512"/> -->
            <texture name="texplane" type="2d" builtin="checker" rgb1=".25 .25 .25" rgb2=".3 .3 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
            <material name="matplane" reflectance="0.0" texture="texplane" texrepeat="1 1" texuniform="true"/>
        </asset>
        <worldbody>
            <camera name="fix" pos="{cam_x} {cam_y} {cam_z}" 
            xyaxes="{x_axes_x} {x_axes_y} {x_axes_z} {y_axes_x} {y_axes_y} {y_axes_z}" 
            fovy="{fovy}"
            />
            <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 -2" dir="0 0 1"/>
            {OBJ_QUAT_BODY}
    
        </worldbody>
    
    </mujoco>
    """

    try:
        physics = mujoco.Physics.from_xml_string(xml)
    except:
        print('parse error')
        return
    # Visualize the joint axis.
    option = mujoco.wrapper.core.MjvOption()
    option.flags[enums.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[enums.mjtVisFlag.mjVIS_CONTACTFORCE] = False


    physics.model.vis.scale.contactwidth = 0.1
    physics.model.vis.scale.contactheight = 0.03
    physics.model.vis.scale.forcewidth = 0.05
    physics.model.vis.map.force = 0.1




    for i in range(len(obj_id_list)):

        obj_R=np.array(obj_R_list[i]).reshape(3,3)
        # obj_t=np.array(obj_t_list[i])/1000.0+0.02*np.random.randn(3)
        obj_t=np.array(obj_t_list[i])
        #
        obj_R=np.matmul(ori_cam_R_T,obj_R.copy())
        obj_t=np.einsum('ij,j->i',ori_cam_R_T,obj_t-ori_cam_t)


        obj_R=torch.from_numpy(obj_R).reshape(1,3,3)
        obj_R=matrix_to_quaternion(obj_R)[0].numpy().tolist()
        obj_t=obj_t.tolist()
        new_pose=obj_t+obj_R

        physics.data.joint(f"joint_{i}").qpos=new_pose









    frames = []
    m_step=1
    physics.step(m_step)





    pixels = physics.render(width=width,height=height,camera_id='fix',
                            scene_option=option)
    pixels=pixels[:,:,(2,1,0)]
    cv2.imwrite(out_path,pixels)
    # frames.append(pixels)
    #
    #
    # fig,ax=plt.subplots()
    # im=ax.imshow(frames[0])
    # def update(frame):
    #     im.set_array(frame)
    #     return im
    # ani=FuncAnimation(fig,update,frames=frames,interval=100)
    # plt.savefig(out_path)
    return













if __name__ == '__main__':
    result_dir='/data_sata/pack/result/gcasp_result_500'
    dataset_dir='/data_nvme/nocs_data'
    detection_dir='/data_nvme/nocs_data/segmentation_results'
    latent_dir='/data_sata/pack/latent/equi_inv_52499'

    ply_dir='/data_sata/pack/ply/all'
    cloud_dir='/data_sata/pack/cloud'
    mesh_dir='/data_nvme/nocs_data/obj_models/real_test'

    depth_path_temp='Real/test/scene_{}/{:04d}_depth.png'
    detection_path_temp='REAL275/results_test_scene_{}_{:04d}.pkl'

    output_dir='/data_sata/pack/result/gcasp_result_500_vis'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(os.path.join(dataset_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
        mug_meta = cPickle.load(f)

    subfolders=[entry for entry in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir,entry))]



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

        detection_path=os.path.join(subfolder,'detection.pkl')
        with open(detection_path, 'rb') as file:
            detection_dict = cPickle.load(file)

        combine=[]
        for entry in os.listdir(subfolder):
            if entry.endswith('.obj'):
                obj_index=entry.split('.')[0]
                combine.append(obj_index)


        pred_obj_id_list=[]
        pred_obj_scale_list=[]
        pred_obj_R_list=[]
        pred_obj_t_list=[]

        gt_obj_id_list=[]
        gt_obj_scale_list=[]
        gt_obj_R_list=[]
        gt_obj_t_list=[]


        match=match_box(detection_dict['pred_bboxes'],detection_dict['gt_bboxes'])
        for obj_index in combine:
            obj_index=int(obj_index)
            match_index=match[obj_index]
            pred_score=detection_dict['pred_scores'][obj_index]
            pred_bbox=detection_dict['pred_bboxes'][obj_index]
            if pred_score<0.7:
                continue
            pred_scale=detection_dict['pred_scales'][obj_index]
            pred_RTs=detection_dict['pred_RTs'][obj_index]


            factor=np.cbrt(np.linalg.det(pred_RTs[:3, :3]))
            pred_R=pred_RTs[:3, :3]/factor
            pred_t=pred_RTs[:3,3]


            pred_obj_id_list.append(obj_index)
            pred_obj_scale_list.append(factor)
            pred_obj_R_list.append(pred_R)
            pred_obj_t_list.append(pred_t)





        do_sim(subfolder,pred_obj_id_list,pred_obj_scale_list,pred_obj_R_list,
               pred_obj_t_list,cam_K,out_path=os.path.join(output_dir,f'{scene_id}_{im_id}_diff.jpg'))











