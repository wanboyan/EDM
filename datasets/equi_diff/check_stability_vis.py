import numpy as np
import torch
import os
import json
from dm_control import mujoco as mujoco_dm
import mediapy as media
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pytorch3d.transforms import matrix_to_quaternion
from dm_control.mujoco.wrapper.mjbindings import enums
import mujoco.viewer
from tqdm import tqdm
import itertools
import torch.multiprocessing as mp
from collections import defaultdict
from DMviewer import DMviewer
from check_stability import do_sim

def do_sim_vis(args):
    mesh_dir,obj_id_list,obj2trans_scale,cam_K,cam_R,cam_t,highlight_index,vis_flag=args
    highlight_obj_id=obj_id_list[highlight_index]
    width=640
    height=480
    cam_R=np.array(cam_R).reshape(3,3)
    cam_t=np.array(cam_t)/1000.0
    new_cam_R=cam_R.copy()
    new_cam_R[1]=-cam_R[1]
    new_cam_R[2]=cam_R[2]
    ori_cam_R=cam_R
    cam_R=new_cam_R

    new_cam_t=cam_t.copy()
    new_cam_t[1]=-cam_t[1]
    new_cam_t[2]=cam_t[2]
    ori_cam_t=cam_t
    cam_t=new_cam_t



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
    for obj_id in obj_id_list:
        scale=obj2trans_scale[obj_id]['scale']
        OBJ_ASSET += f"""<mesh name="mesh_{obj_id}" file="{obj_id}.obj" scale="{scale} {scale} {scale}"/>\n"""
        OBJ_QUAT_BODY += f""" <body>
                            <joint name="joint_{obj_id}" type="free" stiffness="0.1" damping="0.99" frictionloss="0.99"/>
                            <geom name="obj_{obj_id}" mesh="mesh_{obj_id}" rgba=".51 .6 .7 0.4" mass="100" />
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
            <geom name="floor" pos="0 0 0" size="4 4 1" type="plane" material="matplane"/>
            <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
            {OBJ_QUAT_BODY}
    
        </worldbody>
    
    </mujoco>
    """


    physics = mujoco_dm.Physics.from_xml_string(xml)
    # Visualize the joint axis.
    option = mujoco_dm.wrapper.core.MjvOption()
    option.flags[enums.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[enums.mjtVisFlag.mjVIS_CONTACTFORCE] = False


    physics.model.vis.scale.contactwidth = 0.1
    physics.model.vis.scale.contactheight = 0.03
    physics.model.vis.scale.forcewidth = 0.05
    physics.model.vis.map.force = 0.1




    for obj_id in obj_id_list:

        obj_t=np.array(obj2trans_scale[obj_id]['trans'])
        #

        obj_t=np.einsum('ij,j->i',ori_cam_R_T,obj_t-ori_cam_t)

        obj_R=np.eye(3,dtype=np.float)
        obj_R=np.matmul(ori_cam_R_T,obj_R.copy())
        obj_R=torch.from_numpy(obj_R).reshape(1,3,3)
        obj_R=matrix_to_quaternion(obj_R)[0].numpy().tolist()


        # obj_R=[1.0,0.0,0.0,0.0]
        obj_t=obj_t.tolist()
        new_pose=obj_t+obj_R

        physics.data.joint(f"joint_{obj_id}").qpos=new_pose









    frames = []
    m_step=100
    physics.step(m_step)
    physics.named.model.geom_rgba[f'obj_{highlight_obj_id}']=[1 ,0 ,0,0.6]

    # pixels = physics.render(width=width,height=height,camera_id='fix',
    #                     scene_option=option)
    # mujoco.viewer.launch(physics.model._model,physics.data._data)
    viewer = DMviewer(physics)
    viewer.render()


    return 0





if __name__ == '__main__':
    data_dir='/data_sata/pack/test'
    result_dir='/data_sata/pack/results_noise'
    mesh_dir='/data_nvme/pack/container/sim'
    start = 0
    end = 1
    test_subset_dict=json.load(open('/data_sata/pack/test_sim.json'))

    subfolders=[entry for entry in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir,entry))]


    ds_dir=os.path.join(data_dir, 'ycbv/train_pbr')

    out_dict={}
    for folder_id in range(start, end):

        with open(os.path.join(ds_dir, f'{int(folder_id):06d}/scene_gt_info.json')) as f:
            scene_gt_info_dict = json.load(f)

        with open(os.path.join(ds_dir, f'{int(folder_id):06d}/scene_camera.json')) as f:
            scene_camera_dict = json.load(f)

        with open(os.path.join(ds_dir, f'{int(folder_id):06d}/scene_gt.json')) as f:
            scene_gt_dict = json.load(f)

        # for data_idx in range(len(self.scene_gt_info_dict)):
        for data_id in tqdm(scene_gt_info_dict.keys()):
            if data_id!='3':
                continue
            len_item = len(scene_gt_info_dict[str(data_id)])
            obj2sets=defaultdict(lambda :[])
            key=f'{folder_id}_{data_id}'
            if key not in subfolders:
                continue
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


            cam_R_w2c=scene_camera_dict[str(data_id)]['cam_R_w2c']
            cam_t_w2c=scene_camera_dict[str(data_id)]['cam_t_w2c']
            cam_K=scene_camera_dict[str(data_id)]['cam_K']

            k_energy_list=[]
            arg_list=[]
            pool=mp.Pool()
            for combine in combineations:
                vis_flag=None
                arg_list.append((subfolder,combine,trans_scale_dict,cam_K,cam_R_w2c,cam_t_w2c,high_light_index,vis_flag))

            results=pool.map(do_sim, arg_list)
            pool.close()
            pool.join()

            results=np.array(results)
            results_index=np.argsort(results)
            arg_best=(subfolder,combineations[results_index[0]],trans_scale_dict,cam_K,cam_R_w2c,cam_t_w2c,high_light_index,'best')
            arg_worst=(subfolder,combineations[results_index[6]],trans_scale_dict,cam_K,cam_R_w2c,cam_t_w2c,high_light_index,'worst')
            # do_sim_vis(arg_best)
            do_sim_vis(arg_worst)



