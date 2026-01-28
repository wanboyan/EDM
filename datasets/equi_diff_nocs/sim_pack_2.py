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
from tqdm import tqdm



def do_sim(mesh_dir,obj_id_list,obj_scale_list,obj_R_list,obj_t_list,cam_K,cam_R,cam_t):
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
    for i,(model_id,scale) in enumerate(zip(obj_id_list,obj_scale_list)):
        OBJ_ASSET += f"""<mesh name="mesh_{i}" file="{model_id}.obj" scale="{scale} {scale} {scale}"/>\n"""
        OBJ_QUAT_BODY += f""" <body>
                            <joint name="joint_{i}" type="free" stiffness="0.1" damping="0.99" frictionloss="0.99"/>
                            <geom name="obj_{i}" mesh="mesh_{i}" rgba=".51 .6 .7 0.2" mass="100" />
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


    physics = mujoco.Physics.from_xml_string(xml)
    # Visualize the joint axis.
    option = mujoco.wrapper.core.MjvOption()
    option.flags[enums.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    option.flags[enums.mjtVisFlag.mjVIS_CONTACTFORCE] = False


    physics.model.vis.scale.contactwidth = 0.1
    physics.model.vis.scale.contactheight = 0.03
    physics.model.vis.scale.forcewidth = 0.05
    physics.model.vis.map.force = 0.1




    for i in range(len(obj_id_list)):

        obj_R=np.array(obj_R_list[i]).reshape(3,3)
        # obj_t=np.array(obj_t_list[i])/1000.0+0.02*np.random.randn(3)
        obj_t=np.array(obj_t_list[i])/1000.0
        #
        obj_R=np.matmul(ori_cam_R_T,obj_R.copy())
        obj_t=np.einsum('ij,j->i',ori_cam_R_T,obj_t-ori_cam_t)


        obj_R=torch.from_numpy(obj_R).reshape(1,3,3)
        obj_R=matrix_to_quaternion(obj_R)[0].numpy().tolist()
        obj_t=obj_t.tolist()
        new_pose=obj_t+obj_R

        physics.data.joint(f"joint_{i}").qpos=new_pose









    frames = []
    m_step=100
    physics.step(m_step)
    k_energy=physics.data.energy[1]


    contacts = physics.data.contact
    unique_pair=set()
    for j,cp in enumerate(contacts):
        forcetorque = np.zeros(6)
        mujoco.mj_contactForce(physics.model._model, physics.data._data, j, forcetorque)
        force=forcetorque[:3]
        geo_id_1=cp.geom1
        geo_id_2=cp.geom2
        threshold=10
        if geo_id_1 !=0:
            frame=cp.frame.reshape(3,3)
            global_froce=np.einsum('ij,i->j',frame,force)
            global_z_force=global_froce[2]
            xpos_1=physics.data.geom_xpos[geo_id_1]
            xpos_2=physics.data.geom_xpos[geo_id_2]
            c_pos=cp.pos
            z_1=xpos_1[2]
            z_2=xpos_2[2]
            z_c=c_pos[2]
            if abs(force[0])>=threshold:
                geo_name_1=mujoco.mj_id2name(physics.model._model,mujoco.mjtObj.mjOBJ_GEOM,geo_id_1)
                geo_name_2=mujoco.mj_id2name(physics.model._model,mujoco.mjtObj.mjOBJ_GEOM,geo_id_2)

                if z_1>=z_2:
                    unique_pair.add((geo_name_1,geo_name_2))
                    physics.model.geom_rgba[geo_id_1]=[0 ,1 ,0,0.2]
                    physics.model.geom_rgba[geo_id_2]=[1 ,0 ,0,0.3]
                elif z_1<z_2:
                    unique_pair.add((geo_name_2,geo_name_1))
                    physics.model.geom_rgba[geo_id_2]=[0 ,1 ,0,0.2]
                    physics.model.geom_rgba[geo_id_1]=[1 ,0 ,0,0.3]

    vis_flag=False
    if vis_flag:
        pixels = physics.render(width=width,height=height,camera_id='fix',
                                scene_option=option)
        frames.append(pixels)


        fig,ax=plt.subplots()
        im=ax.imshow(frames[0])
        def update(frame):
            im.set_array(frame)
            return im
        ani=FuncAnimation(fig,update,frames=frames,interval=100)
        plt.show()
    return unique_pair,k_energy






data_dir='/data_sata/pack/test'
mesh_dir='/data_nvme/pack/container/sim'
out_json_path='/data_sata/pack/test_sim.json'
start = 0
end = 1

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
        len_item = len(scene_gt_info_dict[str(data_id)])
        cam_R_w2c=scene_camera_dict[str(data_id)]['cam_R_w2c']
        cam_t_w2c=scene_camera_dict[str(data_id)]['cam_t_w2c']
        cam_K=scene_camera_dict[str(data_id)]['cam_K']
        obj_id_list=[]
        obj_scale_list=[]
        obj_R_list=[]
        obj_t_list=[]
        break_flag=False
        pick=False
        for obj_idx in range(len_item):
            frac=scene_gt_info_dict[str(data_id)][obj_idx]['visib_fract']
            cam_R_m2c = scene_gt_dict[str(data_id)][obj_idx]['cam_R_m2c']
            cam_t_m2c = scene_gt_dict[str(data_id)][obj_idx]['cam_t_m2c']
            scale= scene_gt_dict[str(data_id)][obj_idx]['scale']
            obj_id= scene_gt_dict[str(data_id)][obj_idx]['obj_id']
            obj_id_list.append(obj_id)
            obj_scale_list.append(scale)
            obj_R_list.append(cam_R_m2c)
            obj_t_list.append(cam_t_m2c)
            if frac<0.15 and frac>0.001:
                break_flag=True
                break
            if frac<0.35:
                pick=True
        if break_flag and not pick:
            continue
        pairs,k_energy=do_sim(mesh_dir,obj_id_list,obj_scale_list,obj_R_list,obj_t_list,
                                   cam_K,cam_R_w2c,cam_t_w2c)
        if len(pairs)>0:
            out_dict[str(folder_id)+'_'+str(data_id)]={'pairs':list(pairs),'k_energy':k_energy}

with open(out_json_path, 'w') as f:
    json.dump(out_dict, f)