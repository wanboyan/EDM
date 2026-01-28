import os
import cv2
import math
import random
import numpy as np
import _pickle as cPickle
# from config.config import *
import absl.flags as flags
from datasets.data_augmentation import defor_2D, get_rotation
FLAGS = flags.FLAGS
import json
import torch
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
from tools.eval_utils import load_depth, get_bbox
from tools.dataset_utils import *
from nfmodel.uti_tool import *
from pathlib import Path
from datasets.lmo.lmo_utils import *
import torch
import pickle
import h5py


class PoseDataset(data.Dataset):
    def __init__(self,data_dir=None, obj_list=None,mode=''):
        self.data_dir=data_dir
        self.mode=mode
        self.obj_id_list=obj_list

        if mode=='train':
            self.ds_dir = os.path.join(self.data_dir, 'train_pbr')
            start = 0
            end = 50
            self.visib_fract_thresh = 0.15
        else:
            self.ds_dir = os.path.join(self.data_dir, 'test')
            start = 2
            end = 3
            self.visib_fract_thresh = 0.0


        self.all_gt_info = []
        for folder_id in range(start, end):
            self.scene_gt_dict = {}
            self.scene_gt_info_dict = {}
            self.scene_camera_dict = {}
            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_gt_info.json')) as f:
                self.scene_gt_info_dict = json.load(f)

            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_camera.json')) as f:
                self.scene_camera_dict = json.load(f)

            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_gt.json')) as f:
                self.scene_gt_dict = json.load(f)

            # for data_idx in range(len(self.scene_gt_info_dict)):
            for data_id in self.scene_gt_info_dict.keys():
                len_item = len(self.scene_gt_info_dict[str(data_id)])
                for obj_idx in range(len_item):
                    # self.all_gt_info[str(obj_id)] = []
                    if self.scene_gt_dict[str(data_id)][obj_idx]['obj_id'] in self.obj_id_list:
                        if self.scene_gt_info_dict[str(data_id)][obj_idx]['visib_fract'] > self.visib_fract_thresh:
                            single_annot = {}
                            single_annot['folder_id'] = folder_id
                            single_annot['frame_id'] = int(data_id)
                            single_annot['cam_R_m2c'] = self.scene_gt_dict[str(data_id)][obj_idx]['cam_R_m2c']
                            single_annot['cam_t_m2c'] = self.scene_gt_dict[str(data_id)][obj_idx]['cam_t_m2c']
                            single_annot['obj_id'] = self.scene_gt_dict[str(data_id)][obj_idx]['obj_id']
                            single_annot['obj_index'] = obj_idx
                            single_annot['cam_K'] = self.scene_camera_dict[str(data_id)]['cam_K']
                            single_annot['depth_scale'] = self.scene_camera_dict[str(data_id)]['depth_scale']
                            # self.all_gt_info[str(obj_id)].append(single_annot)
                            self.all_gt_info.append(single_annot)


        # random.shuffle(self.all_gt_info)
        self.length=len(self.all_gt_info)
        print('total depth',len(self.all_gt_info))




    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frame_id=index
        data_gt_info = self.all_gt_info[frame_id]
        folder_id = data_gt_info['folder_id']
        frame_id = data_gt_info['frame_id']
        obj_index=data_gt_info['obj_index']

        depth_scale=data_gt_info['depth_scale']
        rgb_parent_path = os.path.join(self.ds_dir, f'{int(folder_id):06d}', 'rgb')
        depth_parent_path = os.path.join(self.ds_dir, f'{int(folder_id):06d}', 'depth')
        mask_parent_path = os.path.join(self.ds_dir, f'{int(folder_id):06d}', 'mask_visib')

        if self.mode=='train':
            rgb_path = os.path.join(rgb_parent_path, f'{int(frame_id):06d}.jpg')
        else:
            rgb_path = os.path.join(rgb_parent_path, f'{int(frame_id):06d}.png')
        depth_path = os.path.join(depth_parent_path, f'{int(frame_id):06d}.png')
        mask_path=os.path.join(mask_parent_path, f'{int(frame_id):06d}_{int(obj_index):06d}.png')

        mask = cv2.imread(mask_path,-1)
        depth = cv2.imread(depth_path, -1)
        rgb = cv2.imread(rgb_path)

        mask_target = mask.copy().astype(np.float)
        mask_target[mask ==0] = 0.0
        mask_target[mask >0] = 1.0

        x_min, y_min, x_max, y_max=calculate_bbox_from_mask(mask_target)
        im_H, im_W = rgb.shape[0], rgb.shape[1]
        bbox_xyxy = np.array([x_min, y_min, x_max, y_max])
        bbox_center, scale = aug_bbox_DZI(FLAGS, bbox_xyxy, im_H, im_W)
        coord_2d = get_2d_coord_np(im_W, im_H).transpose(1, 2, 0)

        roi_coord_2d = crop_resize_by_warp_affine(
            coord_2d, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)
        roi_img = crop_resize_by_warp_affine(
            rgb, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        ).transpose(2, 0, 1)

        roi_mask = crop_resize_by_warp_affine(
            mask_target, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_mask = np.expand_dims(roi_mask, axis=0)
        roi_depth = crop_resize_by_warp_affine(
            depth, bbox_center, scale, FLAGS.img_size, interpolation=cv2.INTER_NEAREST
        )
        roi_depth=roi_depth[None,:,:]*depth_scale

        depth_valid = roi_depth > 0
        roi_m_d_valid = roi_mask.astype(np.bool) * depth_valid
        if np.sum(roi_m_d_valid) <= 10:
            invalid=1
        else:
            invalid=0


        roi_mask_def = defor_2D(roi_mask, rand_r=FLAGS.roi_mask_r, rand_pro=FLAGS.roi_mask_pro)

        R_m2c = np.array(data_gt_info['cam_R_m2c']).reshape(3, 3)
        t_m2c = np.array(data_gt_info['cam_t_m2c'])


        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters(a=FLAGS.rotation_noise_angle)
        sphere_points = make_sphere(FLAGS.query_num,r=FLAGS.query_radius)

        # import matplotlib.pyplot as plt
        # plt.imshow(roi_img.transpose(1, 2, 0))
        # plt.show()
        K = np.array(data_gt_info['cam_K']).reshape(3, 3)


        data_dict = {}
        data_dict['roi_mask_deform'] = torch.as_tensor(roi_mask_def, dtype=torch.float32).contiguous()
        data_dict['roi_mask'] = torch.as_tensor(roi_mask, dtype=torch.float32).contiguous()
        data_dict['roi_depth'] = torch.as_tensor(roi_depth.astype(np.float32)).contiguous()
        data_dict['roi_coord_2d'] = torch.as_tensor(roi_coord_2d, dtype=torch.float32).contiguous()
        data_dict['rotation'] = torch.as_tensor(R_m2c, dtype=torch.float32).contiguous()
        data_dict['cam_K'] = torch.as_tensor(K.astype(np.float32)).contiguous()
        data_dict['roi_depth'] = torch.as_tensor(roi_depth.astype(np.float32)).contiguous()
        data_dict['translation'] = torch.as_tensor(t_m2c, dtype=torch.float32).contiguous()
        data_dict['sphere_points'] = torch.as_tensor(sphere_points,dtype=torch.float32).contiguous()
        data_dict['aug_rt_t'] = torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous()
        data_dict['aug_rt_R'] = torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        data_dict['scene_id'] = data_gt_info['folder_id']
        data_dict['img_id'] = data_gt_info['frame_id']
        data_dict['obj_id'] = data_gt_info['obj_id']
        data_dict['rgb'] = rgb
        data_dict['invalid']=invalid
        return data_dict

    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm

