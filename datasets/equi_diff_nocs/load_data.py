import os
import cv2
import math
import random
import numpy as np
import _pickle as cPickle
# from config.config import *
import absl.flags as flags
from datasets.data_augmentation import defor_2D, get_rotation,dilate_mask
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
    def __init__(self,data_dir=None, obj_list=None,mode='',per_cat=None):
        self.data_dir=data_dir
        self.mode=mode
        self.obj_id_list=obj_list

        self.cat_names=['bottle','bowl','camera','can','laptop','mug']
        self.cat_id2cat_name={'03642806':'laptop','03797390':'mug','02946921':'can',
                         '02880940':'bowl','02942699':'camera','02876657':'bottle'}
        self.cat_name2cat_cls={'bottle': 0, 'bowl': 1, 'camera': 2, 'can': 3, 'laptop': 4, 'mug': 5}
        self.cat_name2count={'bottle': 0, 'bowl': 0, 'camera': 0, 'can': 0, 'laptop': 0, 'mug': 0}

        self.model_id2cat_id=json.load(open(os.path.join(self.data_dir,'cat_dict.json')))

        self.dataset_dir=os.path.join(self.data_dir,per_cat)

        if mode=='train':
            self.ds_dir = os.path.join(self.dataset_dir, 'train/ycbv/train_pbr')
            start = 0
            end = 100
            self.visib_fract_thresh = 0.15
        else:
            self.ds_dir = os.path.join(self.dataset_dir, 'test/ycbv/train_pbr')
            start = 0
            end = 1
            self.visib_fract_thresh = 0.0

        self.cloud_dir=FLAGS.cloud_dir

        self.all_gt_info = []
        self.r_vector=torch.load(FLAGS.occ_r_path)
        for folder_id in range(start, end):
            self.scene_gt_dict = {}
            self.scene_gt_info_dict = {}
            self.scene_camera_dict = {}
            try:
                with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_gt_info.json')) as f:
                    self.scene_gt_info_dict = json.load(f)
            except:
                print('total folder is',folder_id-1)
                break

            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_camera.json')) as f:
                self.scene_camera_dict = json.load(f)

            with open(os.path.join(self.ds_dir, f'{int(folder_id):06d}/scene_gt.json')) as f:
                self.scene_gt_dict = json.load(f)

            # for data_idx in range(len(self.scene_gt_info_dict)):
            for data_id in self.scene_gt_info_dict.keys():
                len_item = len(self.scene_gt_info_dict[str(data_id)])
                for obj_idx in range(len_item):
                    # self.all_gt_info[str(obj_id)] = []
                    # if self.scene_gt_dict[str(data_id)][obj_idx]['obj_id'] in self.obj_id_list:
                        if self.scene_gt_info_dict[str(data_id)][obj_idx]['visib_fract'] > self.visib_fract_thresh:
                            single_annot = {}
                            single_annot['obj_id'] = self.scene_gt_dict[str(data_id)][obj_idx]['obj_id']
                            cat_id=self.model_id2cat_id[single_annot['obj_id']]
                            cat_name=self.cat_id2cat_name[cat_id]
                            if per_cat in self.cat_names and cat_name!=per_cat:
                                raise "train only one cat but find more cat"
                            single_annot['folder_id'] = folder_id
                            single_annot['frame_id'] = int(data_id)
                            single_annot['cam_R_m2c'] = self.scene_gt_dict[str(data_id)][obj_idx]['cam_R_m2c']
                            single_annot['cam_t_m2c'] = self.scene_gt_dict[str(data_id)][obj_idx]['cam_t_m2c']
                            single_annot['scale'] = self.scene_gt_dict[str(data_id)][obj_idx]['scale']

                            single_annot['obj_index'] = obj_idx
                            single_annot['cam_K'] = self.scene_camera_dict[str(data_id)]['cam_K']
                            single_annot['depth_scale'] = self.scene_camera_dict[str(data_id)]['depth_scale']
                            # self.all_gt_info[str(obj_id)].append(single_annot)
                            self.all_gt_info.append(single_annot)
                            self.cat_name2count[cat_name]+=1

        # random.shuffle(self.all_gt_info)
        self.length=len(self.all_gt_info)
        print('total depth',len(self.all_gt_info))
        print(self.cat_name2count)





    def __len__(self):
        return self.length

    def __getitem__(self, index):
        frame_id=index
        data_gt_info = self.all_gt_info[frame_id]
        folder_id = data_gt_info['folder_id']
        frame_id = data_gt_info['frame_id']
        obj_index=data_gt_info['obj_index']
        model_id=data_gt_info['obj_id']
        if FLAGS.use_occ:
            latent_path=os.path.join(FLAGS.occ_latent_dir,f'{model_id}.npy')
            # if not os.path.exists(FLAGS.occ_r_path):
            #     occ_r_vector=torch.randn(768,3)
            #     occ_r_vector=F.normalize(occ_r_vector,dim=-1)
            #     torch.save(occ_r_vector,FLAGS.occ_r_path)
            # else:
            occ_r_vector=self.r_vector
        else:
            latent_path=os.path.join(FLAGS.latent_dir,f'{model_id}.npy')
            occ_r_vector=0
        try:
            latent_code=np.load(latent_path)
        except:
            print(f'latent code for {model_id} not exist')
            return self.__getitem__(index+1)

        model_scale=data_gt_info['scale']
        cloud_path=os.path.join(self.cloud_dir,model_id+'.npy')
        cloud=np.load(cloud_path)

        lx = max(cloud[:, 0]) - min(cloud[:, 0])
        ly = max(cloud[:, 1]) - min(cloud[:, 1])
        lz = max(cloud[:, 2]) - min(cloud[:, 2])
        cal_scale=np.array([lx,ly,lz])



        depth_scale=data_gt_info['depth_scale']
        rgb_parent_path = os.path.join(self.ds_dir, f'{int(folder_id):06d}', 'rgb')
        depth_parent_path = os.path.join(self.ds_dir, f'{int(folder_id):06d}', 'depth')
        mask_parent_path = os.path.join(self.ds_dir, f'{int(folder_id):06d}', 'mask_visib')

        rgb_path = os.path.join(rgb_parent_path, f'{int(frame_id):06d}.jpg')
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


        roi_mask_def = dilate_mask(roi_mask, rand_r=FLAGS.roi_mask_r, rand_pro=FLAGS.roi_mask_pro)

        # import matplotlib.pyplot as plt
        # plt.imshow(roi_img.transpose(1, 2, 0))
        # plt.show()
        #
        # roi_img[:,roi_mask_def==0]=0
        # import matplotlib.pyplot as plt
        # plt.imshow(roi_img.transpose(1, 2, 0))
        # plt.show()

        R_m2c = np.array(data_gt_info['cam_R_m2c']).reshape(3, 3)
        t_m2c = np.array(data_gt_info['cam_t_m2c'])


        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters(a=FLAGS.rotation_noise_angle)
        sphere_points = make_sphere(FLAGS.query_num,r=FLAGS.query_radius)

        # import matplotlib.pyplot as plt
        # plt.imshow(roi_img.transpose(1, 2, 0))
        # plt.show()
        K = np.array(data_gt_info['cam_K']).reshape(3, 3)

        model_id=data_gt_info['obj_id']
        cat_id=self.model_id2cat_id[model_id]
        cat_name=self.cat_id2cat_name[cat_id]
        cls=self.cat_name2cat_cls[cat_name]
        cls_vector=torch.zeros(len(self.cat_names)).float()
        cls_vector[cls]=1

        data_dict = {}
        data_dict['roi_mask_deform'] = torch.as_tensor(roi_mask_def, dtype=torch.float32).contiguous()
        data_dict['roi_img'] = torch.as_tensor(roi_img, dtype=torch.int).contiguous()
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
        data_dict['aug_scale'] = torch.as_tensor(bb_aug, dtype=torch.float32).contiguous()
        data_dict['scene_id'] = data_gt_info['folder_id']
        data_dict['img_id'] = data_gt_info['frame_id']
        data_dict['obj_id'] = data_gt_info['obj_id']
        data_dict['obj_index'] = data_gt_info['obj_index']
        data_dict['rgb'] = rgb
        data_dict['invalid']=invalid
        data_dict['model_points']=torch.as_tensor(cloud, dtype=torch.float32).contiguous()
        data_dict['scale']=torch.as_tensor(model_scale, dtype=torch.float32).contiguous()
        data_dict['latent_code']=torch.as_tensor(latent_code, dtype=torch.float32).contiguous()
        data_dict['cls_vector']=torch.as_tensor(cls_vector, dtype=torch.float32).contiguous()
        data_dict['occ_r_vector']=torch.as_tensor(occ_r_vector, dtype=torch.float32).contiguous()
        return data_dict

    def generate_aug_parameters(self, s_x=(0.7, 1.3), ax=50, ay=50, az=50, a=90):
        # for bb aug
        ex= np.random.rand(1)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]

        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array(ex, dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm

