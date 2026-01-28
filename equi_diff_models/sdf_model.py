#!/usr/bin/env python3

import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import sys
import os
from pathlib import Path
import numpy as np
import math

from einops import rearrange, reduce

from equi_diff_models.archs.sdf_decoder import *
from equi_diff_models.archs.encoders.conv_pointnet import ConvPointnet



class SdfModel(pl.LightningModule):

    def __init__(self, specs):
        super().__init__()

        self.specs = specs
        model_specs = self.specs["SdfModelSpecs"]
        self.hidden_dim = model_specs["hidden_dim"]
        self.latent_dim = model_specs["latent_dim"]
        self.skip_connection = model_specs.get("skip_connection", True)
        self.tanh_act = model_specs.get("tanh_act", False)
        self.pn_hidden = model_specs.get("pn_hidden_dim", self.latent_dim)

        self.pointnet = ConvPointnet(c_dim=self.latent_dim, hidden_dim=self.pn_hidden, plane_resolution=64)

        self.model = SdfDecoder(latent_size=self.latent_dim, hidden_dim=self.hidden_dim, skip_connection=self.skip_connection, tanh_act=self.tanh_act)

        self.model.train()
        #print(self.model)


    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), self.specs["sdf_lr"])
        return optimizer


    def training_step(self, x, idx):

        xyz = x['xyz'] # (B, 16000, 3)
        gt = x['gt_sdf'] # (B, 16000)
        pc = x['point_cloud'] # (B, 1024, 3)

        shape_features = self.pointnet(pc, xyz)

        pred_sdf = self.model(xyz, shape_features)

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction = 'none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        return sdf_loss



    def forward(self, pc, xyz):
        shape_features = self.pointnet(pc, xyz)

        return self.model(xyz, shape_features).squeeze()

    def forward_with_plane_features(self, plane_features, xyz):
        '''
        plane_features: B, D*3, res, res (e.g. B, 768, 64, 64)
        xyz: B, N, 3
        '''
        point_features = self.pointnet.forward_with_plane_features(plane_features, xyz) # point_features: B, N, D
        pred_sdf = self.model( torch.cat((xyz, point_features),dim=-1) )
        return pred_sdf # [B, num_points]

    def generate_mesh_2(self, plane_features,thred=0.01):
        N=128
        max_batch=int(2 ** 14)
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        num_samples = N ** 3

        samples.requires_grad = False

        head = 0

        while head < num_samples:
            with torch.no_grad():
                sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
                samples[head : min(head + max_batch, num_samples), 3]=self.forward_with_plane_features(plane_features,sample_subset.unsqueeze(0))[0,:,0]
                head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)
        import skimage.measure

        numpy_3d_sdf_tensor = sdf_values.data.cpu().numpy()
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=thred, spacing=[voxel_size] * 3
        )

        # transform from voxel coordinates to camera coordinates
        # note x and y are flipped in the output of marching_cubes
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
        return mesh_points,faces,normals
