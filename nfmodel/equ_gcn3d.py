f"""
@Author: Zhi-Hao Lin
@Contact: r08942062@ntu.edu.tw
@Time: 2020/03/06
@Document: Basic operation/blocks of 3D-GCN
"""

import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nfmodel.uti_tool import *
from nfmodel.vec_layers import *
import absl.flags as flags
FLAGS = flags.FLAGS
from pytorch3d.ops.knn import knn_points
from einops import rearrange
# def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int):
#     """
#     Return: (bs, vertice_num, neighbor_num)
#     """
#     _, neighbor_index, _ = knn_points(
#         vertices, vertices, K=neighbor_num, return_nn=True)
#     return neighbor_index

def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    # print('gcn neighbor')
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2)) #(bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim= 2) #(bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim= 2) #(bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k= 1, dim= -1, largest= False)[1]
    return nearest_index

def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)" ):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """

    bs, v, n = index.size()

    # ss = time.time()
    if bs==1:
        # id_0 = torch.arange(bs).view(-1, 1,1)
        tensor_indexed = tensor[torch.Tensor([[0]]).long(), index[0]].unsqueeze(dim=0)
    else:
            id_0 = torch.arange(bs).view(-1, 1, 1).long()
            tensor_indexed = tensor[id_0, index]

    # ee = time.time()
    # print('tensor_indexed time: ', str(ee - ss))
    return tensor_indexed

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    # ss = time.time()
    neighbors = indexing_neighbor(vertices, neighbor_index) # (bs, v, n, 3)

    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim= -1)
    return neighbor_direction_norm.float()

class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)
    
    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_num)", 
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        # ss = time.time()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        # R = get_rotation(0,0,0)
        # R = torch.from_numpy(R).cuda()
        # R = R.unsqueeze(0).repeat(bs,1,1).float() ## bs 3,3
        # vertices2 = torch.bmm(R,vertices.transpose(1,2)).transpose(2,1)
        # neighbor_direction_norm2 = get_neighbor_direction_norm(vertices2, neighbor_index)


        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)

        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim= 2) # (bs, vertice_num, kernel_num)
        return feature

class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments: 
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self, 
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        return feature_fuse

class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self, 
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice _num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map, neighbor_index) #(bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim= 2)[0] #(bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool,

import nfmodel.rotation as fr
import os

GAMMA_SIZE = 3
curdir=os.path.dirname(os.path.realpath(__file__))
ANCHOR_PATH = os.path.join(curdir,"sphere12.ply")

Rs, R_idx, canonical_relative = fr.icosahedron_so3_trimesh(ANCHOR_PATH, GAMMA_SIZE)
vs, v_adjs, v_level2s, v_opps, vRs = fr.icosahedron_trimesh_to_vertices(ANCHOR_PATH)

def get_anchorsV():
    """return 60*3*3 matrix as rotation anchors determined by the symmetry of icosahedron vertices"""
    return vRs.copy()

def get_anchorsV12():
    """return 12*3*3 matrix as the section (representative rotation) of icosahedron vertices.
    For each vertex on the sphere (icosahedron) (coset space S2 = SO(3)/SO(2)),
    pick one rotation as its representation in SO(3), which is also called a section function (G/H -> G)"""
    return vRs.reshape(12, 5, 3, 3)[:,0].copy()    # 12*3*3

def get_icosahedron_vertices():
    return vs.copy(), v_adjs.copy(), v_level2s.copy(), v_opps.copy(), vRs.copy()

def get_relativeV_index():
    """return two 60(rotation anchors)*12(indices on s2) index matrices"""
    trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)
    return trace_idx_ori.copy(), trace_idx_rot.copy()

def get_relativeV12_index():
    """return two 12(rotation anchors)*12(indices on s2) index matrices"""
    trace_idx_ori, trace_idx_rot = fr.get_relativeV_index(vRs, vs)  # 60(rotation anchors)*12(indices on s2), 60*12
    trace_idx_ori = trace_idx_ori.reshape(12, 5, 12)[:,0]   # 12(rotation anchors)*12(indices on s2)
    trace_idx_rot = trace_idx_rot.reshape(12, 5, 12)[:,0]   # 12(rotation anchors)*12(indices on s2)
    return trace_idx_ori.copy(), trace_idx_rot.copy()

def get_relativeVR_index(full=False):
    """if full==true, return two 60(rotation anchors)*60(indices on anchor rotations) index matrices,
    otherwise, return two 60(rotation anchors)*12(indices on the anchor rotations that are sections of icoshedron vertices) index matrices.
    The latter case is different from get_relativeV_index(), because here the second indices are in the range of 60.
    """
    trace_idx_ori, trace_idx_rot = fr.get_relativeR_index(vRs)
    # da     # find correspinding original element for each rotated (60,60)
    # bd     # find corresponding rotated element for each original
    trace_idx_rot = trace_idx_rot.swapaxes(0,1)    # db    # changed 11/10/2022, before it is transpose(0,1), which has no effect
    if full:
        return trace_idx_ori.copy(), trace_idx_rot.copy()

    trace_idx_ori = trace_idx_ori.reshape(-1, 12, 5)    # d,12,5
    trace_idx_rot = trace_idx_rot.reshape(-1, 12, 5)
    trace_idx_ori = trace_idx_ori[..., 0]   # d,12
    trace_idx_rot = trace_idx_rot[..., 0]   # d,12
    return trace_idx_ori.copy(), trace_idx_rot.copy()



def sort_level_vs(v_idxs, vs):
    idx_l1_0 = v_idxs[0]
    idx_l1_sorted = []
    idx_l1_sorted.append(idx_l1_0)
    vs_l1 = vs[v_idxs][:, :2]  # only take the xy plane
    vs_l1_0 = vs_l1[[0]]    # 1*2
    dtheta = 2 * np.pi / 5
    ct = np.cos(dtheta)
    st = np.sin(dtheta)
    rotmat = np.array([[ct, -st], [st, ct]], dtype=np.float32)
    for _ in range(4):
        vs_l1_0 = rotmat.dot(vs_l1_0.T).T   # 1*2
        diff = vs_l1 - vs_l1_0  # 5*2
        diff_l1 = np.abs(diff).sum(1)
        iidx_cur = np.argmin(diff_l1)
        idx_cur = v_idxs[iidx_cur]
        idx_l1_sorted.append(idx_cur)
    idx_l1_sorted = np.array(idx_l1_sorted)
    return idx_l1_sorted

class equi_conv(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=13, anchor_size=12, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.anchor_size = anchor_size
        self.is_surface=is_surface

        assert self.kernel_size == 13, f"kernel_size {kernel_size} not implemented" # c2, c1, 36(3*4+2*12)
        if not self.is_surface:
            self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,36))      # c2, c1, 36(3*4+2*12)
            self.bias=nn.Parameter(torch.FloatTensor(self.dim_out,5))

        self.k_pos_w=nn.Parameter(torch.FloatTensor(self.dim_out,36))

        ### permute the weights under rotations
        trace_idx_ori, trace_idx_rot = get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12

        full_trace_idx_ori,full_trace_idx_rot=get_relativeV_index()
        self.register_buffer('full_trace_idx_ori',torch.tensor(full_trace_idx_ori.astype(np.int64)))
        self.register_buffer('full_trace_idx_rot',torch.tensor(full_trace_idx_rot.astype(np.int64)))
        # trace_idxv_ori = trace_idxv_ori.transpose(1,0)  # 12(indices on s2)*12(rotation anchors)
        # trace_idxv_rot = trace_idxv_rot.transpose(1,0)  # 12*12

        # vertices = np.concatenate([kernels, np.zeros_like(kernels[[0]])], axis=0) # 13,3
        trace_idxv_ori = np.concatenate([trace_idx_ori,np.ones_like(trace_idx_ori[:, [0]])*12],axis=1)   # 12(na)*13(nk)
        trace_idxv_rot = np.concatenate([trace_idx_rot,np.ones_like(trace_idx_rot[:, [0]])*12],axis=1)   # 12*13

        self.register_buffer('trace_idxv_ori', torch.tensor(trace_idxv_ori.astype(np.int64)).contiguous())   # 12(na)*13(nk)
        self.register_buffer('trace_idxv_rot', torch.tensor(trace_idxv_rot.astype(np.int64)).contiguous())

        self.register_buffer('trace_idx_ori', torch.tensor(trace_idx_ori.astype(np.int64)).contiguous()) # 12(na rotations)*12(na channels)
        self.register_buffer('trace_idx_rot', torch.tensor(trace_idx_rot.astype(np.int64)).contiguous())

        ### pick the self, neighbor, level2, opposite, center indices
        vs, v_adjs, v_level2s, v_opps, _ = get_icosahedron_vertices() # 12*5, 12*5, 12
        self.register_buffer('vs', torch.from_numpy(vs).float())
        v0_adjs = v_adjs[0]         # 5
        v0_level2s = v_level2s[0]   # 5
        v0_opps = v_opps[0]         # a number
        inv_idxs = torch.empty(anchor_size, dtype=torch.int64)
        inv_idxs[0] = 0
        inv_idxs[v0_adjs] = 1
        inv_idxs[v0_level2s] = 2
        inv_idxs[v0_opps] = 3

        v0_adjs_sorted = sort_level_vs(v0_adjs, vs)
        v0_level2s_sorted = sort_level_vs(v0_level2s, vs)
        v0_adjs_sorted = torch.tensor(v0_adjs_sorted, dtype=torch.int64)
        v0_level2s_sorted = torch.tensor(v0_level2s_sorted, dtype=torch.int64)

        idx_map = torch.empty(kernel_size * anchor_size, dtype=torch.int64) # each element is an index in the range 36
        ### the three kernel points on the z axis
        idx_map[:anchor_size] = inv_idxs
        idx_map[v0_opps*anchor_size:(v0_opps+1)*anchor_size ] = inv_idxs + 4
        idx_map[-anchor_size:] = inv_idxs + 8
        ### the rest kernel points on the 2 rings
        idx_seq = torch.arange(12,24, dtype=torch.int64)
        idx_seq2 = torch.arange(24,36, dtype=torch.int64)
        idx_map[v0_adjs_sorted[0]*anchor_size:(v0_adjs_sorted[0]+1)*anchor_size] = idx_seq
        idx_map[v0_level2s_sorted[0]*anchor_size:(v0_level2s_sorted[0]+1)*anchor_size] = idx_seq2

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[1]*anchor_size:(v0_adjs_sorted[1]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[1]*anchor_size:(v0_level2s_sorted[1]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[2]*anchor_size:(v0_adjs_sorted[2]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[2]*anchor_size:(v0_level2s_sorted[2]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[3]*anchor_size:(v0_adjs_sorted[3]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[3]*anchor_size:(v0_level2s_sorted[3]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[4]*anchor_size:(v0_adjs_sorted[4]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[4]*anchor_size:(v0_level2s_sorted[4]+1)*anchor_size] = idx_seq2[idx_seq_new]
        self.register_buffer('idx_map', idx_map.contiguous())

        idxs_k = self.trace_idxv_rot.transpose(0,1)[:,None,:].expand(-1, anchor_size, -1)  # a(rotations),k -> k, a(channels), a(rotations)

        idxs_k_norm=self.trace_idxv_rot.transpose(0,1)[None,:,:].expand(self.dim_out,-1,-1)
        self.register_buffer('idxs_k_norm', idxs_k_norm.contiguous())
        idxs_a = self.trace_idx_rot.transpose(0,1)[None].expand(kernel_size, -1, -1) # a(rotations),a(channels) -> k, a(channels), a(rotations)

        idxs_k_w = idxs_k[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)# c2, c1, k, a(channels), a(rotations)
        idxs_a_w = idxs_a[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_k_w', idxs_k_w.contiguous())  #   c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_a_w', idxs_a_w.contiguous())  #   c2, c1, k, a(channels), a(rotations)

        idxs_k_pos = idxs_k[None].expand(self.dim_out, -1,-1,-1)
        idxs_a_pos = idxs_a[None].expand(self.dim_out, -1, -1, -1)
        self.register_buffer('idxs_k_pos', idxs_k_pos.contiguous())
        self.register_buffer('idxs_a_pos', idxs_a_pos.contiguous())

        idxs_k_bias = self.trace_idxv_rot.transpose(0,1)
        idxs_k_bias = idxs_k_bias[None].expand(self.dim_out, -1, -1)

        self.register_buffer('idxs_k_bias', idxs_k_bias.contiguous())


        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        if not self.is_surface:
            nn.init.xavier_normal_(self.W)
            nn.init.xavier_normal_(self.bias)
        nn.init.xavier_normal_(self.k_pos_w)

    def forward(self, neighbor_index,vertices,feature_map=None):
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        k_pos_w_ori=self.k_pos_w[:,self.idx_map].reshape(self.dim_out,self.kernel_size,self.anchor_size)
        k_pos_w = k_pos_w_ori[..., None].expand(-1,-1,-1, self.anchor_size)

        k_pos_w = torch.gather(k_pos_w, 1, self.idxs_k_pos)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)
        k_pos_w = torch.gather(k_pos_w, 2, self.idxs_a_pos) # [dim_out,k,k-1,r]

        vs_theta=torch.einsum('bpnc,ac->bapn',neighbor_direction_norm,self.vs)



        k_pos_ori=torch.einsum("dka,ac->dkc",k_pos_w_ori,self.vs)
        k_pos_ori_norm=torch.norm(k_pos_ori,dim=-1)[:,:,None].expand(-1,-1,self.anchor_size)
        idxs_k_norm=self.idxs_k_norm

        k_pos_ori_norm=torch.gather(k_pos_ori_norm,1,idxs_k_norm)[None,:,:,None,None,:]+1e-5

        # k_cloud_1=k_pos[0,:,:,0].detach().cpu().numpy()
        # k_cloud_2=k_pos[1,:,:,0].detach().cpu().numpy()
        # show_open3d(k_cloud_1,k_cloud_2)

        theta = torch.einsum('dkar,bapn->bdkpnr',k_pos_w,vs_theta)/k_pos_ori_norm

        theta = self.relu(theta)[:,:,:self.anchor_size,:,:,:]
        if not self.is_surface:
            W = self.W[:,:,self.idx_map].reshape(self.dim_out, self.dim_in, self.kernel_size, self.anchor_size)    #C2,C1,kernel_size * anchor_size  #C2,C1,kernel_size * anchor_size
            W = W[..., None].expand(-1,-1,-1,-1, self.anchor_size)
            W = torch.gather(W, 2, self.idxs_k_w)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)
            W = torch.gather(W, 3, self.idxs_a_w)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels permuted),a(rotations)

            bias_idx_map=self.idx_map[:self.anchor_size].unsqueeze(0)
            bias_idx_map=torch.cat([bias_idx_map,torch.ones_like(bias_idx_map)[:,0:1]*4],dim=-1)[0]


            bias = self.bias[:,bias_idx_map].reshape(self.dim_out, self.kernel_size)    #C2,C1,kernel_size * anchor_size  #C2,C1,kernel_size * anchor_size

            bias= bias[..., None].expand(-1,-1, self.anchor_size)
            bias = torch.gather(bias, 1, self.idxs_k_bias)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)

            try:
                feature_out = torch.einsum("dckar, bcpa->bdkpr", W, feature_map)
            except:
                print(1)
            feature_out=feature_out+bias[None,:,:,None,:]

            kernel_activation=feature_out[:,:,:self.anchor_size,:,:]
            new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,None,:,None].expand(-1,self.dim_out,self.anchor_size,-1,self.anchor_size)
            kernel_activation=torch.gather(kernel_activation,3,new_neighbor_index).reshape(bs,self.dim_out,self.anchor_size,vertice_num,neighbor_num,self.anchor_size)
            kernel_activation=kernel_activation*theta

            kernel_activation=torch.max(kernel_activation,dim=4)[0]
            kernel_activation=torch.sum(kernel_activation,dim=2)



            center_activation=feature_out[:,:,self.anchor_size,:,:]
            fuse_fea=center_activation+kernel_activation
            return fuse_fea
        else:
            kernel_activation=torch.max(theta,dim=4)[0]
            kernel_activation=torch.sum(kernel_activation,dim=2)
            return kernel_activation



class equi_conv_bin(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=13, anchor_size=12, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.anchor_size = anchor_size
        self.is_surface=is_surface

        assert self.kernel_size == 13, f"kernel_size {kernel_size} not implemented" # c2, c1, 36(3*4+2*12)
        if not self.is_surface:
            self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,36))      # c2, c1, 36(3*4+2*12)
            self.bias=nn.Parameter(torch.FloatTensor(self.dim_out,5))

        self.k_pos_w=nn.Parameter(torch.FloatTensor(self.dim_out,36))

        ### permute the weights under rotations
        trace_idx_ori, trace_idx_rot = get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12

        full_trace_idx_ori,full_trace_idx_rot=get_relativeV_index()
        self.register_buffer('full_trace_idx_ori',torch.tensor(full_trace_idx_ori.astype(np.int64)))
        self.register_buffer('full_trace_idx_rot',torch.tensor(full_trace_idx_rot.astype(np.int64)))
        # trace_idxv_ori = trace_idxv_ori.transpose(1,0)  # 12(indices on s2)*12(rotation anchors)
        # trace_idxv_rot = trace_idxv_rot.transpose(1,0)  # 12*12

        # vertices = np.concatenate([kernels, np.zeros_like(kernels[[0]])], axis=0) # 13,3
        trace_idxv_ori = np.concatenate([trace_idx_ori,np.ones_like(trace_idx_ori[:, [0]])*12],axis=1)   # 12(na)*13(nk)
        trace_idxv_rot = np.concatenate([trace_idx_rot,np.ones_like(trace_idx_rot[:, [0]])*12],axis=1)   # 12*13

        self.register_buffer('trace_idxv_ori', torch.tensor(trace_idxv_ori.astype(np.int64)).contiguous())   # 12(na)*13(nk)
        self.register_buffer('trace_idxv_rot', torch.tensor(trace_idxv_rot.astype(np.int64)).contiguous())

        self.register_buffer('trace_idx_ori', torch.tensor(trace_idx_ori.astype(np.int64)).contiguous()) # 12(na rotations)*12(na channels)
        self.register_buffer('trace_idx_rot', torch.tensor(trace_idx_rot.astype(np.int64)).contiguous())

        ### pick the self, neighbor, level2, opposite, center indices
        vs, v_adjs, v_level2s, v_opps, _ = get_icosahedron_vertices() # 12*5, 12*5, 12
        self.register_buffer('vs', torch.from_numpy(vs).float())
        v0_adjs = v_adjs[0]         # 5
        v0_level2s = v_level2s[0]   # 5
        v0_opps = v_opps[0]         # a number
        inv_idxs = torch.empty(anchor_size, dtype=torch.int64)
        inv_idxs[0] = 0
        inv_idxs[v0_adjs] = 1
        inv_idxs[v0_level2s] = 2
        inv_idxs[v0_opps] = 3

        v0_adjs_sorted = sort_level_vs(v0_adjs, vs)
        v0_level2s_sorted = sort_level_vs(v0_level2s, vs)
        v0_adjs_sorted = torch.tensor(v0_adjs_sorted, dtype=torch.int64)
        v0_level2s_sorted = torch.tensor(v0_level2s_sorted, dtype=torch.int64)

        idx_map = torch.empty(kernel_size * anchor_size, dtype=torch.int64) # each element is an index in the range 36
        ### the three kernel points on the z axis
        idx_map[:anchor_size] = inv_idxs
        idx_map[v0_opps*anchor_size:(v0_opps+1)*anchor_size ] = inv_idxs + 4
        idx_map[-anchor_size:] = inv_idxs + 8
        ### the rest kernel points on the 2 rings
        idx_seq = torch.arange(12,24, dtype=torch.int64)
        idx_seq2 = torch.arange(24,36, dtype=torch.int64)
        idx_map[v0_adjs_sorted[0]*anchor_size:(v0_adjs_sorted[0]+1)*anchor_size] = idx_seq
        idx_map[v0_level2s_sorted[0]*anchor_size:(v0_level2s_sorted[0]+1)*anchor_size] = idx_seq2

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[1]*anchor_size:(v0_adjs_sorted[1]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[1]*anchor_size:(v0_level2s_sorted[1]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[2]*anchor_size:(v0_adjs_sorted[2]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[2]*anchor_size:(v0_level2s_sorted[2]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[3]*anchor_size:(v0_adjs_sorted[3]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[3]*anchor_size:(v0_level2s_sorted[3]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[4]*anchor_size:(v0_adjs_sorted[4]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[4]*anchor_size:(v0_level2s_sorted[4]+1)*anchor_size] = idx_seq2[idx_seq_new]
        self.register_buffer('idx_map', idx_map.contiguous())

        idxs_k = self.trace_idxv_rot.transpose(0,1)[:,None,:].expand(-1, anchor_size, -1)  # a(rotations),k -> k, a(channels), a(rotations)

        idxs_k_norm=self.trace_idxv_rot.transpose(0,1)[None,:,:].expand(self.dim_out,-1,-1)
        self.register_buffer('idxs_k_norm', idxs_k_norm.contiguous())
        idxs_a = self.trace_idx_rot.transpose(0,1)[None].expand(kernel_size, -1, -1) # a(rotations),a(channels) -> k, a(channels), a(rotations)

        idxs_k_w = idxs_k[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)# c2, c1, k, a(channels), a(rotations)
        idxs_a_w = idxs_a[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_k_w', idxs_k_w.contiguous())  #   c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_a_w', idxs_a_w.contiguous())  #   c2, c1, k, a(channels), a(rotations)

        idxs_k_pos = idxs_k[None].expand(self.dim_out, -1,-1,-1)
        idxs_a_pos = idxs_a[None].expand(self.dim_out, -1, -1, -1)
        self.register_buffer('idxs_k_pos', idxs_k_pos.contiguous())
        self.register_buffer('idxs_a_pos', idxs_a_pos.contiguous())

        idxs_k_bias = self.trace_idxv_rot.transpose(0,1)
        idxs_k_bias = idxs_k_bias[None].expand(self.dim_out, -1, -1)

        self.register_buffer('idxs_k_bias', idxs_k_bias.contiguous())


        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        if not self.is_surface:
            nn.init.xavier_normal_(self.W)
            nn.init.xavier_normal_(self.bias)
        nn.init.xavier_normal_(self.k_pos_w)

    def forward(self, neighbor_index,vertices,feature_map=None):
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        k_pos_w_ori=self.k_pos_w[:,self.idx_map].reshape(self.dim_out,self.kernel_size,self.anchor_size)
        k_pos_w = k_pos_w_ori[..., None].expand(-1,-1,-1, self.anchor_size)

        k_pos_w = torch.gather(k_pos_w, 1, self.idxs_k_pos)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)
        k_pos_w = torch.gather(k_pos_w, 2, self.idxs_a_pos) # [dim_out,k,k-1,r]

        vs_theta=torch.einsum('bpnc,ac->bapn',neighbor_direction_norm,self.vs)



        k_pos_ori=torch.einsum("dka,ac->dkc",k_pos_w_ori,self.vs)
        k_pos_ori_norm=torch.norm(k_pos_ori,dim=-1)[:,:,None].expand(-1,-1,self.anchor_size)
        idxs_k_norm=self.idxs_k_norm

        k_pos_ori_norm=torch.gather(k_pos_ori_norm,1,idxs_k_norm)[None,:,:,None,None,:]+1e-5

        # k_cloud_1=k_pos[0,:,:,0].detach().cpu().numpy()
        # k_cloud_2=k_pos[1,:,:,0].detach().cpu().numpy()
        # show_open3d
        if not self.is_surface:
            W = self.W[:,:,self.idx_map].reshape(self.dim_out, self.dim_in, self.kernel_size, self.anchor_size)    #C2,C1,kernel_size * anchor_size  #C2,C1,kernel_size * anchor_size
            W = W[..., None].expand(-1,-1,-1,-1, self.anchor_size)
            W = torch.gather(W, 2, self.idxs_k_w)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)
            W = torch.gather(W, 3, self.idxs_a_w)
            bias_idx_map=self.idx_map[:self.anchor_size].unsqueeze(0)
            bias_idx_map=torch.cat([bias_idx_map,torch.ones_like(bias_idx_map)[:,0:1]*4],dim=-1)[0]


            bias = self.bias[:,bias_idx_map].reshape(self.dim_out, self.kernel_size)    #C2,C1,kernel_size * anchor_size  #C2,C1,kernel_size * anchor_size

            bias= bias[..., None].expand(-1,-1, self.anchor_size)
            bias = torch.gather(bias, 1, self.idxs_k_bias)
            feature_out = torch.einsum("dckar, bcpa->bdkpr", W, feature_map)
            feature_out=feature_out+bias[None,:,:,None,:]

            kernel_activation=feature_out[:,:,:self.anchor_size,:,:]
            center_activation=feature_out[:,:,self.anchor_size,:,:]

        activation_bins=[]
        for i in range(12):
            k_pos_w_bin=k_pos_w[:,:,:,i:i+1]
            k_pos_ori_norm_bin=k_pos_ori_norm[:,:,:,:,:,i:i+1]

            theta_bin = torch.einsum('dkar,bapn->bdkpnr',k_pos_w_bin,vs_theta)/k_pos_ori_norm_bin

            theta_bin = self.relu(theta_bin)[:,:,:self.anchor_size,:,:,:]
            if not self.is_surface:
                kernel_activation_bin=kernel_activation[:,:,:,:,i:i+1]
                new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,None,:,None].expand(-1,self.dim_out,self.anchor_size,-1,-1)
                kernel_activation_bin=torch.gather(kernel_activation_bin,3,new_neighbor_index).reshape(bs,self.dim_out,self.anchor_size,vertice_num,neighbor_num,-1)
                kernel_activation_bin=kernel_activation_bin*theta_bin
            else:
                kernel_activation_bin=theta_bin
            kernel_activation_bin=torch.max(kernel_activation_bin,dim=4)[0]
            kernel_activation_bin=torch.sum(kernel_activation_bin,dim=2)

            activation_bins.append(kernel_activation_bin)

        kernel_activation=torch.cat(activation_bins,dim=-1)
        if not self.is_surface:
            fuse_fea=center_activation+kernel_activation
        else:
            fuse_fea=kernel_activation
        return fuse_fea


class equ_pool_layer(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice _num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        dim_out=feature_map.shape[1]
        anchor_size=12
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,dim_out,-1,anchor_size)
        neighbor_feature=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,dim_out,vertice_num,self.neighbor_num,anchor_size)
        pooled_feature = torch.max(neighbor_feature, dim= 3)[0] #(bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, :,sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool,feature_map_pool




class equi_conv2(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=13, anchor_size=12, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.anchor_size = anchor_size
        self.is_surface=is_surface

        assert self.kernel_size == 13, f"kernel_size {kernel_size} not implemented" # c2, c1, 36(3*4+2*12)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,36))      # c2, c1, 36(3*4+2*12)


        ### permute the weights under rotations
        trace_idx_ori, trace_idx_rot = get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12

        full_trace_idx_ori,full_trace_idx_rot=get_relativeV_index()
        self.register_buffer('full_trace_idx_ori',torch.tensor(full_trace_idx_ori.astype(np.int64)))
        self.register_buffer('full_trace_idx_rot',torch.tensor(full_trace_idx_rot.astype(np.int64)))
        # trace_idxv_ori = trace_idxv_ori.transpose(1,0)  # 12(indices on s2)*12(rotation anchors)
        # trace_idxv_rot = trace_idxv_rot.transpose(1,0)  # 12*12

        # vertices = np.concatenate([kernels, np.zeros_like(kernels[[0]])], axis=0) # 13,3
        trace_idxv_ori = np.concatenate([trace_idx_ori,np.ones_like(trace_idx_ori[:, [0]])*12],axis=1)   # 12(na)*13(nk)
        trace_idxv_rot = np.concatenate([trace_idx_rot,np.ones_like(trace_idx_rot[:, [0]])*12],axis=1)   # 12*13

        self.register_buffer('trace_idxv_ori', torch.tensor(trace_idxv_ori.astype(np.int64)).contiguous())   # 12(na)*13(nk)
        self.register_buffer('trace_idxv_rot', torch.tensor(trace_idxv_rot.astype(np.int64)).contiguous())

        self.register_buffer('trace_idx_ori', torch.tensor(trace_idx_ori.astype(np.int64)).contiguous()) # 12(na rotations)*12(na channels)
        self.register_buffer('trace_idx_rot', torch.tensor(trace_idx_rot.astype(np.int64)).contiguous())

        ### pick the self, neighbor, level2, opposite, center indices
        vs, v_adjs, v_level2s, v_opps, _ = get_icosahedron_vertices() # 12*5, 12*5, 12
        self.register_buffer('vs', torch.from_numpy(vs).float())
        v0_adjs = v_adjs[0]         # 5
        v0_level2s = v_level2s[0]   # 5
        v0_opps = v_opps[0]         # a number
        inv_idxs = torch.empty(anchor_size, dtype=torch.int64)
        inv_idxs[0] = 0
        inv_idxs[v0_adjs] = 1
        inv_idxs[v0_level2s] = 2
        inv_idxs[v0_opps] = 3

        v0_adjs_sorted = sort_level_vs(v0_adjs, vs)
        v0_level2s_sorted = sort_level_vs(v0_level2s, vs)
        v0_adjs_sorted = torch.tensor(v0_adjs_sorted, dtype=torch.int64)
        v0_level2s_sorted = torch.tensor(v0_level2s_sorted, dtype=torch.int64)

        idx_map = torch.empty(kernel_size * anchor_size, dtype=torch.int64) # each element is an index in the range 36
        ### the three kernel points on the z axis
        idx_map[:anchor_size] = inv_idxs
        idx_map[v0_opps*anchor_size:(v0_opps+1)*anchor_size ] = inv_idxs + 4
        idx_map[-anchor_size:] = inv_idxs + 8
        ### the rest kernel points on the 2 rings
        idx_seq = torch.arange(12,24, dtype=torch.int64)
        idx_seq2 = torch.arange(24,36, dtype=torch.int64)
        idx_map[v0_adjs_sorted[0]*anchor_size:(v0_adjs_sorted[0]+1)*anchor_size] = idx_seq
        idx_map[v0_level2s_sorted[0]*anchor_size:(v0_level2s_sorted[0]+1)*anchor_size] = idx_seq2

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[4,0,1,2,3]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[1]*anchor_size:(v0_adjs_sorted[1]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[1]*anchor_size:(v0_level2s_sorted[1]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[3,4,0,1,2]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[2]*anchor_size:(v0_adjs_sorted[2]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[2]*anchor_size:(v0_level2s_sorted[2]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[2,3,4,0,1]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[3]*anchor_size:(v0_adjs_sorted[3]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[3]*anchor_size:(v0_level2s_sorted[3]+1)*anchor_size] = idx_seq2[idx_seq_new]

        idx_seq_new = torch.empty(anchor_size, dtype=torch.int64)
        v0_adjs_sorted_shifted = v0_adjs_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_adjs_sorted] = v0_adjs_sorted_shifted
        v0_level2s_sorted_shifted = v0_level2s_sorted[[1,2,3,4,0]]
        idx_seq_new[v0_level2s_sorted] = v0_level2s_sorted_shifted
        idx_seq_new[0] = 0
        idx_seq_new[v0_opps] = v0_opps
        idx_map[v0_adjs_sorted[4]*anchor_size:(v0_adjs_sorted[4]+1)*anchor_size] = idx_seq[idx_seq_new]
        idx_map[v0_level2s_sorted[4]*anchor_size:(v0_level2s_sorted[4]+1)*anchor_size] = idx_seq2[idx_seq_new]
        self.register_buffer('idx_map', idx_map.contiguous())

        idxs_k = self.trace_idxv_rot.transpose(0,1)[:,None,:].expand(-1, anchor_size, -1)  # a(rotations),k -> k, a(channels), a(rotations)

        idxs_k_norm=self.trace_idxv_rot.transpose(0,1)[None,:,:].expand(self.dim_out,-1,-1)
        self.register_buffer('idxs_k_norm', idxs_k_norm.contiguous())
        idxs_a = self.trace_idx_rot.transpose(0,1)[None].expand(kernel_size, -1, -1) # a(rotations),a(channels) -> k, a(channels), a(rotations)

        idxs_k_w = idxs_k[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        idxs_a_w = idxs_a[None,None].expand(self.dim_out, self.dim_in, -1,-1,-1)  # c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_k_w', idxs_k_w.contiguous())  #   c2, c1, k, a(channels), a(rotations)
        self.register_buffer('idxs_a_w', idxs_a_w.contiguous())  #   c2, c1, k, a(channels), a(rotations)

        idxs_k_pos = idxs_k[None].expand(self.dim_out, -1,-1,-1)
        idxs_a_pos = idxs_a[None].expand(self.dim_out, -1, -1, -1)
        self.register_buffer('idxs_k_pos', idxs_k_pos.contiguous())
        self.register_buffer('idxs_a_pos', idxs_a_pos.contiguous())

        idxs_k_bias = self.trace_idxv_rot.transpose(0,1)
        idxs_k_bias = idxs_k_bias[None].expand(self.dim_out, -1, -1)

        self.register_buffer('idxs_k_bias', idxs_k_bias.contiguous())


        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        nn.init.xavier_normal_(self.W)


    def forward(self, neighbor_index,vertices,feature_map=None):

        bs, vertice_num, neighbor_num = neighbor_index.size()
        if self.is_surface:
            feature_map=torch.ones_like(vertices)[:,None,:,0:1].expand(-1,-1,-1,self.anchor_size)
            feature_map.requires_grad_(True)
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        vs_norm=F.normalize(self.vs,dim=1)
        vs_theta=torch.einsum('bpnc,ac->bapn',neighbor_direction_norm,vs_norm)
        if self.is_surface:
            inter_w=vs_theta
        else:
            inter_w=F.softmax(vs_theta,dim=-1)

        new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,self.dim_in,-1,self.anchor_size)
        feature_neighbor=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,self.dim_in,vertice_num,neighbor_num,self.anchor_size)

        new_feat=torch.einsum('bkpn,bcpna->bckpa',inter_w,feature_neighbor)
        new_feat=torch.cat([new_feat,feature_map[:,:,None,:,:]],dim=2)

        W = self.W[:,:,self.idx_map].reshape(self.dim_out, self.dim_in, self.kernel_size, self.anchor_size)    #C2,C1,kernel_size * anchor_size  #C2,C1,kernel_size * anchor_size
        W = W[..., None].expand(-1,-1,-1,-1, self.anchor_size)
        W = torch.gather(W, 2, self.idxs_k_w)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels),a(rotations)
        W = torch.gather(W, 3, self.idxs_a_w)  # c2,c1,k,a(channels),a(rotations) -> c2,c1,k,a(channels permuted),a(rotations)




        feature_out = torch.einsum("dckar, bckpa->bdpr", W, new_feat)

        return feature_out

class equi_conv3(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.is_surface=is_surface

        if not self.is_surface:
            self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,self.kernel_size,3))      # c2, c1, 36(3*4+2*12)
            self.W_center=nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in))
        self.directions=nn.Parameter(torch.FloatTensor(self.dim_out,self.kernel_size,3))
        Rs=get_anchorsV()
        self.register_buffer('Rs', torch.from_numpy(Rs).float().contiguous())
        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        if not self.is_surface:
            nn.init.xavier_normal_(self.W)
            nn.init.xavier_normal_(self.W_center)
        nn.init.xavier_normal_(self.directions)


    def forward(self, neighbor_index,vertices,feature_map=None):

        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        kernel_direction_norm = F.normalize(self.directions, dim= -1)



        bin=5
        interval=self.Rs.shape[0]//5
        r_values=[]
        r_indexes=[]
        for i in range(bin):
            Rs=self.Rs[i*interval:(i+1)*interval]
            kernel_direction_norm_ro=torch.einsum('dkj,rij->dkir',kernel_direction_norm,Rs).detach()

            theta=torch.einsum('dkjr,bpnj->bdkpnr',kernel_direction_norm_ro,neighbor_direction_norm)
            # theta=self.relu(theta)
            if not self.is_surface:
                W_ro=torch.einsum('dckj,rij->dckir',self.W,Rs)
                feature_out=torch.einsum('dckir,bcpi->bdkpr',W_ro,feature_map).detach()
                new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,None,:,None].expand(-1,self.dim_out,self.kernel_size,-1,interval)
                kernel_activation=torch.gather(feature_out,3,new_neighbor_index).reshape(bs,self.dim_out,self.kernel_size,vertice_num,neighbor_num,interval)

                kernel_activation=kernel_activation*theta

            else:
                kernel_activation=theta
            kernel_activation_weight=torch.max(kernel_activation,dim=4)[0]
            kernel_activation=torch.sum(kernel_activation_weight,dim=2)
            kernel_activation_r_value,kernel_activation_r_index=torch.max(kernel_activation,dim=-1)
            r_values.append(kernel_activation_r_value)
            r_indexes.append(kernel_activation_r_index+i*interval)
        r_values=torch.stack(r_values,dim=-1)
        r_indexes=torch.stack(r_indexes,dim=-1)
        r_indexes_indexes=torch.max(r_values,dim=-1,keepdim=True)[1]
        r_indexes=torch.gather(r_indexes,-1,r_indexes_indexes)
        r_indexes=r_indexes[:,:,:,:,None,None].expand(-1,-1,-1,-1,3,3)
        R_num=self.Rs.shape[0]
        match_R=self.Rs[None,None,None,:,:,:].expand(bs,self.dim_out,vertice_num,-1,-1,-1)
        match_R=torch.gather(match_R,3,r_indexes).squeeze(3)
        kernel_direction_norm_ro=torch.einsum('dkj,bdpij->bdkpi',kernel_direction_norm,match_R)
        theta=torch.einsum('bdkpj,bpnj->bdkpn',kernel_direction_norm_ro,neighbor_direction_norm)
        theta=self.relu(theta)
        if not self.is_surface:
            W_ro=torch.einsum('dckj,bdpij->bdckpi',self.W,match_R)
            feature_out=torch.einsum('bdckpi,bcpi->bdkp',W_ro,feature_map)
            new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,None,:].expand(-1,self.dim_out,self.kernel_size,-1)
            kernel_activation=torch.gather(feature_out,3,new_neighbor_index).reshape(bs,self.dim_out,self.kernel_size,
                                                                                     vertice_num,neighbor_num)
            kernel_activation=kernel_activation*theta
        else:
            kernel_activation=theta
        kernel_activation_weight=torch.max(kernel_activation,dim=4)[0]
        kernel_activation=torch.einsum('bdkp,bdkpi->bdpi',kernel_activation_weight,kernel_direction_norm_ro)

        if not self.is_surface:
            center_out=torch.einsum('dc,bcpi->bdpi',self.W_center,feature_map)
            fuse=kernel_activation+center_out
        else:
            fuse=kernel_activation


        return fuse

class equi_conv4(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.is_surface=is_surface

        if FLAGS.rotation_path is not '':
            rotation_dict=torch.load(FLAGS.rotation_path)
            symR=rotation_dict['symR']
            vs_=rotation_dict['vs']
            verticeR=rotation_dict['verticeR']
        else:
            symR=get_anchorsV()[5:10]
            symR=torch.from_numpy(symR).double()
            verticeR=get_anchorsV12()
            verticeR=torch.from_numpy(verticeR).double()
            vs_=torch.from_numpy(vs).double()
            rotation_dict={}
            rotation_dict['vs']=vs_
            rotation_dict['symR']=symR
            rotation_dict['verticeR']=verticeR
            torch.save(rotation_dict, FLAGS.rotation_path)

        self.register_buffer('vs',vs_)
        self.register_buffer('symR',symR)
        self.register_buffer('verticeR',verticeR)
        if not self.is_surface:
            self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,9))
            self.bias=nn.Parameter(torch.FloatTensor(self.dim_out,5))
            # self.w_qs = nn.Linear(self.dim_out, self.dim_out, bias=False)
            # self.w_ks = nn.Linear(self.dim_out, self.dim_out, bias=False)
            # self.w_vs = nn.Linear(self.dim_out,self.dim_out, bias=False)
        self.directions=nn.Parameter(torch.FloatTensor(self.dim_out,2,3))
        if FLAGS.tovec_version=='v2':
            self.tovec=tovec_2(self.dim_in)
        elif FLAGS.tovec_version=='v1':
            self.tovec=tovec(self.dim_in)
        elif FLAGS.tovec_version=='v3':
            self.tovec=tovec_3(self.dim_in)
        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.dim_out* (12+1))
        self.directions.data.uniform_(-stdv, stdv)
        if not self.is_surface:
            self.W.data.uniform_(-stdv, stdv)
            self.bias.data.uniform_(-stdv, stdv)
        # if not self.is_surface:
        #     nn.init.xavier_normal_(self.W)
        #     nn.init.xavier_normal_(self.bias)
        # nn.init.xavier_normal_(self.directions)

    def forward(self, neighbor_index,vertices,feature_map=None):
        if not self.is_surface:
            # raise ValueError("NOT SURFACE")
            feature_map=self.tovec(feature_map,self.vs)

        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        kernel_direction_norm = F.normalize(self.directions, dim= -1)
        # sym_kernel_direction_norm=torch.einsum('dsj,vij->dsvi',kernel_direction_norm.double(),self.symR)\
        #     .reshape(self.dim_out,-1,3).float()
        sym_kernel_direction_norm=torch.matmul(kernel_direction_norm[:,:,None,None,:],self.symR.float().permute(0,2,1)).reshape(self.dim_out,-1,3)
        top=torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device)[None,None,:].expand(self.dim_out,-1,-1)
        bottom=torch.tensor([0,0,-1]).float().to(sym_kernel_direction_norm.device)[None,None,:].expand(self.dim_out,-1,-1)
        kernel_direction_norm=torch.cat([top,bottom,sym_kernel_direction_norm],dim=1)
        # show_open3d(kernel_direction_norm[0].detach().cpu().numpy(),kernel_direction_norm[1].detach().cpu().numpy())
        # kernel_direction_norm_ro=torch.einsum('dkj,rij->dkir',kernel_direction_norm.double(),self.verticeR).float()
        kernel_direction_norm_ro=torch.matmul(kernel_direction_norm[:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2).permute(0,1,3,2)
        # print(kernel_direction_norm_ro[4,:,:,4])
        theta_ro=torch.einsum('dkjr,bpnj->bdkpnr',kernel_direction_norm_ro.contiguous(),neighbor_direction_norm.contiguous())

        theta_ro=self.relu(theta_ro)


        if not self.is_surface:
            # raise ValueError("NOT SURFACE")
            W_top=self.W[:,:,None,0:1]*(torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device))
            W_bottom=self.W[:,:,None,1:2]*(torch.tensor([0,0,-1]).float().to(sym_kernel_direction_norm.device))
            W_center=self.W[:,:,None,8:]*(torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device))
            # W_sym=torch.einsum('dcsj,vij->dcsvi',self.W[:,:,2:8].reshape(self.dim_out,self.dim_in,2,3).double(),self.symR).reshape(self.dim_out,self.dim_in,-1,3).float()
            W_sym=torch.matmul(self.W[:,:,2:8].reshape(self.dim_out,self.dim_in,2,3)[:,:,:,None,None,:],
                               self.symR.float().permute(0,2,1)).reshape(self.dim_out,self.dim_in,-1,3)
            W=torch.cat([W_top,W_bottom,W_sym,W_center],dim=-2)
            # center_out=torch.einsum('bcpr,dc->bdpr',feature_map,self.W[:,:,8])

            bias=torch.cat([self.bias[:,0:1],self.bias[:,1:2],self.bias[:,2:3].expand(-1,5),
                            self.bias[:,3:4].expand(-1,5),self.bias[:,4:]],dim=-1)


            # W_ro=torch.einsum('dckj,rij->dckir',W.double(),self.verticeR).float()
            W_ro=torch.matmul(W[:,:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2)\
                .permute(0,1,2,4,3)
            feature_out=torch.einsum('dckir,bcpi->bdkpr',W_ro,feature_map)+bias[None,:,:,None,None]
            center_out=feature_out[:,:,12,:,:]
            kernel_activation=feature_out[:,:,:12,:,:]
            new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,None,:,None].expand(-1,self.dim_out,12,-1,12)
            kernel_activation=torch.gather(kernel_activation,3,new_neighbor_index).reshape(bs,self.dim_out,12,
                                                                                     vertice_num,neighbor_num,12)
            # bias=bias[None,:,:,None,None,None].expand(bs,-1,-1,vertice_num,neighbor_num,12)
            # kernel_activation=kernel_activation+bias
            kernel_activation=kernel_activation*theta_ro
        else:
            kernel_activation=theta_ro
        kernel_activation=torch.max(kernel_activation,dim=4)[0]
        kernel_activation=torch.sum(kernel_activation,dim=2)
        # kernel_activation=F.softmax(kernel_activation/(0.1),dim=-1)






        if not self.is_surface:
            kernel_activation=kernel_activation+center_out
            # x=kernel_activation.permute(0,2,3,1)
            # q=self.w_qs(x)
            # k=self.w_ks(x)
            # v=self.w_vs(x)
            # atten=F.softmax(torch.einsum('bplc,bpkc->bplk',q,k),dim=-1)
            # out=torch.einsum('bplk,bpkc->bplc',atten,v)
            # out=out.permute(0,3,1,2)
            # kernel_activation=out

        # ro=torch.einsum('bdpr,rj->bdpj',kernel_activation,self.vs)

        return kernel_activation,feature_map



class equi_conv5(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.is_surface=is_surface

        if FLAGS.rotation_path is not '':
            rotation_dict=torch.load(FLAGS.rotation_path)
            symR=rotation_dict['symR']
            vs_=rotation_dict['vs']
            verticeR=rotation_dict['verticeR']
        else:
            symR=get_anchorsV()[5:10]
            symR=torch.from_numpy(symR).double()
            verticeR=get_anchorsV12()
            verticeR=torch.from_numpy(verticeR).double()
            vs_=torch.from_numpy(vs).double()
            rotation_dict={}
            rotation_dict['vs']=vs_
            rotation_dict['symR']=symR
            rotation_dict['verticeR']=verticeR
            torch.save(rotation_dict, FLAGS.rotation_path)

        self.register_buffer('vs',vs_)
        self.register_buffer('symR',symR)
        self.register_buffer('verticeR',verticeR)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,9))
        self.head_num=8
        self.W_dir = nn.Parameter(torch.FloatTensor(self.dim_in*self.head_num,8))
        self.bias=nn.Parameter(torch.FloatTensor(self.dim_out,5))
        self.bias_dir=nn.Parameter(torch.FloatTensor(4))
        self.directions=nn.Parameter(torch.FloatTensor(self.dim_in,2,3))
        if FLAGS.tovec_version=='v2':
            self.tovec=tovec_2(self.dim_in)
        elif FLAGS.tovec_version=='v1':
            self.tovec=tovec(self.dim_in)
        elif FLAGS.tovec_version=='v3':
            self.tovec=tovec_3(self.dim_in)
        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.dim_out* (12+1))
        self.directions.data.uniform_(-stdv, stdv)
        self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.bias_dir.data.uniform_(-stdv, stdv)
        self.W_dir.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index,vertices,feature_map=None):
        if not self.is_surface:
            feature_map=self.tovec(feature_map,self.vs)

        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        kernel_direction_norm = F.normalize(self.directions, dim= -1)

        W_top=self.W[:,:,None,0:1]*(torch.tensor([0,0,1]).float().to(kernel_direction_norm.device))
        W_bottom=self.W[:,:,None,1:2]*(torch.tensor([0,0,-1]).float().to(kernel_direction_norm.device))
        W_center=self.W[:,:,None,8:]*(torch.tensor([0,0,1]).float().to(kernel_direction_norm.device))
        W_sym=torch.matmul(self.W[:,:,2:8].reshape(self.dim_out,self.dim_in,2,3)[:,:,:,None,None,:],
                           self.symR.float().permute(0,2,1)).reshape(self.dim_out,self.dim_in,-1,3)
        W=torch.cat([W_top,W_bottom,W_sym,W_center],dim=-2)
        W_ro=torch.matmul(W[:,:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2) \
            .permute(0,1,2,4,3)

        bias=torch.cat([self.bias[:,0:1],self.bias[:,1:2],self.bias[:,2:3].expand(-1,5),
                        self.bias[:,3:4].expand(-1,5),self.bias[:,4:]],dim=-1)

        if self.is_surface:
            sym_kernel_direction_norm=torch.matmul(kernel_direction_norm[:,:,None,None,:],self.symR.float().permute(0,2,1)).reshape(self.dim_in,-1,3)
            top=torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device)[None,None,:].expand(self.dim_in,-1,-1)
            bottom=torch.tensor([0,0,-1]).float().to(sym_kernel_direction_norm.device)[None,None,:].expand(self.dim_in,-1,-1)
            kernel_direction_norm=torch.cat([top,bottom,sym_kernel_direction_norm],dim=1)
            kernel_direction_norm_ro=torch.matmul(kernel_direction_norm[:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2).permute(0,1,3,2)
            new_feature_map_act=torch.einsum('ckir,bcpni->bkpnr',kernel_direction_norm_ro,neighbor_direction_norm[:,None,:,:,:])
            new_feature_map_act=self.relu(new_feature_map_act)
            new_feature_map=torch.einsum('bkpnr,bcpni->bckpir',new_feature_map_act,neighbor_direction_norm[:,None,:,:,:])

            feature_out=torch.einsum('dckir,bckpir->bdkpr',W_ro[:,:,:12,:,:],new_feature_map)+bias[None,:,:12,None,None]
            kernel_activation=feature_out.sum(2)
            return kernel_activation,None

        else:

            bias_dir=torch.cat([self.bias_dir[0:1],self.bias_dir[1:2],self.bias_dir[2:3].expand(5),
                            self.bias_dir[3:4].expand(5)],dim=-1)

            W_dir_top=self.W_dir[:,None,0:1]*(torch.tensor([0,0,1]).float().to(kernel_direction_norm.device))
            W_dir_bottom=self.W_dir[:,None,1:2]*(torch.tensor([0,0,-1]).float().to(kernel_direction_norm.device))
            W_dir_sym=torch.matmul(self.W_dir[:,2:8].reshape(self.dim_in*self.head_num,2,3)[:,:,None,None,:],
                               self.symR.float().permute(0,2,1)).reshape(self.dim_in*self.head_num,-1,3)
            W_dir=torch.cat([W_dir_top,W_dir_bottom,W_dir_sym],dim=-2)

            W_dir_ro=torch.matmul(W_dir[:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2) \
                .permute(0,1,3,2)



            new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,self.dim_in,-1,3)
            new_feature_map=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,self.dim_in,
                                                                                 vertice_num,neighbor_num,3)

            new_feature_map_cross=torch.cross(new_feature_map,neighbor_direction_norm[:,None,:,:,:],dim=-1)
            W_dir_ro=einops.rearrange(W_dir_ro,'(h c) k i r->h c k i r',h=self.head_num,c=self.dim_in)
            d_per_head=self.dim_in//self.head_num
            new_feature_map_act=torch.einsum('hckir,bcpni->hbkpnr',W_dir_ro,new_feature_map_cross)/d_per_head ** 0.5

            new_feature_map_act=torch.softmax(new_feature_map_act,dim=-2)

            new_feature_map_cross=einops.rearrange(new_feature_map_cross,'b (h c) p n i -> b h c p n i',h=self.head_num)

            new_feature_map=torch.einsum('hbkpnr,bhcpni->bhckpir',new_feature_map_act,new_feature_map_cross)
            new_feature_map=einops.rearrange(new_feature_map,'b h c k p i r -> b (h c) k p i r')
            feature_out=torch.einsum('dckir,bckpir->bdkpr',W_ro[:,:,:12,:,:],new_feature_map)+bias[None,:,:12,None,None]
            center_out=torch.einsum('dcir,bcpi->bdpr',W_ro[:,:,12,:,:],feature_map)+bias[None,:,12,None,None]
            kernel_activation=feature_out.sum(2)+center_out

            return kernel_activation,feature_map






class equi_conv6(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.is_surface=is_surface

        if FLAGS.rotation_path is not '':
            rotation_dict=torch.load(FLAGS.rotation_path)
            symR=rotation_dict['symR']
            vs_=rotation_dict['vs']
            verticeR=rotation_dict['verticeR']
        else:
            symR=get_anchorsV()[5:10]
            symR=torch.from_numpy(symR).double()
            verticeR=get_anchorsV12()
            verticeR=torch.from_numpy(verticeR).double()
            vs_=torch.from_numpy(vs).double()
            rotation_dict={}
            rotation_dict['vs']=vs_
            rotation_dict['symR']=symR
            rotation_dict['verticeR']=verticeR
            torch.save(rotation_dict, FLAGS.rotation_path)

        self.register_buffer('vs',vs_)
        self.register_buffer('symR',symR)
        self.register_buffer('verticeR',verticeR)
        self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,9))
        self.W_dir = nn.Parameter(torch.FloatTensor(self.dim_in,8))
        self.bias=nn.Parameter(torch.FloatTensor(self.dim_out,5))
        self.directions=nn.Parameter(torch.FloatTensor(self.dim_out,2,3))
        if FLAGS.tovec_version=='v2':
            self.tovec=tovec_2(self.dim_in)
        elif FLAGS.tovec_version=='v1':
            self.tovec=tovec(self.dim_in)
        elif FLAGS.tovec_version=='v3':
            self.tovec=tovec_3(self.dim_in)
        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.dim_out* (12+1))
        self.directions.data.uniform_(-stdv, stdv)
        self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.W_dir.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index,vertices,feature_map=None):
        if not self.is_surface:
            feature_map=self.tovec(feature_map,self.vs)
        else:
            feature_map=torch.ones_like(vertices)[:,None,:,:]
            feature_map[:,:,:,1:]=0

        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        kernel_direction_norm = F.normalize(self.directions, dim= -1)
        # sym_kernel_direction_norm=torch.einsum('dsj,vij->dsvi',kernel_direction_norm.double(),self.symR)\
        #     .reshape(self.dim_out,-1,3).float()
        sym_kernel_direction_norm=torch.matmul(kernel_direction_norm[:,:,None,None,:],self.symR.float().permute(0,2,1)).reshape(self.dim_out,-1,3)
        top=torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device)[None,None,:].expand(self.dim_out,-1,-1)
        bottom=torch.tensor([0,0,-1]).float().to(sym_kernel_direction_norm.device)[None,None,:].expand(self.dim_out,-1,-1)
        kernel_direction_norm=torch.cat([top,bottom,sym_kernel_direction_norm],dim=1)
        # show_open3d(kernel_direction_norm[0].detach().cpu().numpy(),kernel_direction_norm[1].detach().cpu().numpy())
        # kernel_direction_norm_ro=torch.einsum('dkj,rij->dkir',kernel_direction_norm.double(),self.verticeR).float()
        kernel_direction_norm_ro=torch.matmul(kernel_direction_norm[:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2).permute(0,1,3,2)




        W_top=self.W[:,:,None,0:1]*(torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device))
        W_bottom=self.W[:,:,None,1:2]*(torch.tensor([0,0,-1]).float().to(sym_kernel_direction_norm.device))
        W_center=self.W[:,:,None,8:]*(torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device))
        # W_sym=torch.einsum('dcsj,vij->dcsvi',self.W[:,:,2:8].reshape(self.dim_out,self.dim_in,2,3).double(),self.symR).reshape(self.dim_out,self.dim_in,-1,3).float()
        W_sym=torch.matmul(self.W[:,:,2:8].reshape(self.dim_out,self.dim_in,2,3)[:,:,:,None,None,:],
                           self.symR.float().permute(0,2,1)).reshape(self.dim_out,self.dim_in,-1,3)
        W=torch.cat([W_top,W_bottom,W_sym,W_center],dim=-2)
        # center_out=torch.einsum('bcpr,dc->bdpr',feature_map,self.W[:,:,8])

        bias=torch.cat([self.bias[:,0:1],self.bias[:,1:2],self.bias[:,2:3].expand(-1,5),
                        self.bias[:,3:4].expand(-1,5),self.bias[:,4:]],dim=-1)



        W_ro=torch.matmul(W[:,:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2) \
            .permute(0,1,2,4,3)

        W_dir_top=self.W_dir[:,None,0:1]*(torch.tensor([0,0,1]).float().to(sym_kernel_direction_norm.device))
        W_dir_bottom=self.W_dir[:,None,1:2]*(torch.tensor([0,0,-1]).float().to(sym_kernel_direction_norm.device))
        W_dir_sym=torch.matmul(self.W_dir[:,2:8].reshape(self.dim_in,2,3)[:,:,None,None,:],
                               self.symR.float().permute(0,2,1)).reshape(self.dim_in,-1,3)
        W_dir=torch.cat([W_dir_top,W_dir_bottom,W_dir_sym],dim=-2)

        W_dir_ro=torch.matmul(W_dir[:,:,None,None,:],self.verticeR.float().permute(0,2,1)).squeeze(-2) \
            .permute(0,1,3,2)

        W_dir_ro=F.normalize(W_dir_ro,dim=-2)



        new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,self.dim_in,-1,3)
        new_feature_map=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,self.dim_in,
                                                                               vertice_num,neighbor_num,3)

        new_feature_map_norm=F.normalize(new_feature_map,dim=-1)

        new_feature_map=torch.cross(new_feature_map,neighbor_direction_norm[:,None,:,:,:],dim=-1)
        new_feature_map_norm=torch.cross(new_feature_map_norm,neighbor_direction_norm[:,None,:,:,:],dim=-1)

        new_feature_map_act=torch.einsum('ckir,bcpni->bkpnr',W_dir_ro,new_feature_map_norm)
        new_feature_map_act=self.relu(new_feature_map_act)

        new_feature_map=torch.einsum('bkpnr,bcpni->bckpir',new_feature_map_act,new_feature_map)
        feature_out=torch.einsum('dckir,bckpir->bdkpr',W_ro[:,:,:12,:,:],new_feature_map)+bias[None,:,:12,None,None]
        center_out=torch.einsum('dcir,bcpi->bdpr',W_ro[:,:,12,:,:],feature_map)+bias[None,:,12,None,None]
        kernel_activation=feature_out.sum(2)+center_out




        return kernel_activation,feature_map




class equi_conv7(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.is_surface=is_surface


        rotation_dict=torch.load(FLAGS.rotation_path)
        symR=rotation_dict['symR']
        vs_=rotation_dict['vs']
        verticeR=rotation_dict['verticeR']

        self.register_buffer('vs',vs_)
        self.register_buffer('symR',symR)
        self.register_buffer('verticeR',verticeR)

        faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
               (1,2,4),(2,7,8),(3,7,9),(3,5,11),
               (4,5,6),(2,4,10),(2,8,10),(7,8,9),
               (3,9,11),(5,6,11),(4,6,10),(0,8,10),
               (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

        face_normal=vs[faces,:].sum(1)
        face_normal=torch.from_numpy(face_normal).float()
        face_normal=F.normalize(face_normal,dim=-1)
        self.register_buffer('face_normal',face_normal)


        face_to_cube=[(1,4,0,2,3),(2,0,1,4,3),(3,1,0,4,2),(4,2,0,3,1),
                      (0,3,1,2,4),(3,2,0,4,1),(4,3,0,2,1),(0,4,1,2,3),
                      (1,0,2,4,3),(2,1,0,4,3),(4,0,1,3,2),(0,1,2,3,4),
                      (1,2,0,3,4),(2,3,0,1,4),(3,4,0,1,2),(1,3,0,2,4),
                      (0,2,1,3,4),(4,1,0,3,2),(3,0,1,4,2),(2,4,0,1,3)]
        face_to_cube=torch.from_numpy(np.array(face_to_cube))
        self.register_buffer('face_to_cube',face_to_cube)

        roll=[(0,1,2,3,4),(0,1,3,4,2),(0,1,4,2,3)]
        roll=torch.from_numpy(np.array(roll))
        self.register_buffer('roll',roll)

        cube_to_face_normal=[(1,4,7,8,11,10,16,18),
                        (0,2,11,12,8,9,15,17),
                         (1,3,12,13,5,9,19,16),
                         (4,2,5,6,14,13,15,18),
                         (0,3,6,7,10,14,19,17),]
        cube_to_face_normal=torch.from_numpy(np.array(cube_to_face_normal))
        cubes=face_normal[cube_to_face_normal]
        self.register_buffer('cubes',cubes)

        cube_to_face_normal_orth=[(1,11,7,4),
                                  (0,11,2,9),
                                  (1,12,5,3),
                                  (4,2,5,14),
                                  (0,10,6,3),]
        cube_to_face_normal_orth=torch.from_numpy(np.array(cube_to_face_normal_orth))
        cubs_orth=face_normal[cube_to_face_normal_orth]
        new_cubs_orth=torch.zeros_like(cubs_orth)[:,:3]
        new_cubs_orth[:,0]=cubs_orth[:,1]-cubs_orth[:,0]
        new_cubs_orth[:,1]=cubs_orth[:,2]-cubs_orth[:,0]
        new_cubs_orth[:,2]=cubs_orth[:,3]-cubs_orth[:,0]
        cubs_orth=torch.cat([new_cubs_orth,-new_cubs_orth],dim=1)
        cubs_orth=F.normalize(cubs_orth,dim=-1)
        self.register_buffer('cubs_orth',cubs_orth)

        cube_to_face=[(1,4,7,8,11,10,16,18),
                         (0,2,11,12,8,9,15,17),
                         (1,3,12,13,5,9,19,16),
                         (4,2,5,6,14,13,15,18),
                         (0,3,6,7,10,14,19,17),]
        cube_to_face=torch.from_numpy(np.array(cube_to_face))
        self.register_buffer('cube_to_face',cube_to_face)

        self.W_dim = nn.Parameter(torch.FloatTensor(self.dim_in+1,self.dim_in))

        self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,19))
        self.head_num=8
        if self.is_surface:
            self.W_dir = nn.Parameter(torch.FloatTensor(self.dim_in,19))
        else:
            self.W_dir = nn.Parameter(torch.FloatTensor(self.dim_in*self.head_num,19))
        self.bias=nn.Parameter(torch.FloatTensor(self.dim_out,5))
        self.bias_dir=nn.Parameter(torch.FloatTensor(4))
        self.directions=nn.Parameter(torch.FloatTensor(self.dim_in,2,3))

        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.dim_out* (5+1))
        self.directions.data.uniform_(-stdv, stdv)
        self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.bias_dir.data.uniform_(-stdv, stdv)
        self.W_dir.data.uniform_(-stdv, stdv)
        self.W_dim.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index,vertices,feature_map=None):

        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)


        W_top=torch.cat([self.W[:,:,None,0:2],self.W[:,:,None,2:3].repeat(1,1,1,3)],dim=-1)
        W_bottom=torch.cat([self.W[:,:,None,3:5],self.W[:,:,None,5:6].repeat(1,1,1,3)],dim=-1)
        W_center=torch.cat([self.W[:,:,None,6:8],self.W[:,:,None,8:9].repeat(1,1,1,3)],dim=-1)
        W_sym=self.W[:,:,9:19].reshape(self.dim_out,self.dim_in,2,5)
        W_sym=W_sym[:,:,:,self.roll].reshape(self.dim_out,self.dim_in,6,5)
        W=torch.cat([W_top,W_bottom,W_sym,W_center],dim=-2)[:,:,:,None,:].repeat(1,1,1,20,1)
        ro=self.face_to_cube[None,None,None,:,:].repeat(self.dim_out,self.dim_in,9,1,1)
        W_ro=torch.zeros_like(W)
        W_ro.scatter_(-1,ro,W)
        if self.is_surface:
            W_dir_top=torch.cat([self.W_dir[:,None,0:2],self.W_dir[:,None,2:3].repeat(1,1,3)],dim=-1)
            W_dir_bottom=torch.cat([self.W_dir[:,None,3:5],self.W_dir[:,None,5:6].repeat(1,1,3)],dim=-1)
            W_dir_center=torch.cat([self.W_dir[:,None,6:8],self.W_dir[:,None,8:9].repeat(1,1,3)],dim=-1)
            W_dir_sym=self.W_dir[:,9:19].reshape(self.dim_in,2,5)
            W_dir_sym=W_dir_sym[:,:,self.roll].reshape(self.dim_in,6,5)
            W_dir=torch.cat([W_dir_top,W_dir_bottom,W_dir_sym,W_dir_center],dim=-2)[:,:,None,:].repeat(1,1,20,1)
            ro=self.face_to_cube[None,None,:,:].repeat(self.dim_in,9,1,1)
            W_dir_ro=torch.zeros_like(W_dir)
            W_dir_ro.scatter_(-1,ro,W_dir)
        else:
            W_dir_top=torch.cat([self.W_dir[:,None,0:2],self.W_dir[:,None,2:3].repeat(1,1,3)],dim=-1)
            W_dir_bottom=torch.cat([self.W_dir[:,None,3:5],self.W_dir[:,None,5:6].repeat(1,1,3)],dim=-1)
            W_dir_center=torch.cat([self.W_dir[:,None,6:8],self.W_dir[:,None,8:9].repeat(1,1,3)],dim=-1)
            W_dir_sym=self.W_dir[:,9:19].reshape(self.dim_in*self.head_num,2,5)
            W_dir_sym=W_dir_sym[:,:,self.roll].reshape(self.dim_in*self.head_num,6,5)
            W_dir=torch.cat([W_dir_top,W_dir_bottom,W_dir_sym,W_dir_center],dim=-2)[:,:,None,:].repeat(1,1,20,1)
            ro=self.face_to_cube[None,None,:,:].repeat(self.dim_in*self.head_num,9,1,1)
            W_dir_ro=torch.zeros_like(W_dir)
            W_dir_ro.scatter_(-1,ro,W_dir)



        bias=torch.cat([self.bias[:,0:1],self.bias[:,1:2],self.bias[:,2:3].expand(-1,3),
                        self.bias[:,3:4].expand(-1,3),self.bias[:,4:]],dim=-1)



        if self.is_surface:


            # de_neighbor_direction_norm=torch.einsum('c v i , b q n i -> b q n c v',self.cubes,neighbor_direction_norm)
            # de_neighbor_direction_norm=torch.max(de_neighbor_direction_norm,dim=-1)[0]


            de_neighbor_direction_norm=torch.einsum('f i , b q n i -> b q n f',self.face_normal,neighbor_direction_norm)
            de_neighbor_direction_norm=torch.tanh(de_neighbor_direction_norm)
            color=torch.zeros_like(de_neighbor_direction_norm)[:,:,:,:1].repeat(1,1,1,5)
            b,q,n,_=de_neighbor_direction_norm.shape
            color.scatter_add_(-1,self.face_to_cube[:,0][None,None,None,:].repeat(b,q,n,1),de_neighbor_direction_norm)
            de_neighbor_direction_norm=color
            new_feature_map_act=torch.einsum('ckri,bcpni->bkpnr',W_dir_ro,de_neighbor_direction_norm[:,None,:,:,:])
            new_feature_map_act=self.relu(new_feature_map_act)
            new_feature_map=torch.einsum('bkpnr,bcpni->bckpir',new_feature_map_act,de_neighbor_direction_norm[:,None,:,:,:])

            feature_out=torch.einsum('dckri,bckpir->bdkpr',W_ro[:,:,:9,:,:],new_feature_map)
            kernel_activation=feature_out.sum(2)

            kernel_color=torch.zeros_like(kernel_activation)[:,:,:,:1].repeat(1,1,1,5)
            b,d,p,_=kernel_activation.shape
            kernel_color.scatter_add_(-1,self.face_to_cube[:,0][None,None,None,:].repeat(b,d,p,1),kernel_activation)

            # kernel_activation=kernel_activation[:,:,:,self.cube_to_face].max(-1)[0]
            kernel_activation=kernel_color
            return kernel_activation,None

        else:




            new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,self.dim_in,-1,5)
            new_feature_map=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,self.dim_in,
                                                                                   vertice_num,neighbor_num,5)

            # de_neighbor_direction_norm=torch.einsum('c v i , b q n i -> b q n c v',self.cubes,neighbor_direction_norm)
            # de_neighbor_direction_norm=torch.max(de_neighbor_direction_norm,dim=-1)[0]

            de_neighbor_direction_norm=torch.einsum('f i , b q n i -> b q n f',self.face_normal,neighbor_direction_norm)
            de_neighbor_direction_norm=torch.tanh(de_neighbor_direction_norm)
            color=torch.zeros_like(de_neighbor_direction_norm)[:,:,:,:1].repeat(1,1,1,5)
            b,q,n,_=de_neighbor_direction_norm.shape
            color.scatter_add_(-1,self.face_to_cube[:,0][None,None,None,:].repeat(b,q,n,1),de_neighbor_direction_norm)
            de_neighbor_direction_norm=color



            new_feature_map_cross=torch.cat([new_feature_map,de_neighbor_direction_norm[:,None,:,:,:]],dim=1)
            new_feature_map_cross=torch.einsum('co, b c p n i->bopni',self.W_dim,new_feature_map_cross)
            W_dir_ro=einops.rearrange(W_dir_ro,'(h c) k r i->h c k i r',h=self.head_num,c=self.dim_in)
            d_per_head=self.dim_in//self.head_num
            new_feature_map_act=torch.einsum('hckir,bcpni->hbkpnr',W_dir_ro[:,:,:8,:,:],new_feature_map_cross)/d_per_head ** 0.5

            new_feature_map_act=torch.softmax(new_feature_map_act,dim=-2)

            new_feature_map_cross=einops.rearrange(new_feature_map_cross,'b (h c) p n i -> b h c p n i',h=self.head_num)

            new_feature_map=torch.einsum('hbkpnr,bhcpni->bhckpir',new_feature_map_act,new_feature_map_cross)
            new_feature_map=einops.rearrange(new_feature_map,'b h c k p i r -> b (h c) k p i r')
            feature_out=torch.einsum('dckri,bckpir->bdkpr',W_ro[:,:,:8,:,:],new_feature_map)
            center_out=torch.einsum('dcri,bcpi->bdpr',W_ro[:,:,8,:,:],feature_map)
            kernel_activation=feature_out.sum(2)+center_out
            # kernel_activation=kernel_activation[:,:,:,self.cube_to_face].max(-1)[0]

            kernel_color=torch.zeros_like(kernel_activation)[:,:,:,:1].repeat(1,1,1,5)
            b,d,p,_=kernel_activation.shape
            kernel_color.scatter_add_(-1,self.face_to_cube[:,0][None,None,None,:].repeat(b,d,p,1),kernel_activation)

            # kernel_activation=kernel_activation[:,:,:,self.cube_to_face].max(-1)[0]
            kernel_activation=kernel_color
            return kernel_activation,feature_map




class equi_conv8(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=7, is_surface=False) -> None:
        """Linear layer projecting features aggregated at the kernel points to the centers.
        Using the exact derivation
        [b, c1, k, p, a] -> [b, c2, p, a]"""
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.kernel_size = kernel_size
        self.is_surface=is_surface


        rotation_dict=torch.load(FLAGS.rotation_path)
        symR=rotation_dict['symR']
        vs_=rotation_dict['vs']
        verticeR=rotation_dict['verticeR']

        self.register_buffer('vs',vs_.float())
        self.register_buffer('symR',symR)
        self.register_buffer('verticeR',verticeR)

        faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
               (1,2,4),(2,7,8),(3,7,9),(3,5,11),
               (4,5,6),(2,4,10),(2,8,10),(7,8,9),
               (3,9,11),(5,6,11),(4,6,10),(0,8,10),
               (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

        face_normal=vs[faces,:].sum(1)
        face_normal=torch.from_numpy(face_normal).float()
        face_normal=F.normalize(face_normal,dim=-1)
        self.register_buffer('face_normal',face_normal)





        v2colors=[0,0,2,5,3,4,1,1,4,3,5,2]
        color2v=[(0,1),(6,7),(2,11),(4,9),(5,8),(3,10)]
        color_com=[(0,1,2,3,4,5),
                   (0,5,4,3,2,1),
                   (2,5,4,1,0,3),
                   (5,0,1,3,2,4),
                   (3,5,2,0,4,1),
                   (4,0,5,2,1,3),
                   (1,3,4,2,0,5),
                   (1,0,2,4,3,5),
                   (4,1,2,5,0,3),
                   (3,1,4,0,2,5),
                   (5,1,0,4,2,3),
                   (2,4,5,3,0,1)]
        color_com=torch.from_numpy(np.array(color_com))
        v2colors=torch.from_numpy(np.array(v2colors))
        color2v=torch.from_numpy(np.array(color2v))

        roll=np.array([(0,1,2,3,4,5),(0,2,3,4,5,1),(0,3,4,5,1,2),(0,4,5,1,2,3),(0,5,1,2,3,4)])
        roll=torch.from_numpy(roll)

        self.register_buffer('v2colors',v2colors)
        self.register_buffer('color2v',color2v)
        self.register_buffer('color_com',color_com)
        self.register_buffer('roll',roll)


        self.W_dim = nn.Parameter(torch.FloatTensor(self.dim_in+1,self.dim_in))

        self.W = nn.Parameter(torch.FloatTensor(self.dim_out,self.dim_in,18))
        self.head_num=8
        if self.is_surface:
            self.W_dir = nn.Parameter(torch.FloatTensor(self.dim_in,18))
        else:
            self.W_dir = nn.Parameter(torch.FloatTensor(self.dim_in*self.head_num,18))
        self.bias=nn.Parameter(torch.FloatTensor(self.dim_out,5))
        self.bias_dir=nn.Parameter(torch.FloatTensor(4))
        self.directions=nn.Parameter(torch.FloatTensor(self.dim_in,2,3))

        self.relu = nn.ReLU(inplace= True)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.dim_out* (6))
        self.directions.data.uniform_(-stdv, stdv)
        self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.bias_dir.data.uniform_(-stdv, stdv)
        self.W_dir.data.uniform_(-stdv, stdv)
        self.W_dim.data.uniform_(-stdv, stdv)

    def forward(self, neighbor_index,vertices,feature_map=None):

        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        kernel_direction_norm = F.normalize(self.directions, dim= -1)

        W_top=torch.cat([self.W[:,:,None,0:1],self.W[:,:,None,1:2].repeat(1,1,1,5)],dim=-1)
        W_bottom=torch.cat([self.W[:,:,None,2:3],self.W[:,:,None,3:4].repeat(1,1,1,5)],dim=-1)
        W_center=torch.cat([self.W[:,:,None,4:5],self.W[:,:,None,5:6].repeat(1,1,1,5)],dim=-1)
        W_sym=self.W[:,:,6:18].reshape(self.dim_out,self.dim_in,2,6)
        W_sym=W_sym[:,:,:,self.roll].reshape(self.dim_out,self.dim_in,10,6)
        W=torch.cat([W_top,W_bottom,W_sym,W_center],dim=-2)[:,:,:,None,:].repeat(1,1,1,12,1)
        ro=self.color_com[None,None,None,:,:].repeat(self.dim_out,self.dim_in,13,1,1)
        W_ro=torch.zeros_like(W)
        W_ro.scatter_(-1,ro,W)
        if not self.is_surface:
            W_dir_top=torch.cat([self.W_dir[:,None,0:1],self.W_dir[:,None,1:2].repeat(1,1,5)],dim=-1)
            W_dir_bottom=torch.cat([self.W_dir[:,None,2:3],self.W_dir[:,None,3:4].repeat(1,1,5)],dim=-1)
            W_dir_center=torch.cat([self.W_dir[:,None,4:5],self.W_dir[:,None,5:6].repeat(1,1,5)],dim=-1)
            W_dir_sym=self.W_dir[:,6:18].reshape(self.dim_in*self.head_num,2,6)
            W_dir_sym=W_dir_sym[:,:,self.roll].reshape(self.dim_in*self.head_num,10,6)
            W_dir=torch.cat([W_dir_top,W_dir_bottom,W_dir_sym,W_dir_center],dim=-2)[:,:,None,:].repeat(1,1,12,1)
            ro=self.color_com[None,None,:,:].repeat(self.dim_in*self.head_num,13,1,1)
            W_dir_ro=torch.zeros_like(W_dir)
            W_dir_ro.scatter_(-1,ro,W_dir)
        else:
            W_dir_top=torch.cat([self.W_dir[:,None,0:1],self.W_dir[:,None,1:2].repeat(1,1,5)],dim=-1)
            W_dir_bottom=torch.cat([self.W_dir[:,None,2:3],self.W_dir[:,None,3:4].repeat(1,1,5)],dim=-1)
            W_dir_center=torch.cat([self.W_dir[:,None,4:5],self.W_dir[:,None,5:6].repeat(1,1,5)],dim=-1)
            W_dir_sym=self.W_dir[:,6:18].reshape(self.dim_in,2,6)
            W_dir_sym=W_dir_sym[:,:,self.roll].reshape(self.dim_in,10,6)
            W_dir=torch.cat([W_dir_top,W_dir_bottom,W_dir_sym,W_dir_center],dim=-2)[:,:,None,:].repeat(1,1,12,1)
            ro=self.color_com[None,None,:,:].repeat(self.dim_in,13,1,1)
            W_dir_ro=torch.zeros_like(W_dir)
            W_dir_ro.scatter_(-1,ro,W_dir)




        bias=torch.cat([self.bias[:,0:1],self.bias[:,1:2],self.bias[:,2:3].expand(-1,3),
                        self.bias[:,3:4].expand(-1,3),self.bias[:,4:]],dim=-1)



        if self.is_surface:

            de_neighbor_direction_norm=torch.einsum('v i , b q n i -> b q n v',self.vs,neighbor_direction_norm)
            de_neighbor_direction_norm=de_neighbor_direction_norm[:,:,:,self.color2v]
            de_neighbor_direction_norm=torch.max(de_neighbor_direction_norm,dim=-1)[0]


            new_feature_map_act=torch.einsum('ckri,bcpni->bkpnr',W_dir_ro,de_neighbor_direction_norm[:,None,:,:,:])
            new_feature_map_act=self.relu(new_feature_map_act)
            new_feature_map=torch.einsum('bkpnr,bcpni->bckpir',new_feature_map_act,de_neighbor_direction_norm[:,None,:,:,:])

            feature_out=torch.einsum('dckri,bckpir->bdkpr',W_ro[:,:,:13,:,:],new_feature_map)
            kernel_activation=feature_out.sum(2)

            kernel_activation=kernel_activation[:,:,:,self.color2v].max(-1)[0]

            return kernel_activation,None

        else:


            new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,self.dim_in,-1,6)
            new_feature_map=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,self.dim_in,
                                                                                   vertice_num,neighbor_num,6)

            de_neighbor_direction_norm=torch.einsum('v i , b q n i -> b q n v',self.vs,neighbor_direction_norm)
            de_neighbor_direction_norm=de_neighbor_direction_norm[:,:,:,self.color2v]
            de_neighbor_direction_norm=torch.max(de_neighbor_direction_norm,dim=-1)[0]

            new_feature_map_cross=torch.cat([new_feature_map,de_neighbor_direction_norm[:,None,:,:,:]],dim=1)
            new_feature_map_cross=torch.einsum('co, b c p n i->bopni',self.W_dim,new_feature_map_cross)
            W_dir_ro=einops.rearrange(W_dir_ro,'(h c) k r i->h c k i r',h=self.head_num,c=self.dim_in)
            d_per_head=self.dim_in//self.head_num
            new_feature_map_act=torch.einsum('hckir,bcpni->hbkpnr',W_dir_ro[:,:,:12,:,:],new_feature_map_cross)/d_per_head ** 0.5

            new_feature_map_act=torch.softmax(new_feature_map_act,dim=-2)

            new_feature_map_cross=einops.rearrange(new_feature_map_cross,'b (h c) p n i -> b h c p n i',h=self.head_num)

            new_feature_map=torch.einsum('hbkpnr,bhcpni->bhckpir',new_feature_map_act,new_feature_map_cross)
            new_feature_map=einops.rearrange(new_feature_map,'b h c k p i r -> b (h c) k p i r')
            feature_out=torch.einsum('dckri,bckpir->bdkpr',W_ro[:,:,:12,:,:],new_feature_map)
            center_out=torch.einsum('dcri,bcpi->bdpr',W_ro[:,:,12,:,:],feature_map)
            kernel_activation=feature_out.sum(2)+center_out
            kernel_activation=kernel_activation[:,:,:,self.color2v].max(-1)[0]
            return kernel_activation,feature_map















class equ_pool_layer2(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice _num, channel_num)
        """
        from pytorch3d.ops import sample_farthest_points
        bs, vertice_num, _ = vertices.size()
        r=feature_map.shape[-1]
        dim_out=feature_map.shape[1]
        N_new = vertice_num // self.pooling_rate
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,dim_out,-1,r)
        neighbor_feature=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,dim_out,vertice_num,self.neighbor_num,r)
        pooled_feature = torch.max(neighbor_feature, dim= 3)[0] #(bs, vertice_num, channel_num)

        # with torch.no_grad():
        #     vertices_pool, idx = sample_farthest_points(vertices, K=N_new)
        # idx=idx[:,None,:,None].expand(-1,dim_out,-1,r)
        # feature_map_pool=torch.gather(pooled_feature,2,idx)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, :,sample_idx, :]
        return vertices_pool,feature_map_pool

class anchor_pool(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out=dim_out
    def forward(self,x):
        # b
        out=torch.max(x,dim=-1)[0]
        return out


class tovec(nn.Module):
    def __init__(self, dim_out,dim_in=None):
        super().__init__()
        self.dim_out=dim_out
        if dim_in is not None:
            self.dim_in=dim_in
        else:
            self.dim_in=dim_out
        self.fc=nn.Linear(self.dim_in, self.dim_out, bias=False)
        self.atten_factor=FLAGS.atten_factor
    def forward(self,x,vs):
        # bs
        # x=self.attention_layer(x)
        if FLAGS.softmax:
            x=x.permute(0,2,3,1)
            q=self.fc(x)
            atten=F.softmax(q*self.atten_factor,dim=2)
            x=atten*x
            x=x.permute(0,3,1,2)
            out=x.unsqueeze(-1).expand(-1,-1,-1,-1,3)*(vs.float())
            out=out.sum(-2)
        else:
            out=x[:,:,:,:,None]*vs.float()
            out=out.sum(-2)
        return out


class tovec_2(nn.Module):
    def __init__(self, dim_out,dim_in=None):
        super().__init__()
        self.dim_out=dim_out
        if dim_in is not None:
            self.dim_in=dim_in
        else:
            self.dim_in=dim_out
        self.fc=nn.Linear(self.dim_in, self.dim_in, bias=True)
        self.atten_factor=FLAGS.atten_factor
        self.final_fc=VNLinear(self.dim_in,self.dim_in)
    def forward(self,x,vs):
        assert FLAGS.softmax==1
        x=x.permute(0,2,3,1)
        q=self.fc(x)
        atten=F.softmax(q,dim=2)
        atten=atten.permute(0,3,1,2)
        out=atten.unsqueeze(-1).expand(-1,-1,-1,-1,3)*(vs.float())
        out=out.sum(-2)
        out=self.final_fc(out.permute(0,2,1,3)).permute(0,2,1,3)
        return out


class tovec_3(nn.Module):
    def __init__(self, dim_out,dim_in=None):
        super().__init__()
        self.dim_out=dim_out
        if dim_in is not None:
            self.dim_in=dim_in
        else:
            self.dim_in=dim_out
        self.fc=nn.Linear(self.dim_in, self.dim_in, bias=True)
        self.final_fc=VNLinear(self.dim_in,self.dim_in)
    def forward(self,x,vs):
        # print( 'tovec3 !!!')
        assert FLAGS.softmax==1
        x=x.permute(0,2,3,1)
        q=self.fc(x)
        atten=F.softmax(q,dim=2)
        x=atten*x
        x=x.permute(0,3,1,2)
        out=x.unsqueeze(-1).expand(-1,-1,-1,-1,3)*(vs.float())
        out=out.sum(-2)
        out=self.final_fc(out.permute(0,2,1,3)).permute(0,2,1,3)
        return out




class equ_pool_layer3(nn.Module):
    def __init__(self, in_features,pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num
        mode='so3'
        self.pool=VecMaxPool(in_features=in_features, mode=mode, softmax_factor=1.0)
    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice _num, channel_num)
        """
        from pytorch3d.ops import sample_farthest_points
        bs, vertice_num, _ = vertices.size()
        r=feature_map.shape[-1]
        dim_out=feature_map.shape[1]
        N_new = vertice_num // self.pooling_rate
        if FLAGS.use_pool==1:
            neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
            new_neighbor_index=neighbor_index.reshape(bs,-1)[:,None,:,None].expand(-1,dim_out,-1,r)
            neighbor_feature=torch.gather(feature_map,2,new_neighbor_index).reshape(bs,dim_out,vertice_num,self.neighbor_num,r)
            pooled_feature = torch.max(neighbor_feature, dim= 3)[0]
            # pooled_feature=self.pool(neighbor_feature.permute(0,1,4,2,3)).permute(0,1,3,2)
            pooled_feature=pooled_feature
        else:
            pooled_feature=feature_map

        if FLAGS.down_sample_method == 'farthest':
            # print('farthest sampling!!')
            with torch.no_grad():
                vertices_pool, idx = sample_farthest_points(vertices, K=N_new)
            idx=idx[:,None,:,None].expand(-1,dim_out,-1,r)
            feature_map_pool=torch.gather(pooled_feature,2,idx)
        else:

            pool_num = int(vertice_num / self.pooling_rate)
            sample_idx = torch.randperm(vertice_num)[:pool_num]
            vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
            feature_map_pool = pooled_feature[:, :,sample_idx, :]
        return vertices_pool,feature_map_pool




if __name__ == "__main__":
    def eval_gcn_equi():
        vertices = torch.randn(4, 1024, 3).float().cuda()
        fea=torch.rand(vertices.shape[0],1,vertices.shape[1],12).float().cuda()

        R12=get_anchorsV12()
        R60=get_anchorsV()
        trace_idx_ori, trace_idx_rot = get_relativeV12_index()    # 12(rotation anchors)*12(indices on s2), 12*12
        full_trace_idx_ori,full_trace_idx_rot=get_relativeV_index()


        index=44
        index2=14

        # R=torch.from_numpy(R12[index]).float()
        # ro_rot=torch.from_numpy(trace_idx_rot[index]).long()
        # ro_ori=torch.from_numpy(trace_idx_ori[index]).long()

        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()
        ro_rot=torch.from_numpy(full_trace_idx_rot[index]).long().cuda()
        ro_ori=torch.from_numpy(full_trace_idx_ori[index]).long().cuda()
        conv=equi_conv(dim_in=1,dim_out=32).cuda()

        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)
        # fea_rot=torch.einsum('ij,bcpj->bcpi',R,fea)
        fea_rot=fea[:,:,:,ro_rot]
        neighbor_index = get_neighbor_index(vertices, 10)
        out=conv(neighbor_index,vertices,fea)
        neighbor_index_rot = get_neighbor_index(vertices_rot, 10).cuda()
        out_rot=conv(neighbor_index_rot,vertices_rot,fea_rot)
        # out_ori=out_rot[:,:,:,ro_ori]
        out_ori=torch.einsum('ij,bcpi->bcpj',R2,out_rot)
        # out_diff=out-out_ori
        # out_norm=torch.norm(out_diff,dim=-1).sum()

        out_norm=F.normalize(out,dim=-1)
        out_ori_norm=F.normalize(out_ori,dim=-1)
        dot=torch.einsum('bdpi,bdpi->bdp',out_norm,out_ori_norm)
        theta = torch.acos(dot)
        nan_mask = torch.isnan(theta)

    # Replace NaN values with zeros
        theta = torch.where(nan_mask, torch.tensor(0.0).to(theta.device), theta)
        theta=theta.mean()
        return
    def eval_gcn_neuron():
        vertices = torch.randn(4, 1024, 3).float().cuda()
        fea=torch.rand(vertices.shape[0],1,vertices.shape[1],3).float().cuda()

        R60=get_anchorsV()

        index=37
        index2=14


        R=torch.from_numpy(R60[index]).float().cuda()
        R2=torch.from_numpy(R60[index2]).float().cuda()

        conv=equi_conv4(dim_in=1,dim_out=32).cuda()

        vertices_rot=torch.einsum('ij,bpj->bpi',R,vertices)
        fea_rot=torch.einsum('ij,bcpj->bcpi',R,fea)

        neighbor_index = get_neighbor_index(vertices, 10)
        out=conv(neighbor_index,vertices,fea)
        neighbor_index_rot = get_neighbor_index(vertices_rot, 10).cuda()
        out_rot=conv(neighbor_index_rot,vertices_rot,fea_rot)
        out_ori=torch.einsum('ij,bcpi->bcpj',R,out_rot)

        out_norm=F.normalize(out,dim=-1)
        out_ori_norm=F.normalize(out_ori,dim=-1)
        dot=torch.einsum('bdpi,bdpi->bdp',out_norm,out_ori_norm)
        theta = torch.acos(dot)
        nan_mask = torch.isnan(theta)

        # Replace NaN values with zeros
        theta = torch.where(nan_mask, torch.tensor(0.0).to(theta.device), theta)
        theta=theta.mean()
        return
    # eval_gcn_neuron()
    eval_gcn_equi()