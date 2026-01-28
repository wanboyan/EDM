import torch
import torch.nn as nn
import torch.nn.functional as F
import pointnet2_utils




class Sampler(nn.Module):
    def __init__(self):
        super(Sampler, self).__init__()
        self.npoint = 10

    def forward(self, xyz: torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []
        xyz = pointnet2_utils.furthest_point_sample(xyz, self.npoint)



        return xyz

x = torch.rand(3,20,3).to("cuda:0")

sampler = Sampler().to("cuda:0")

result =  sampler(x)


print(result)