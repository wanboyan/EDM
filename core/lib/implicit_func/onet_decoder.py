import torch.nn as nn
import torch.nn.functional as F
import torch

import absl.flags as flags
FLAGS = flags.FLAGS

# Resnet Blocks
class ResnetBlockFC(nn.Module):
    """Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    """

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CResnetBlockConv1d(nn.Module):
    """Conditional batch normalization-based Resnet block class.

    Args:
        c_dim (int): dimension of latend conditioned code c
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
        norm_method (str): normalization method
        legacy (bool): whether to use legacy blocks
    """

    def __init__(
        self, c_dim, size_in, size_h=None, size_out=None, norm_method="batch_norm", legacy=False
    ):
        super().__init__()
        # Attributes
        if size_h is None:
            size_h = size_in
        if size_out is None:
            size_out = size_in

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        if not legacy:
            self.bn_0 = CBatchNorm1d(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d(c_dim, size_h, norm_method=norm_method)
        else:
            self.bn_0 = CBatchNorm1d_legacy(c_dim, size_in, norm_method=norm_method)
            self.bn_1 = CBatchNorm1d_legacy(c_dim, size_h, norm_method=norm_method)

        self.fc_0 = nn.Conv1d(size_in, size_h, 1)
        self.fc_1 = nn.Conv1d(size_h, size_out, 1)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv1d(size_in, size_out, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, c):
        net = self.fc_0(self.actvn(self.bn_0(x, c)))
        dx = self.fc_1(self.actvn(self.bn_1(net, c)))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx


class CBatchNorm1d(nn.Module):
    """Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    """

    def __init__(self, c_dim, f_dim, norm_method="batch_norm"):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.conv_gamma = nn.Conv1d(c_dim, f_dim, 1)
        self.conv_beta = nn.Conv1d(c_dim, f_dim, 1)
        if norm_method == "batch_norm":
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == "instance_norm":
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == "group_norm":
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError("Invalid normalization method!")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, x, c):
        assert x.size(0) == c.size(0)
        assert c.size(1) == self.c_dim

        # c is assumed to be of size batch_size x c_dim x T
        if len(c.size()) == 2:
            c = c.unsqueeze(2)

        # Affine mapping
        gamma = self.conv_gamma(c)
        beta = self.conv_beta(c)

        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class CBatchNorm1d_legacy(nn.Module):
    """Conditional batch normalization legacy layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    """

    def __init__(self, c_dim, f_dim, norm_method="batch_norm"):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.norm_method = norm_method
        # Submodules
        self.fc_gamma = nn.Linear(c_dim, f_dim)
        self.fc_beta = nn.Linear(c_dim, f_dim)
        if norm_method == "batch_norm":
            self.bn = nn.BatchNorm1d(f_dim, affine=False)
        elif norm_method == "instance_norm":
            self.bn = nn.InstanceNorm1d(f_dim, affine=False)
        elif norm_method == "group_norm":
            self.bn = nn.GroupNorm1d(f_dim, affine=False)
        else:
            raise ValueError("Invalid normalization method!")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.ones_(self.fc_gamma.bias)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, x, c):
        batch_size = x.size(0)
        # Affine mapping
        gamma = self.fc_gamma(c)
        beta = self.fc_beta(c)
        gamma = gamma.view(batch_size, self.f_dim, 1)
        beta = beta.view(batch_size, self.f_dim, 1)
        # Batchnorm
        net = self.bn(x)
        out = gamma * net + beta

        return out


class Decoder(nn.Module):
    """Basic Decoder network for OFlow class.
    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This basic
    decoder does not use batch normalization.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
    """

    def __init__(
        self, dim=3, z_dim=128, c_dim=128, hidden_size=128, leaky=False, out_dim=1, **kwargs
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        # Submodules
        self.fc_p = nn.Linear(dim, hidden_size)

        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)
        if not c_dim == 0:
            self.fc_c = nn.Linear(c_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z=None, c=None, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        batch_size = p.shape[0]
        p = p.view(batch_size, -1, self.dim)
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(1)
            net = net + net_z

        if self.c_dim != 0:
            net_c = self.fc_c(c).unsqueeze(1)
            net = net + net_c

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class DecoderCat(nn.Module):
    # * No input linear amplifier, directly cat inout

    def __init__(self, input_dim=3, hidden_size=128, leaky=False, out_dim=1, **kwargs):
        super().__init__()

        self.fc_in = nn.Linear(input_dim, hidden_size)

        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, x):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        net = self.fc_in(x)

        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out


class DecoderCBatchNorm(nn.Module):
    """Conditioned Batch Norm Decoder network for OFlow class.

    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This decoder
    uses conditioned batch normalization to inject the latent codes.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation

    """

    def __init__(
        self, dim=3, z_dim=128, c_dim=128, hidden_size=256, leaky=False, out_dim=1, legacy=False
    ):
        super().__init__()
        self.z_dim = z_dim
        self.dim = dim
        if not z_dim == 0:
            self.fc_z = nn.Linear(z_dim, hidden_size)

        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size, legacy=legacy)

        if not legacy:
            self.bn = CBatchNorm1d(c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, out_dim, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, z, c, return_feat_list=False, **kwargs):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.fc_p(p)

        if self.z_dim != 0:
            net_z = self.fc_z(z).unsqueeze(2)
            net = net + net_z

        if return_feat_list:
            feat_list = [net]

        net = self.block0(net, c)
        if return_feat_list:
            feat_list.append(net)
        net = self.block1(net, c)
        if return_feat_list:
            feat_list.append(net)
        net = self.block2(net, c)
        if return_feat_list:
            feat_list.append(net)
        net = self.block3(net, c)
        if return_feat_list:
            feat_list.append(net)
        net = self.block4(net, c)
        if return_feat_list:
            feat_list.append(net)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        if return_feat_list:
            return out, feat_list
        else:
            return out

from e3nn import o3
from .tensor_product_rescale import (TensorProductRescale, LinearRS,
                                     FullyConnectedTensorProductRescale,
                                     FullyConnectedTensorProductRescaleSwishGate,
                                     DepthwiseTensorProduct,MyDepthwiseTensorProduct,
                                     MyDepthwiseTensorProduct_2,
                                     irreps2gate, sort_irreps_even_first)
from .fast_activation import Gate



class Decoder_v2(nn.Module):
    """Basic Decoder network for OFlow class.
    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This basic
    decoder does not use batch normalization.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
    """

    def __init__(
            self,
    ):
        super().__init__()
        self.sph1=o3.SphericalHarmonics(irreps_out = '1x0e+1x1e+1x2e', normalize = True, normalization='component')

        self.irreps_node_input = o3.Irreps('256x1e')
        self.irreps_edge_attr = o3.Irreps('1x0e+1x1e+1x2e')
        self.irreps_node_output = o3.Irreps('128x0e+64x1e+32x2e')


        self.dtp: TensorProductRescale = MyDepthwiseTensorProduct(self.irreps_node_input,
                                                                self.irreps_edge_attr,
                                                                self.irreps_node_output,
                                                                bias=False,)
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), self.irreps_node_output)

        self.block0 = ResBlock(self.irreps_node_output,self.irreps_edge_attr)
        self.block1 = ResBlock(self.irreps_node_output,self.irreps_edge_attr)
        self.block2 = ResBlock(self.irreps_node_output,self.irreps_edge_attr)
        self.block3 = ResBlock(self.irreps_node_output,self.irreps_edge_attr)
        self.block4 = ResBlock(self.irreps_node_output,self.irreps_edge_attr)

        self.inv = MyDepthwiseTensorProduct_2(self.irreps_node_output,
                                              self.irreps_node_output,
                                                                    o3.Irreps('1x0e'),
                                                                    bias=False,)
        self.actvn = F.relu
        self.fc_out=nn.Linear(self.inv.irreps_out.dim,1)

    def forward(self, z_so3,query):
        bs=z_so3.shape[0]
        query_num=query.shape[1]
        sph1=self.sph1(query)
        length=torch.norm(query,dim=-1)
        query_sph=sph1*length.unsqueeze(-1)
        z_so3=z_so3[:,None,:,:].repeat(1,query_num,1,1)

        z_so3=z_so3.reshape(bs*query_num,-1)
        query_sph=query_sph.reshape(bs*query_num,-1)
        input=self.dtp(z_so3,query_sph)
        net=self.lin(input)

        net = self.block0(net,query_sph)
        net = self.block1(net,query_sph)
        net = self.block2(net,query_sph)
        net = self.block3(net,query_sph)
        net = self.block4(net,query_sph)
        inv=self.inv(net,net)

        out = self.fc_out(self.actvn(inv))
        out = out.reshape(bs,query_num)
        return out


class ResBlock(nn.Module):
    def __init__(
            self,irreps_node_input: o3.Irreps,irreps_edge_attr
    ):
        super().__init__()
        self.lin_act_1=Lin_act(irreps_node_input,irreps_node_input,irreps_edge_attr)
        self.lin_act_2=Lin_act(irreps_node_input,irreps_node_input,irreps_edge_attr)
    def forward(self, node_input: torch.Tensor,query):
        output=self.lin_act_1(node_input,query)
        output=self.lin_act_2(output,query)
        output=output+node_input
        return output

class Lin_act(nn.Module):
    def __init__(
            self,irreps_node_input: o3.Irreps,irreps_node_output: o3.Irreps,irreps_edge_attr: o3.Irreps
    ):
        super().__init__()
        self.irreps_node_input=irreps_node_input
        self.irreps_node_output=irreps_node_output
        self.irreps_edge_attr=irreps_edge_attr
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        irreps_lin_output: o3.Irreps = irreps_scalars + irreps_gates + irreps_gated
        irreps_lin_output: o3.Irreps = irreps_lin_output.simplify()

        self.dtp: TensorProductRescale = MyDepthwiseTensorProduct_2(self.irreps_node_input,
                                                                self.irreps_node_input,
                                                                o3.Irreps('1x0e'),
                                                                bias=False,)
        irreps_lin_input=(self.dtp.irreps_out.simplify()+self.irreps_node_input).simplify()
        self.lin = LinearRS(irreps_lin_input, irreps_lin_output)
        gate = Gate(
            irreps_scalars, [torch.nn.SiLU() for _ in irreps_scalars],  # scalar
            irreps_gates, [torch.sigmoid for _ in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )
        self.gate = gate
    def forward(self, node_input: torch.Tensor,query):
        out=self.dtp(node_input,node_input)
        out=torch.cat([out,node_input],dim=-1)
        out = self.lin(out)
        out = self.gate(out)
        return out

import numpy as np
class Decoder_v3(nn.Module):
    """Basic Decoder network for OFlow class.
    The decoder network maps points together with latent conditioned codes
    c and z to log probabilities of occupancy for the points. This basic
    decoder does not use batch normalization.

    Args:
        dim (int): dimension of input points
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): dimension of hidden size
        leaky (bool): whether to use leaky ReLUs as activation
    """

    def __init__(
            self, hidden_size=256, leaky=False, out_dim=1,
    ):
        super().__init__()

        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        self.pe_pow=10
        self.input_dim=277
        self.fc_q=nn.Linear(self.input_dim, hidden_size)
        self.fc_v=nn.Linear(513, hidden_size)
        self.pe_sigma = np.pi * torch.pow(2, torch.linspace(0, self.pe_pow - 1, self.pe_pow))
        self.block0 = ResnetBlockFC(hidden_size)
        self.block1 = ResnetBlockFC(hidden_size)
        self.block2 = ResnetBlockFC(hidden_size)
        self.block3 = ResnetBlockFC(hidden_size)
        self.block4 = ResnetBlockFC(hidden_size)

        self.fc_out = nn.Linear(hidden_size, out_dim)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, z_so3,z_inv,query):
        """Performs a forward pass through the network.

        Args:
            p (tensor): points tensor
            z (tensor): latent code z
            c (tensor): latent conditioned code c
        """
        batch_size = query.shape[0]
        length=torch.norm(query,dim=-1,keepdim=True)
        query_num=query.shape[1]
        z_inv=z_inv[:,None,:].repeat(1,query_num,1)
        z_so3_repeat=z_so3[:,None,:,:].repeat(1,query_num,1,1)
        z_so3_dis=torch.einsum('bqdi,ri->bqdr',z_so3_repeat,self.vs)
        query_dis=torch.einsum('bqi,ri->bqr',query,self.vs)

        query_dis_pe=self.positional_encoder(query_dis).permute(0,1,3,2)
        input=torch.cat([z_so3_dis,query_dis_pe],dim=2).permute(0,1,3,2)

        q = self.actvn(self.fc_q(input))
        q = torch.mean(q,dim=2)
        net=torch.cat([length,q,z_inv],dim=-1)
        net=self.fc_v(net)
        net = self.block0(net)
        net = self.block1(net)
        net = self.block2(net)
        net = self.block3(net)
        net = self.block4(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        return out

    def positional_encoder(self, x):
        device = x.device
        y = torch.cat(
            [
                x[..., None],
                torch.sin(x[:, :, :, None] * self.pe_sigma[None, None, None].to(device)),
                torch.cos(x[:, :, :, None] * self.pe_sigma[None, None, None].to(device)),
            ],
            dim=-1,
        )
        return y
