import torch.nn.functional as F

from torch import nn, einsum, Tensor
import torch
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
from torch.nn.init import xavier_uniform_
from e3nn import o3
import e3nn.nn
import collections
from e3nn.math import perm
from eqnet.transformer.graph_norm import *


from typing import List, Tuple, Optional, Dict, Any,Union, Callable

def inner_dot_product(x, y, *, dim = -1, keepdim = True):
    return (x * y).sum(dim = dim, keepdim = keepdim)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class VNLinear(nn.Module):
    def __init__(
            self,
            dim_in,
            dim_out,
            bias_epsilon = 0.
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim_out, dim_in))
        xavier_uniform_(self.weight)


    def forward(self, x):
        out = einsum('... i c, o i -> ... o c', x, self.weight)


        return out

class VNReLU(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.empty(dim, dim))
        self.U = nn.Parameter(torch.empty(dim, dim))
        xavier_uniform_(self.W)
        xavier_uniform_(self.U)

    def forward(self, x):
        q = einsum('... i c, o i -> ... o c', x, self.W)
        k = einsum('... i c, o i -> ... o c', x, self.U)

        qk = inner_dot_product(q, k)

        k_norm = k.norm(dim = -1, keepdim = True).clamp(min = self.eps)
        q_projected_on_k = q - inner_dot_product(q, k / k_norm) * k

        out = torch.where(
            qk >= 0.,
            q,
            q_projected_on_k
        )

        return out

def VNFeedForward(dim, dim_inner, bias_epsilon = 0.):
    return nn.Sequential(
        VNLinear(dim, dim_inner, bias_epsilon = bias_epsilon),
        VNReLU(dim_inner),
        VNLinear(dim_inner, dim, bias_epsilon = bias_epsilon)
    )

class VNLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = LayerNorm(dim)

    def forward(self, x):
        norms = x.norm(dim = -1)
        x = x / rearrange(norms.clamp(min = self.eps), '... -> ... 1')
        ln_out = self.ln(norms)
        return x * rearrange(ln_out, '... -> ... 1')

class VNLayerNorm2(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.ln = nn.BatchNorm1d(dim)

    def forward(self, x):
        norms = x.norm(dim = -1)
        # print('asd')
        x = x / rearrange(norms.clamp(min = self.eps), '... -> ... 1')
        ln_out = self.ln(norms.permute(0,2,1)).permute(0,2,1)
        return x * rearrange(ln_out, '... -> ... 1')


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


from typing import Union, Optional, List, Tuple, Dict
import math

@torch.jit.script
def gaussian(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    # pi = 3.14159
    # a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) # / (a * std)

class GaussianRadialBasis(torch.nn.Module):
    def __init__(self, dim: int, max_val: Union[float, int], min_val: Union[float, int] = 0.):
        super().__init__()
        self.dim: int = dim
        self.max_val: float = float(max_val)
        self.min_val: float = float(min_val)
        if self.min_val < 0.:
            warnings.warn(f"Negative min_val ({self.min_val}) is provided for radial basis encoder. Are you sure?")

        self.mean_init_max = 1.0
        self.mean_init_min = 0
        mean = torch.linspace(self.mean_init_min, self.mean_init_max, self.dim+2)[1:-1].unsqueeze(0)
        self.mean = torch.nn.Parameter(mean)

        self.std_logit = torch.nn.Parameter(torch.zeros(1, self.dim))        # Softplus logit
        self.weight_logit = torch.nn.Parameter(torch.zeros(1, self.dim))     # Sigmoid logit

        init_std = 2.0 / self.dim
        torch.nn.init.constant_(self.std_logit, math.log(math.exp((init_std)) -1)) # Inverse Softplus

        self.max_weight = 4.
        torch.nn.init.constant_(self.weight_logit, -math.log(self.max_weight/1. - 1)) # Inverse Softplus

        self.normalizer = math.sqrt(self.dim)


    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        dist = (dist - self.min_val) / (self.max_val - self.min_val)
        dist = dist.unsqueeze(-1)

        x = dist.expand(-1, self.dim)
        mean = self.mean
        std = F.softplus(self.std_logit) + 1e-5
        x = gaussian(x, mean, std)
        x = torch.sigmoid(self.weight_logit) * self.max_weight * x

        return x * self.normalizer

@torch.jit.script
def soft_step(x, n: int = 3):
    return (x>0) * ((x<1)*((n+1)*x.pow(n)-n*x.pow(n+1)) + (x>=1))




def soft_square_cutoff_2(x: torch.Tensor, ranges: Optional[Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]], n:int = 3) -> torch.Tensor:
    """
    Input:
        ranges: (left_end, left_begin, right_begin, right_end)
        n: n-th polynomial is used.
    """
    if ranges is None:
        return x

    if len(ranges) != 4:
        raise ValueError(f"Wrong ranges armument: {ranges}")
    left_end, left_begin, right_begin, right_end = ranges
    if left_end is None or left_begin is None:
        if not (left_end is None and left_begin is None):
            raise ValueError(f"Wrong ranges armument: {ranges}")
        div_l: float = 1.
    else:
        div_l: float = left_begin - left_end

    if right_end is None or right_begin is None:
        if not (right_end is None and right_begin is None):
            raise ValueError(f"Wrong ranges armument: {ranges}")
        div_r: float = 1.
    else:
        div_r: float = right_end - right_begin


    if right_begin is not None and left_end is None:
        y = 1-soft_step((x-right_begin) / div_r, n=n)
    elif left_end is not None and right_begin is None:
        y = soft_step((x-left_end) / div_l, n=n)
    elif right_begin is not None and left_end is not None and left_begin is not None:
        if left_begin > right_begin:
            raise ValueError(f"Wrong ranges armument: {ranges}")
        y = (1-soft_step((x-right_begin) / div_r, n=n)) * (x>0.5*(left_begin+right_begin)) + soft_step((x-left_end) / div_l, n=n) * (x<=0.5*(left_begin+right_begin))
    else:
        y = torch.ones_like(x)

    return y

def cutoff_irreps(f: torch.Tensor,
                  edge_cutoff: Optional[torch.Tensor],
                  cutoff_scalar: Optional[torch.Tensor],
                  cutoff_nonscalar: Optional[torch.Tensor],
                  irreps: List[Tuple[int, Tuple[int, int]]],
                  log: bool = False) -> torch.Tensor:
    if edge_cutoff is None and cutoff_scalar is None and cutoff_nonscalar is None:
        return f

    f_cutoff = []
    last_idx = 0
    for n, (l,p) in irreps:
        d = n * (2*l + 1)
        if l == 0 and cutoff_scalar is not None:
            if log is True:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * torch.exp(cutoff_scalar[..., None])
                )
            else:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * cutoff_scalar[..., None]
                )
        elif l != 0 and cutoff_nonscalar is not None:
            if log is True:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * torch.exp(cutoff_nonscalar[..., None])
                )
            else:
                f_cutoff.append(
                    f[..., last_idx: last_idx+d] * cutoff_nonscalar[..., None]
                )
        else:
            f_cutoff.append(f[..., last_idx: last_idx+d])

        last_idx = last_idx + d

    f_cutoff = torch.cat(f_cutoff, dim=-1)

    if edge_cutoff is not None:
        if log is True:
            f_cutoff = f_cutoff * torch.exp(edge_cutoff[..., None])
        else:
            f_cutoff = f_cutoff * edge_cutoff[..., None]

    return f_cutoff

class SmoothLeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope: float = 0.2):
        super().__init__()
        self.alpha: float = negative_slope


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = ((1 + self.alpha) / 2) * x
        x2 = ((1 - self.alpha) / 2) * x * (2 * torch.sigmoid(x) - 1)
        return x1 + x2


    def extra_repr(self):
        return 'negative_slope={}'.format(self.alpha)

from e3nn.math import normalize2mom

#@compile_mode('script')
class Activation(torch.nn.Module):
    r"""Scalar activation function.
    Unlike e3nn.nn.Activation, this module directly apply activation when irreps is type-0.

    Odd scalar inputs require activation functions with a defined parity (odd or even).

    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input

    acts : list of function or None
        list of activation functions, `None` if non-scalar or identity

    Examples
    --------
    Note that 'acts' is a list of nonlinearity (activation).
    >>> a = Activation(irreps_in = "256x0o", acts = [torch.abs])
    >>> a.irreps_out
    256x0e

    Note that 'acts' is a list of nonlinearity (activation).
    >>> a = Activation(irreps_in = "256x0o+16x1e", acts = [torch.nn.SiLU(), None])
    >>> a.irreps_out
    256x0o+16x1e

    'acts' must be 'None' for non-scalar (L>=1) irrep parts.
    >>> a = Activation(irreps_in = "256x0o+16x1e", acts = [torch.nn.SiLU(), torch.nn.SiLU()])
    >>> < ValueError("Activation: cannot apply an activation function to a non-scalar input.") >
    """
    def __init__(self, irreps_in: o3.Irreps, acts: List[Optional[Callable]]):
        """__init__() is Completely Identical to e3nn.nn.Activation.__init__()
        """
        super().__init__()
        irreps_in: o3.Irreps = o3.Irreps(irreps_in)
        assert len(irreps_in) == len(acts), (irreps_in, acts)

        # normalize the second moment
        acts: List[Optional[Callable]] = [normalize2mom(act) if act is not None else None for act in acts] # normalize moment (e3nn functionality)

        from e3nn.util._argtools import _get_device

        irreps_out = []
        for (mul, (l_in, p_in)), act in zip(irreps_in, acts):
            if act is not None:
                if l_in != 0:
                    raise ValueError("Activation: cannot apply an activation function to a non-scalar input.")

                x = torch.linspace(0, 10, 256, device=_get_device(act))

                a1, a2 = act(x), act(-x)
                if (a1 - a2).abs().max() < 1e-5:
                    p_act = 1
                elif (a1 + a2).abs().max() < 1e-5:
                    p_act = -1
                else:
                    p_act = 0

                p_out = p_act if p_in == -1 else p_in
                irreps_out.append((mul, (0, p_out)))

                if p_out == 0:
                    raise ValueError("Activation: the parity is violated! The input scalar is odd but the activation is neither even nor odd.")
            else:
                irreps_out.append((mul, (l_in, p_in)))

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out)

        self.is_acts_none: List[bool] = []
        for i, act in enumerate(acts):
            if act is None:
                acts[i] = torch.nn.Identity()
                self.is_acts_none.append(True)
            else:
                self.is_acts_none.append(False)
        self.acts = torch.nn.ModuleList(acts)
        self.is_acts_none = tuple(self.is_acts_none)

        assert len(self.irreps_in) == len(self.acts)

        # If there is only one irrep in o3.Irreps, and the irrep is a scalar, then just apply the only activation to it.
        # For example, "8x0e" is the case.
        # On the other hand, "8x0e+7x0e", "8x1e", "8x0e+7x1e" is not the case.
        if len(self.acts) == 1 and self.acts[0] is not None: # activation for non-scalar irrep cannot be 'None', thus the only irrep must be scalar (L=0).
            self.simple: bool = True
        else:
            self.simple: bool = False

    #def __repr__(self):
    #    acts = "".join(["x" if a is not None else " " for a in self.acts])
    #    return f"{self.__class__.__name__} [{self.acts}] ({self.irreps_in} -> {self.irreps_out})"

    def extra_repr(self):
        output_str = super(Activation, self).extra_repr()
        output_str = output_str + '{} -> {}, '.format(self.irreps_in, self.irreps_out)
        return output_str


    def forward(self, features: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # If there is only one irrep in o3.Irreps, and the irrep is a scalar, then just apply the only activation to it.
        # For example, "8x0e" is the case.
        # On the other hand, "8x0e+7x0e", "8x1e", "8x0e+7x1e" is not the case.
        if self.simple: # activation for non-scalar irrep cannot be 'None', thus the only irrep must be scalar (L=0).
            return self.acts[0](features)

        # Otherwise, same behavior as e3nn.nn.Activation.forward()
        output = []
        index = 0
        for (mul, ir), act, is_act_none in zip(self.irreps_in, self.acts, self.is_acts_none):
            if not is_act_none:
                output.append(act(features.narrow(dim, index, mul)))
            else:
                output.append(features.narrow(dim, index, mul * (2*ir[0] + 1)))
            index += mul * (2*ir[0] + 1)

        if len(output) > 1:
            return torch.cat(output, dim=dim)
        elif len(output) == 1:
            return output[0]
        else:
            return torch.zeros_like(features)


#@compile_mode('script')
class Gate(torch.nn.Module):
    '''
        1. Use `narrow` to split tensor.
        2. Use `Activation` in this file.
    '''
    def __init__(self, irreps_scalars: o3.Irreps, act_scalars: List[Callable],
                 irreps_gates: o3.Irreps, act_gates: List[Callable],
                 irreps_gated: o3.Irreps):
        super().__init__()
        irreps_scalars: o3.Irreps = o3.Irreps(irreps_scalars)
        irreps_gates: o3.Irreps = o3.Irreps(irreps_gates)
        irreps_gated: o3.Irreps = o3.Irreps(irreps_gated)

        if len(irreps_gates) > 0 and irreps_gates.lmax > 0:
            raise ValueError(f"Gate scalars must be scalars, instead got irreps_gates = {irreps_gates}")
        if len(irreps_scalars) > 0 and irreps_scalars.lmax > 0:
            raise ValueError(f"Scalars must be scalars, instead got irreps_scalars = {irreps_scalars}")
        if irreps_gates.num_irreps != irreps_gated.num_irreps:
            raise ValueError(f"There are {irreps_gated.num_irreps} irreps in irreps_gated, but a different number ({irreps_gates.num_irreps}) of gate scalars in irreps_gates")

        self.irreps_scalars: o3.Irreps = irreps_scalars
        self.irreps_gates: o3.Irreps = irreps_gates
        self.irreps_gated: o3.Irreps = irreps_gated
        self._irreps_in: o3.Irreps = (irreps_scalars + irreps_gates + irreps_gated).simplify()

        self.act_scalars = Activation(irreps_scalars, act_scalars)
        irreps_scalars: o3.Irreps = self.act_scalars.irreps_out

        self.act_gates = Activation(irreps_gates, act_gates)
        irreps_gates: o3.Irreps = self.act_gates.irreps_out

        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        irreps_gated: o3.Irreps = self.mul.irreps_out

        for (mul, ir), (mul2, ir2) in zip(self.irreps_scalars, irreps_scalars):
            assert mul == mul2 and ir[0] == ir2[0] and ir[1] == ir2[1]
        for (mul, ir), (mul2, ir2) in zip(self.irreps_gates, irreps_gates):
            assert mul == mul2 and ir[0] == ir2[0] and ir[1] == ir2[1]
        for (mul, ir), (mul2, ir2) in zip(self.irreps_gated, irreps_gated):
            assert mul == mul2 and ir[0] == ir2[0] and ir[1] == ir2[1]

        self._irreps_out = irreps_scalars + irreps_gated

        self.scalars_dim: int = self.irreps_scalars.dim
        self.gates_dim: int = self.irreps_gates.dim
        self.gated_dim: int = self.irreps_gated.dim
        self.input_dim: int = self.irreps_in.dim
        assert self.scalars_dim + self.gates_dim + self.gated_dim == self.input_dim


    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features.shape == (..., scalar + gates + gated)
        assert features.shape[-1] == self.input_dim
        scalars = features.narrow(-1, 0, self.scalars_dim)
        gates = features.narrow(-1, self.scalars_dim, self.gates_dim)
        gated = features.narrow(-1, (self.scalars_dim + self.gates_dim), self.gated_dim)

        scalars = self.act_scalars(scalars)
        if gates.shape[-1]:
            gates = self.act_gates(gates)
            gated = self.mul(gated, gates)
            features = torch.cat([scalars, gated], dim=-1)
        else:
            features = scalars
        return features


    @property
    def irreps_in(self):
        """Input representations."""
        return self._irreps_in


    @property
    def irreps_out(self):
        """Output representations."""
        return self._irreps_out




class TensorProductRescale(torch.nn.Module):
    def __init__(self,
                 irreps_in1: o3.Irreps, irreps_in2: o3.Irreps, irreps_out: o3.Irreps,
                 instructions: List[Tuple[int, int, int, str, bool, float]],
                 bias: bool = True, rescale: bool = True,
                 internal_weights: Optional[bool] = None, shared_weights: Optional[bool] = None,
                 normalization: Optional[str] = None):

        super().__init__()

        self.irreps_in1: o3.Irreps = irreps_in1
        self.irreps_in2: o3.Irreps = irreps_in2
        self.irreps_out: o3.Irreps = irreps_out
        self.rescale: bool = rescale
        self.use_bias: bool = bias

        # e3nn.__version__ == 0.4.4
        # Use `path_normalization` == 'none' to remove normalization factor
        self.tp = o3.TensorProduct(irreps_in1=self.irreps_in1,
                                   irreps_in2=self.irreps_in2, irreps_out=self.irreps_out,
                                   instructions=instructions, normalization=normalization,
                                   internal_weights=internal_weights, shared_weights=shared_weights,
                                   path_normalization='none')

        self.init_rescale_bias()

        # self.register_buffer(name='_bias_slices',
        #                      tensor=torch.tensor([(slice_.start, slice_.stop) for slice_ in self.bias_slices], dtype=torch.long),
        #                      persistent=False)
        self.bias_slices = tuple([(slice_.start, slice_.stop) for slice_ in self.bias_slices])

    @torch.jit.unused
    def calculate_fan_in(self, ins):
        return {
            'uvw': (self.irreps_in1[ins.i_in1].mul * self.irreps_in2[ins.i_in2].mul),
            'uvu': self.irreps_in2[ins.i_in2].mul,
            'uvv': self.irreps_in1[ins.i_in1].mul,
            'uuw': self.irreps_in1[ins.i_in1].mul,
            'uuu': 1,
            'uvuv': 1,
            'uvu<v': 1,
            'u<vw': self.irreps_in1[ins.i_in1].mul * (self.irreps_in2[ins.i_in2].mul - 1) // 2,
        }[ins.connection_mode]

    @torch.jit.unused
    def init_rescale_bias(self) -> None:

        irreps_out = self.irreps_out
        # For each zeroth order output irrep we need a bias
        # Determine the order for each output tensor and their dims
        self.irreps_out_orders = [int(irrep_str[-2]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(irreps_out).split('+')]
        self.irreps_out_slices = irreps_out.slices()

        # Store tuples of slices and corresponding biases in a list
        self.bias = None
        self.bias_slices = []
        self.bias_slice_idx = []
        self.irreps_bias = self.irreps_out.simplify()
        self.irreps_bias_orders = [int(irrep_str[-2]) for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_parity = [irrep_str[-1] for irrep_str in str(self.irreps_bias).split('+')]
        self.irreps_bias_dims = [int(irrep_str.split('x')[0]) for irrep_str in str(self.irreps_bias).split('+')]
        if self.use_bias:
            self.bias = []
            for slice_idx in range(len(self.irreps_bias_orders)):
                if self.irreps_bias_orders[slice_idx] == 0 and self.irreps_bias_parity[slice_idx] == 'e':
                    out_slice = self.irreps_bias.slices()[slice_idx]
                    out_bias = torch.nn.Parameter(
                        torch.zeros(self.irreps_bias_dims[slice_idx], dtype=self.tp.weight.dtype))
                    self.bias += [out_bias]
                    self.bias_slices += [out_slice]
                    self.bias_slice_idx += [slice_idx]
        self.bias = torch.nn.ParameterList(self.bias)

        self.slices_sqrt_k = {}
        with torch.no_grad():
            # Determine fan_in for each slice, it could be that each output slice is updated via several instructions
            slices_fan_in = {}  # fan_in per slice
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                fan_in = self.calculate_fan_in(instr)
                slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                            fan_in if slice_idx in slices_fan_in.keys() else fan_in)
            for instr in self.tp.instructions:
                slice_idx = instr[2]
                if self.rescale:
                    sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                else:
                    sqrt_k = 1.
                self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Re-initialize weights in each instruction
            if self.tp.internal_weights:
                for weight, instr in zip(self.tp.weight_views(), self.tp.instructions):
                    # The tensor product in e3nn already normalizes proportional to 1 / sqrt(fan_in), and the weights are by
                    # default initialized with unif(-1,1). However, we want to be consistent with torch.nn.Linear and
                    # initialize the weights with unif(-sqrt(k),sqrt(k)), with k = 1 / fan_in
                    slice_idx = instr[2]
                    if self.rescale:
                        sqrt_k = 1 / slices_fan_in[slice_idx] ** 0.5
                        weight.data.mul_(sqrt_k)
                    #else:
                    #    sqrt_k = 1.
                    #
                    #if self.rescale:
                    #weight.data.uniform_(-sqrt_k, sqrt_k)
                    #    weight.data.mul_(sqrt_k)
                    #self.slices_sqrt_k[slice_idx] = (self.irreps_out_slices[slice_idx], sqrt_k)

            # Initialize the biases
            #for (out_slice_idx, out_slice, out_bias) in zip(self.bias_slice_idx, self.bias_slices, self.bias):
            #    sqrt_k = 1 / slices_fan_in[out_slice_idx] ** 0.5
            #    out_bias.uniform_(-sqrt_k, sqrt_k)


    def forward_tp_rescale_bias(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        out = self.tp(x, y, weight)

        #if self.rescale and self.tp.internal_weights:
        #    for (slice, slice_sqrt_k) in self.slices_sqrt_k.values():
        #        out[:, slice] /= slice_sqrt_k
        if self.use_bias:
            for slice, bias in zip(self.bias_slices, self.bias):
                #out[:, slice] += bias
                out.narrow(1, slice[0], slice[1] - slice[0]).add_(bias)

        return out


    def forward(self, x: torch.Tensor, y: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = self.forward_tp_rescale_bias(x, y, weight)
        return out


class FullyConnectedTensorProductRescale(TensorProductRescale):
    def __init__(self,
                 irreps_in1: o3.Irreps, irreps_in2: o3.Irreps, irreps_out: o3.Irreps,
                 bias: bool = True, rescale: bool = True,
                 internal_weights: Optional[bool] = None, shared_weights: Optional[bool] = None,
                 normalization: Optional[str] = None):

        instructions = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(irreps_in1, irreps_in2, irreps_out,
                         instructions=instructions,
                         bias=bias, rescale=rescale,
                         internal_weights=internal_weights, shared_weights=shared_weights,
                         normalization=normalization)


def DepthwiseTensorProduct(irreps_node_input: o3.Irreps,
                           irreps_edge_attr: o3.Irreps,
                           irreps_node_output: o3.Irreps,
                           internal_weights: bool = False,
                           bias: bool = True,
                           rescale: bool = True) -> TensorProductRescale:
    '''
        The irreps of output is pre-determined.
        `irreps_node_output` is used to get certain types of vectors.
    '''
    irreps_output = []
    instructions = []

    for i, (mul, ir_in) in enumerate(irreps_node_input):
        for j, (_, ir_edge) in enumerate(irreps_edge_attr):
            for ir_out in ir_in * ir_edge:
                if ir_out in irreps_node_output or ir_out == o3.Irrep(0, 1):
                    k = len(irreps_output)
                    irreps_output.append((mul, ir_out))
                    instructions.append((i, j, k, 'uvu', True))

    irreps_output = o3.Irreps(irreps_output)
    irreps_output, p, _ = sort_irreps_even_first(irreps_output) #irreps_output.sort()
    instructions = [(i_1, i_2, p[i_out], mode, train)
                    for i_1, i_2, i_out, mode, train in instructions]
    tp = TensorProductRescale(irreps_node_input, irreps_edge_attr,
                              irreps_output, instructions,
                              internal_weights=internal_weights,
                              shared_weights=internal_weights,
                              bias=bias, rescale=rescale)
    return tp



class LinearRS(FullyConnectedTensorProductRescale):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, bias: bool = True, rescale: bool = True):
        super().__init__(irreps_in, o3.Irreps('1x0e'), irreps_out,
                         bias=bias, rescale=rescale, internal_weights=True,
                         shared_weights=True, normalization=None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.ones_like(x[:, 0:1])
        out = self.forward_tp_rescale_bias(x, y)
        return out

def sort_irreps_even_first(irreps: o3.Irreps):
    Ret = collections.namedtuple("sort", ["irreps", "p", "inv"])
    out = [(ir.l, -ir.p, i, mul) for i, (mul, ir) in enumerate(irreps)]
    out = sorted(out)
    inv = tuple(i for _, _, i, _ in out)
    p = perm.inverse(inv)
    irreps = o3.Irreps([(mul, (l, -p)) for l, p, _, mul in out])
    return Ret(irreps, p, inv)

def get_mul_0(irreps: o3.Irreps) -> int:
    mul_0: int = 0
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            mul_0 += mul
    return mul_0


def get_norm_layer(norm_type):
    if norm_type == 'graph':
        return EquivariantGraphNorm
    elif norm_type == 'instance':
        return EquivariantInstanceNorm
    elif norm_type == 'layer':
        return EquivariantLayerNormV2
    elif norm_type == 'fast_layer':
        return EquivariantLayerNormFast
    elif norm_type is None:
        return None
    else:
        raise ValueError('Norm type {} not supported.'.format(norm_type))


class RadialProfile(nn.Module):
    """
    Simple FC Layer for radial function.

    Parameters
    ----------
    ch_list : List[int]
        number of fc neurons of each layer: [input_dim, h1_dim, h2_dim, ..., out_dim]
    """
    def __init__(self, ch_list: List[int], use_layer_norm: bool = True, use_offset: bool = True):
        super().__init__()
        modules = []
        input_channels = ch_list[0]
        for i in range(len(ch_list)):
            if i == 0:
                continue
            if (i == len(ch_list) - 1) and use_offset:
                use_biases = False
            else:
                use_biases = True
            modules.append(nn.Linear(input_channels, ch_list[i], bias=use_biases))
            input_channels = ch_list[i]

            if i == len(ch_list) - 1:
                break

            if use_layer_norm:
                modules.append(nn.LayerNorm(ch_list[i]))
            #modules.append(nn.ReLU())
            #modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[torch.nn.functional.silu]))
            #modules.append(Activation(o3.Irreps('{}x0e'.format(ch_list[i])),
            #    acts=[ShiftedSoftplus()]))
            modules.append(torch.nn.SiLU())

        self.net = nn.Sequential(*modules)

        self.offset = None
        if use_offset:
            self.offset = nn.Parameter(torch.zeros(ch_list[-1]))
            fan_in = ch_list[-2]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.offset, -bound, bound)


    def forward(self, f_in: torch.Tensor) -> torch.Tensor:
        f_out: torch.Tensor = self.net(f_in)
        if self.offset is not None:
            f_out = f_out + self.offset.reshape(1, -1)
        return f_out

def irreps2gate(irreps: o3.Irreps) -> Tuple[o3.Irreps, o3.Irreps, o3.Irreps]:
    """
    Parameters
    ----------
    irreps : `e3nn.o3.Irreps`
        representation of the input

    Returns
    ----------
    irreps_scalars : `e3nn.o3.Irreps`
        scalar parts of input irreps (L=0)
    irreps_gates : `e3nn.o3.Irreps`
        another scalar irreps (L=0) that will serve as a gate to irreps_gated (L>0)
    irreps_gated : `e3nn.o3.Irreps`
        non-scalar parts of input irreps (L>0)

    Examples
    --------
    >>> irreps_scalars, irreps_gates, irreps_gated = irreps2gate("3x0e+4x1e+5x2e+20x0e")
    >>> irreps_scalars
    23x0e
    >>> irreps_gates
    9x0e
    >>> irreps_gated
    4x1e+5x2e


    >>> irreps_scalars, irreps_gates, irreps_gated = irreps2gate("3x0e+20x0e")
    >>> irreps_scalars
    23x0e
    >>> irreps_gates
    <empty o3.Irreps>
    >>> irreps_gated
    <empty o3.Irreps>

    """
    irreps_scalars = []
    irreps_gated = []
    for mul, ir in irreps:
        if ir.l == 0 and ir.p == 1:
            irreps_scalars.append((mul, ir))
        else:
            irreps_gated.append((mul, ir))
    irreps_scalars = o3.Irreps(irreps_scalars).simplify()
    irreps_gated = o3.Irreps(irreps_gated).simplify()
    if irreps_gated.dim > 0:
        ir = '0e'
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()
    return irreps_scalars, irreps_gates, irreps_gated


class SeparableFCTP(torch.nn.Module):
    '''
        Use separable FCTP for spatial convolution.

        DTP + RadialFC + Linear (+ LayerNorm + Gate)

        Parameters
        ----------
        fc_neurons : list of function or None
            list of activation functions, `None` if non-scalar or identity
    '''
    def __init__(self, irreps_node_input: o3.Irreps, irreps_edge_attr: o3.Irreps, irreps_node_output: o3.Irreps,
                 fc_neurons: Optional[List[int]], use_activation: bool = False, norm_layer: Optional[str] = None,
                 internal_weights: bool = False):

        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)

        norm = get_norm_layer(norm_layer)

        self.dtp: TensorProductRescale = DepthwiseTensorProduct(self.irreps_node_input,
                                                                self.irreps_edge_attr,
                                                                self.irreps_node_output,
                                                                bias=False,
                                                                internal_weights=internal_weights)

        self.dtp_rad = None
        if fc_neurons is not None:
            self.dtp_rad = RadialProfile(fc_neurons + [self.dtp.tp.weight_numel]) # Simple Linear layer for radial function. Each layer dim is: [fc_neuron1 (input), fc_neuron2, ..., weight_numel (output)]
            for (slice, slice_sqrt_k) in self.dtp.slices_sqrt_k.values():  # Seems to be for normalization
                self.dtp_rad.net[-1].weight.data[slice, :] *= slice_sqrt_k # Seems to be for normalization
                self.dtp_rad.offset.data[slice] *= slice_sqrt_k            # Seems to be for normalization

        irreps_lin_output: o3.Irreps = self.irreps_node_output
        irreps_scalars, irreps_gates, irreps_gated = irreps2gate(self.irreps_node_output)
        if use_activation:
            irreps_lin_output: o3.Irreps = irreps_scalars + irreps_gates + irreps_gated
            irreps_lin_output: o3.Irreps = irreps_lin_output.simplify()
        self.lin = LinearRS(self.dtp.irreps_out.simplify(), irreps_lin_output)

        self.norm = None
        if norm_layer is not None:
            self.norm = norm(self.lin.irreps_out)

        self.gate = None
        if use_activation:
            if irreps_gated.num_irreps == 0: # use typical scalar activation if irreps_out is all scalar (L=0)
                gate = Activation(self.irreps_node_output, acts=[torch.nn.SiLU() for _ in self.irreps_node_output])
            else: # use gate nonlinearity if there are non-scalar (L>0) components in the irreps_out.
                gate = Gate(
                    irreps_scalars, [torch.nn.SiLU() for _ in irreps_scalars],  # scalar
                    irreps_gates, [torch.sigmoid for _ in irreps_gates],  # gates (scalars)
                    irreps_gated  # gated tensors
                )
            self.gate = gate


    def forward(self, node_input: torch.Tensor, edge_attr: torch.Tensor, edge_scalars: Optional[torch.Tensor],
                batch: Optional[torch.Tensor] = None) -> torch.Tensor: # Batch does nothing if you use EquivLayernormV2
        '''
            Depthwise TP: `node_input` TP `edge_attr`, with TP parametrized by
            self.dtp_rad(`edge_scalars`).
        '''
        if self.dtp_rad is not None and edge_scalars is not None:
            weight = self.dtp_rad(edge_scalars)
        else:
            weight = None
        out = self.dtp(node_input, edge_attr, weight)
        out = self.lin(out)
        if self.norm is not None:
            out = self.norm(out, batch=batch)
        if self.gate is not None:
            out = self.gate(out)
        return out


#@compile_mode('script')
class Vec2AttnHeads(torch.nn.Module):
    '''
        Reshape vectors of shape [N, irreps_mid] to vectors of shape
        [N, num_heads, irreps_head].
    '''
    def __init__(self, irreps_head: o3.Irreps, num_heads: int):
        super().__init__()
        self.num_heads: int = num_heads
        self.irreps_head: o3.Irreps = irreps_head
        self.irreps_mid_in = []
        for mul, ir in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for mul, ir in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
        self.mid_in_indices = tuple(self.mid_in_indices)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _ = x.shape
        out = []
        for start_idx, end_idx in self.mid_in_indices:
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, temp.shape[-1] // self.num_heads)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out


    def __repr__(self):
        return '{}(irreps_head={}, num_heads={})'.format(
            self.__class__.__name__, self.irreps_head, self.num_heads)


#@compile_mode('script')
class AttnHeads2Vec(torch.nn.Module):
    '''
        Convert vectors of shape [N, num_heads, irreps_head] into
        vectors of shape [N, irreps_head * num_heads].
    '''
    def __init__(self, irreps_head: o3.Irreps):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for mul, ir in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim
        self.head_indices = tuple(self.head_indices)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, _ = x.shape
        out = []
        for start_idx, end_idx in self.head_indices:
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out


    def __repr__(self):
        return '{}(irreps_head={})'.format(self.__class__.__name__, self.irreps_head)


class GraphAttentionMLP(torch.nn.Module):
    def __init__(self,
                 irreps_emb: o3.Irreps,
                 irreps_edge_attr: o3.Irreps,
                 irreps_node_output: o3.Irreps,
                 mul_alpha,
                 fc_neurons: List[int],
                 irreps_head: o3.Irreps,
                 num_heads: int,
                 alpha_drop: float = 0.1,
                 proj_drop: float = 0.1,
                 debug: bool = False):
        self.debug = debug

        super().__init__()
        self.irreps_emb = o3.Irreps(irreps_emb)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_head: o3.Irreps = o3.Irreps(irreps_head)
        self.num_heads: int = num_heads

        irreps_attn_heads: o3.Irreps = irreps_head * num_heads
        irreps_attn_heads, _, _ = sort_irreps_even_first(irreps_attn_heads) #irreps_attn_heads.sort()
        irreps_attn_heads: o3.Irreps = irreps_attn_heads.simplify()# how many 0e per head # for attention score
        mul_alpha_head: int = mul_alpha // num_heads  # how many 0e per head
        irreps_alpha: o3.Irreps = o3.Irreps('{}x0e'.format(mul_alpha))
        # Use an extra separable FCTP and Swish Gate for value
        self.sep_act = SeparableFCTP(irreps_node_input = self.irreps_emb,
                                     irreps_edge_attr = self.irreps_edge_attr,
                                     irreps_node_output = self.irreps_emb,
                                     fc_neurons = fc_neurons,
                                     use_activation = True,
                                     norm_layer = None,
                                     internal_weights = False)
        self.sep_alpha = LinearRS(self.sep_act.dtp.irreps_out, irreps_alpha)
        self.sep_value = SeparableFCTP(irreps_node_input = self.irreps_emb,
                                       irreps_edge_attr = self.irreps_edge_attr,
                                       irreps_node_output = irreps_attn_heads,
                                       fc_neurons = None,
                                       use_activation = False,
                                       norm_layer = None,
                                       internal_weights = True)
        self.vec2heads_alpha = Vec2AttnHeads(irreps_head = o3.Irreps('{}x0e'.format(mul_alpha_head)),
                                             num_heads = num_heads)
        self.vec2heads_value = Vec2AttnHeads(irreps_head = self.irreps_head,
                                             num_heads = num_heads)

        self.alpha_act = Activation(irreps_in = o3.Irreps('{}x0e'.format(mul_alpha_head)),
                                    acts = [SmoothLeakyReLU(0.2)])
        self.heads2vec = AttnHeads2Vec(irreps_head = irreps_head)

        self.mul_alpha_head = mul_alpha_head
        self.alpha_dot = torch.nn.Parameter(torch.randn(1, num_heads, mul_alpha_head))
        torch.nn.init.xavier_uniform_(self.alpha_dot) # Following GATv2

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        self.proj = LinearRS(irreps_in = irreps_attn_heads,
                             irreps_out = self.irreps_node_output)



    def forward(self, message: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_scalars: torch.Tensor,neighbor_num) -> torch.Tensor:

        weight: torch.Tensor = self.sep_act.dtp_rad(edge_scalars)
        message: torch.Tensor = self.sep_act.dtp(message, edge_attr, weight)
        log_alpha = self.sep_alpha(message)                        # f_ij^(L=0) part  ||  Linear: irreps_in -> 'mul_alpha x 0e'
        log_alpha = self.vec2heads_alpha(log_alpha)                    # reshape (N, Heads*head_dim) -> (N, Heads, head_dim)
        value: torch.Tensor = self.sep_act.lin(message)                      # f_ij^(L>=0) part (before activation)
        value: torch.Tensor = self.sep_act.gate(value)                       # f_ij^(L>=0) part (after activation)
        value: torch.Tensor = self.sep_value(value, edge_attr=edge_attr, edge_scalars=edge_scalars) # DTP + Linear for f_ij^(L>=0) part
        value: torch.Tensor = self.vec2heads_value(value)                    # reshape (N, Heads*head_dim) -> (N, Heads, head_dim)
        # inner product
        log_alpha = self.alpha_act(log_alpha)          # Leaky ReLU
        log_alpha = torch.einsum('ehk, hk -> eh', log_alpha, self.alpha_dot.squeeze(0)) # Linear layer: (N_edge, N_head mul_alpha_head) -> (N_edge, N_head)
        log_alpha=log_alpha.reshape(-1,neighbor_num,self.num_heads)
        alpha=F.softmax(log_alpha,dim=1).unsqueeze(-1)
        head_fea_dim=value.shape[-1]
        value=value.reshape(-1,neighbor_num,self.num_heads,head_fea_dim)
        attn: torch.Tensor = value * alpha                                     # (N_edge, N_head, head_dim)
        attn: torch.Tensor = torch.sum(attn,dim=1)
        attn: torch.Tensor = self.heads2vec(attn)

        node_output: torch.Tensor = self.proj(attn) # Final Linear layer.


        return node_output


    def extra_repr(self):
        output_str = super().extra_repr()
        return output_str






