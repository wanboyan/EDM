import math
import torch
import torch.nn.functional as F
from torch import nn, einsum 

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape

from .pointnet.pointnet_classifier import PointNetClassifier
from .pointnet.conv_pointnet import ConvPointnet
from .pointnet.dgcnn import DGCNN
from .helpers import *

import absl.flags as flags
FLAGS = flags.FLAGS

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5, stable = False):
        super().__init__()
        self.eps = eps
        self.stable = stable
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if self.stable:
            x = x / x.amax(dim = -1, keepdim = True).detach()

        var = torch.var(x, dim = -1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = -1, keepdim = True)
        return (x - mean) * (var + self.eps).rsqrt() * self.g

# mlp

class MLP(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        *,
        expansion_factor = 2.,
        depth = 2,
        norm = False,
    ):
        super().__init__()
        hidden_dim = int(expansion_factor * dim_out)
        norm_fn = lambda: nn.LayerNorm(hidden_dim) if norm else nn.Identity()

        layers = [nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.SiLU(),
            norm_fn()
        )]

        for _ in range(depth - 1):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                norm_fn()
            ))

        layers.append(nn.Linear(hidden_dim, dim_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

# relative positional bias for causal transformer
def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

class DIT_MLP(nn.Module):
    def __init__(self, *,
                 width: int,
                 init_scale: float):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class RelPosBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        num_buckets = 32,
        max_distance = 128
    ):
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        return torch.where(is_small, n, val_if_large)

    def forward(self, i, j, *, device):
        q_pos = torch.arange(i, dtype = torch.long, device = device)
        k_pos = torch.arange(j, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

# feedforward

class SwiGLU(nn.Module):
    """ used successfully in https://arxiv.org/abs/2204.0231 """
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.silu(gate)

def FeedForward(
    dim,
    out_dim = None,
    mult = 4,
    dropout = 0.,
    post_activation_norm = False
):
    """ post-activation norm https://arxiv.org/abs/2110.09456 """

    #print("dropout: ", dropout)
    out_dim = default(out_dim, dim)
    #print("out_dim: ", out_dim)
    inner_dim = int(mult * dim)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        SwiGLU(),
        LayerNorm(inner_dim) if post_activation_norm else nn.Identity(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, out_dim, bias = False)
    )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            *,
            out_dim = None,
            dim_head = 64,
            heads = 8,
            dropout = 0.,
            causal = False,
            rotary_emb = None,
            pb_relax_alpha = 128
    ):
        super().__init__()
        self.pb_relax_alpha = pb_relax_alpha
        self.scale = dim_head ** -0.5 * (pb_relax_alpha ** -1)

        self.heads = heads
        inner_dim = dim_head * heads
        kv_dim = default(kv_dim, dim)

        self.causal = causal

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(kv_dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        out_dim = default(out_dim, dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim, bias = False),
            LayerNorm(out_dim)
        )

    def forward(self, x, context=None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device

        context = default(context, x) #self attention if context is None

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)
        q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)
        #print("attn bias, sim shapes: ", attn_bias.shape, sim.shape)
        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        sim = sim * self.pb_relax_alpha

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Attention_v2(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            *,
            out_dim = None,
            dim_head = 64,
            heads = 8,
            dropout = 0.,
            causal = False,
            rotary_emb = None,
            pb_relax_alpha = 128
    ):
        super().__init__()
        self.pb_relax_alpha = pb_relax_alpha
        self.scale = dim_head ** -0.5 * (pb_relax_alpha ** -1)

        self.heads = heads
        inner_dim = dim_head * heads
        kv_dim = default(kv_dim, dim)

        self.causal = causal

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(kv_dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        out_dim = default(out_dim, dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim, bias = False),
            LayerNorm(out_dim)
        )

    def forward(self, x, context=None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device
        mult=x.shape[2]


        context = default(context, x) #self attention if context is None

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q = rearrange(q, 'b n r (h d) -> b h n r d', h = self.heads)
        if FLAGS.fix_attention and not FLAGS.a_sim:
            q = q * self.scale * (mult ** -0.5)
        else:
            q = q * self.scale

        # rotary embeddings

        if exists(self.rotary_emb):
            q=rearrange(q,'b h n r d -> b h r n d')
            k=rearrange(k,'b n r d -> b r n d')
            q, k = map(self.rotary_emb.rotate_queries_or_keys, (q, k))
            q=rearrange(q,'b h r n d -> b h n r d')
            k=rearrange(k,'b r n d -> b n r d')

        # add null key / value for classifier free guidance in prior net

        if FLAGS.fea_type=='a12':
            nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 12 d', b = b)
        else:
            nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 5 d', b = b)

        k = torch.cat((nk, k), dim = 1)
        v = torch.cat((nv, v), dim = 1)

        # calculate query / key similarities
        if FLAGS.a_sim:
            sim = einsum('b h i r d, b j r d -> b r h i j', q, k)
        else:
            sim = einsum('b h i r d, b j r d -> b h i j', q, k)


        # relative positional encoding (T5 style)
        #print("attn bias, sim shapes: ", attn_bias.shape, sim.shape)
        if exists(attn_bias):
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        # attention

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        sim = sim * self.pb_relax_alpha

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate values
        if FLAGS.a_sim:
            out = einsum('b r h i j, b j r d -> b h i r d', attn, v)
        else:
            out = einsum('b h i j, b j r d -> b h i r d', attn, v)

        out = rearrange(out, 'b h n r d -> b n r (h d)')
        return self.to_out(out)

class Attention_v3(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            *,
            out_dim = None,
            dim_head = 64,
            heads = 8,
            multi=1,
            dropout = 0.,
            causal = False,
            rotary_emb = None,
            pb_relax_alpha = 128
    ):
        super().__init__()
        self.pb_relax_alpha = pb_relax_alpha
        self.scale = dim_head ** -0.5 * (pb_relax_alpha ** -1)

        self.heads = heads
        inner_dim = dim_head * heads
        kv_dim = default(kv_dim, dim)

        self.causal = causal

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(kv_dim, dim_head * 2, bias = False)

        self.rotary_emb = rotary_emb

        out_dim = default(out_dim, dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim, bias = False),
            LayerNorm(out_dim)
        )

    def forward(self, x, context=None):
        b, n, device = *x.shape[:2], x.device


        context = default(context, x) #self attention if context is None

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        q = rearrange(q, 'b n r (h d) -> b n h r d', h = self.heads)
        q = q * self.scale



        # calculate query / key similarities

        sim = einsum('b n h i d, b n j d -> b n h i j', q, k)



        # masking

        max_neg_value = -torch.finfo(sim.dtype).max


        # attention

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        sim = sim * self.pb_relax_alpha

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate values

        out = einsum('b n h i j, b n j d -> b n h i d', attn, v)

        out = rearrange(out, 'b n h i d -> b n i (h d)')
        return self.to_out(out)


class Attention_v2_cross(nn.Module):
    def __init__(
            self,
            dim,
            kv_dim=None,
            *,
            out_dim = None,
            dim_head = 64,
            heads = 8,
            dropout = 0.,
            causal = False,
            rotary_emb = None,
            pb_relax_alpha = 128
    ):
        super().__init__()
        self.pb_relax_alpha = pb_relax_alpha
        self.scale = dim_head ** -0.5 * (pb_relax_alpha ** -1)

        self.heads = heads
        inner_dim = dim_head * heads
        kv_dim = default(kv_dim, dim)

        self.causal = causal

        self.norm = LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias = False)

        self.rotary_emb = rotary_emb

        out_dim = default(out_dim, dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, out_dim, bias = False),
            LayerNorm(out_dim)
        )

    def forward(self, x, context=None, mask = None, attn_bias = None):
        b, n, device = *x.shape[:2], x.device
        mult=x.shape[2]
        if context is not None:
            use_null=False
        else:
            use_null=True
        context = default(context, x) #self attention if context is None

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))

        # q = rearrange(q, 'b n r (h d) -> b h n r d', h = self.heads)
        q,k,v = rearrange_many((q,k,v), 'b n r (h d) -> b h n r d', h = self.heads)
        if FLAGS.fix_attention and not FLAGS.a_sim:
            q = q * self.scale * (mult**-0.5)
        else:
            q = q * self.scale



        if FLAGS.fea_type=='a12':
            nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 12 d', b = b, h=self.heads)
        else:
            nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 5 d', b = b, h=self.heads)

        k = torch.cat((nk, k), dim = 2)
        v = torch.cat((nv, v), dim = 2)

        # calculate query / key similarities


        if FLAGS.a_sim:
            sim = einsum('b h i r d, b h j r d -> b r h i j', q, k)
        else:
            sim = einsum('b h i r d, b h j r d -> b h i j', q, k)


        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = F.pad(mask, (1, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)



        # attention

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        sim = sim * self.pb_relax_alpha

        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        # aggregate values



        if FLAGS.a_sim:
            out = einsum('b r h i j, b h j r d -> b h i r d', attn, v)
        else:
            out = einsum('b h i j, b h j r d -> b h i r d', attn, v)

        out = rearrange(out, 'b h n r d -> b n r (h d)')
        return self.to_out(out)



class Block(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            groups = 8
    ):
        super().__init__()
        self.project = nn.Linear(dim, dim_out)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.project(x)
        x = self.norm(x.permute(0,3,1,2)).permute(0,2,3,1)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x





class ResnetBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            *,
            cond_dim = None,
            time_cond_dim = None,
            groups = 8
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None
        if cond_dim:
            self.cross_attn =  Attention_v2_cross(
                        dim = dim_out,kv_dim=cond_dim
                    )


        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Linear(dim, dim_out) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None, cond = None):

        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)
            scale_shift = time_emb.chunk(2, dim = -1)

        h = self.block1(x, scale_shift = scale_shift)

        if exists(self.cross_attn):
            assert exists(cond)
            h = self.cross_attn(h, context = cond) + h

        h = self.block2(h)
        return h + self.res_conv(x)




class Attention_v4(nn.Module):
    def __init__(
            self,
            dim,
            init_scale,
            heads = 8,

    ):
        super().__init__()
        self.dim=dim
        self.attn_ch=dim//heads
        self.heads=heads
        if FLAGS.use_michel:
            qkv_bias=False
        else:
            qkv_bias=True

        self.c_qkv = nn.Linear(dim, dim * 3,qkv_bias)
        self.c_proj = nn.Linear(dim, dim)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        b, n, a,device = *x.shape[:3], x.device

        x = self.c_qkv(x)
        if FLAGS.a_sim:
            scale = 1 / math.sqrt(math.sqrt(self.attn_ch))
        else:
            scale = 1 / math.sqrt(math.sqrt(self.attn_ch*a))
        x= x.reshape(b,n,a,self.heads,-1)
        q,k,v=torch.split(x,self.attn_ch,dim=-1)

        q = rearrange(q, 'b n r h d -> b h n r d', h = self.heads)
        k = rearrange(k, 'b n r h d -> b h n r d', h = self.heads)
        v = rearrange(v, 'b n r h d -> b h n r d', h = self.heads)




        # calculate query / key similarities

        if FLAGS.a_sim:
            sim = einsum('b h i r d, b h j r d -> b r h i j', q* scale, k* scale)
        else:
            sim = einsum('b h i r d, b h j r d -> b h i j', q* scale, k* scale)



        attn = sim.softmax(dim = -1)

        # aggregate values
        if FLAGS.a_sim:
            out = einsum('b r h i j, b h j r d -> b h i r d', attn, v)
        else:
            out = einsum('b h i j, b h j r d -> b h i r d', attn, v)

        out = rearrange(out, 'b h n r d -> b n r (h d)')
        return self.c_proj(out)


class Attention_v5(nn.Module):
    def __init__(
            self,
            dim,
            init_scale,
            heads = 8,


    ):
        super().__init__()
        self.dim=dim
        self.attn_ch=dim//heads
        self.heads=heads
        if FLAGS.use_michel:
            qkv_bias=False
        else:
            qkv_bias=True

        self.c_qkv = nn.Linear(dim, dim * 3,qkv_bias)
        self.c_proj = nn.Linear(dim, dim)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):

        b, n, a,device = *x.shape[:3], x.device

        x = self.c_qkv(x)
        scale = 1 / math.sqrt(math.sqrt(self.attn_ch))
        x= x.reshape(b,n,a,self.heads,-1)
        q,k,v=torch.split(x,self.attn_ch,dim=-1)

        q = rearrange(q, 'b n r h d -> b n h r d', h = self.heads)
        k = rearrange(k, 'b n r h d -> b n h r d', h = self.heads)
        v = rearrange(v, 'b n r h d -> b n h r d', h = self.heads)



        # calculate query / key similarities

        sim = einsum('b n h i d, b n h j d -> b n h i j', q*scale , k*scale )



        attn = sim.softmax(dim = -1)

        # aggregate values

        out = einsum('b n h i j, b n h j d -> b n h i d', attn, v)

        out = rearrange(out, 'b n h i d -> b n i (h d)')
        return self.c_proj(out)






class pointe_MLP(nn.Module):
    def __init__(self, *, width: int, init_scale=0):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4,width)
        self.gelu = nn.GELU()
        # init_linear(self.c_fc, init_scale)
        # init_linear(self.c_proj, init_scale)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
