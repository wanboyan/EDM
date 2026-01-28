import math
import torch
import torch.nn.functional as F
from torch import nn, einsum 

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape

from rotary_embedding_torch import RotaryEmbedding

from my_diff_utils.model_utils import *
from nfmodel.vec_layers import VNLinear,VNFeedForward
from random import sample
import absl.flags as flags
FLAGS = flags.FLAGS

class CausalTransformer(nn.Module):
    def __init__(
        self,
        dim, 
        depth=4,
        dim_in_out=None,
        cross_attn=True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True, 
        attn_dropout = 0.,
        ff_dropout = 0.3,
        final_proj = True, 
        normformer = False,
        rotary_emb = True,
        point_feature_dim =0
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross =  None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)

        if cross_attn:
            #print("using CROSS ATTN, with dropout {}".format(attn_dropout))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention(dim = dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
        else:
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, out_dim=dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention(dim = dim, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                    Attention(dim = dim, out_dim=dim_in_out, causal = True, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                    FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        # self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

        self.to_vec=to_vec(dim)
        self.cross_attn = cross_attn
        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)


    def forward(self, x, time_emb=None, context=None):
        t_emb,x_sample,learned_emb=x
        n, device = len(x), t_emb.device

        t_emb=t_emb[:,:,None,:].repeat(1,1,12,1)
        learned_emb=learned_emb[:,:,None,:].repeat(1,1,12,1)
        x_sample=torch.einsum('bqi,ri->brq',x_sample,self.vs)[:,None,:,:]
        x = torch.cat([t_emb,x_sample,learned_emb],dim=1)
        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        if self.cross_attn:
            #assert context is not None 
            for idx, (self_attn, cross_attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) # removing attn_bias for now 
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x 
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now 
                #print("x2 shape, context shape: ", x.shape, context.shape)
                
                #print("x3 shape, context shape: ", x.shape, context.shape)
                x = ff(x) + x
        
        else:
            for idx, (attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = attn(x, attn_bias = attn_bias)
                else:
                    x = attn(x, attn_bias = attn_bias) + x
                #print("x2 shape: ", x.shape)
                x = ff(x) + x
                #print("x3 shape: ", x.shape)

        out = self.norm(x)

        return self.to_vec(out,self.vs)
        # return self.project_out(out)



class to_vec(nn.Module):
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

        q=self.fc(x)
        atten=F.softmax(q,dim=2)
        x=atten*x
        x=x.permute(0,1,3,2)
        out=x.unsqueeze(-1).expand(-1,-1,-1,-1,3)*(vs.float())
        out=out.sum(-2)
        out=self.final_fc(out)
        return out






class DiffusionNet(nn.Module):

    def __init__(
        self,
        dim=256,
        dim_in_out=None,
        num_timesteps = None,
        num_time_embeds = 1,
        cond = True,
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = True
        self.cond_dropout = True
        self.point_feature_dim = 128

        self.dim_in_out = default(dim_in_out, dim)
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query = nn.Parameter(torch.randn(self.dim_in_out))
        self.causal_transformer = CausalTransformer(dim = dim, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim)




    def forward(
        self,
        data, 
        diffusion_timesteps,
        pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        if self.cond:
            assert type(data) is tuple
            data, cond = data # adding noise to cond_feature so doing this in diffusion.py

            #print("data, cond shape: ", data.shape, cond.shape) # B, dim_in_out; B, N, 3
            #print("pass cond: ", pass_cond)
            if self.cond_dropout:
                # classifier-free guidance: 20% unconditional 
                mask=torch.bernoulli(torch.zeros_like(cond[:,0,0,0])+0.8)
                cond_feature=torch.einsum('b,b q r d-> b q r d',mask,cond)
            else:
                cond_feature = cond

            
        batch, dim, _,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds(diffusion_timesteps)

        data = data

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        model_inputs = [time_embed, data, learned_queries]





        cond_feature = None if not self.cond else cond_feature
        #print("tokens shape: ", tokens.shape, cond_feature.shape)
        tokens = self.causal_transformer(model_inputs, context=cond_feature)


        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :,:]

        return pred


class CausalTransformer_v2(nn.Module):
    def __init__(
            self,
            dim,
            depth=4,
            dim_in_out=None,
            cross_attn=True,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            norm_in = False,
            norm_out = True,
            attn_dropout = 0.,
            ff_dropout = 0.3,
            final_proj = True,
            normformer = False,
            rotary_emb = True,
            point_feature_dim =0
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross =  None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)

        if cross_attn:
            #print("using CROSS ATTN, with dropout {}".format(attn_dropout))
            self.layers.append(nn.ModuleList([
                Attention_v2(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention_v2(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention_v2(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention_v2(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                Attention_v2(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention_v2(dim = dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options



        self.cross_attn = cross_attn



    def forward(self, x, context=None):

        n, device = x.shape[1], x.device


        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        if self.cross_attn:
            #assert context is not None
            for idx, (self_attn, cross_attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) # removing attn_bias for now
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now
                #print("x2 shape, context shape: ", x.shape, context.shape)

                #print("x3 shape, context shape: ", x.shape, context.shape)
                x = ff(x) + x

        else:
            for idx, (attn, ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = attn(x, attn_bias = attn_bias)
                else:
                    x = attn(x, attn_bias = attn_bias) + x
                #print("x2 shape: ", x.shape)
                x = ff(x) + x
                #print("x3 shape: ", x.shape)

        out = self.norm(x)

        return out
        # return self.project_out(out)



class CausalTransformer_v3(nn.Module):
    def __init__(
            self,
            dim,
            depth=4,
            dim_in_out=None,
            cross_attn=True,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            norm_in = False,
            norm_out = True,
            attn_dropout = 0.,
            ff_dropout = 0.3,
            final_proj = True,
            normformer = False,
            rotary_emb = True,
            point_feature_dim =0
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross =  None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)
        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        if cross_attn:
            #print("using CROSS ATTN, with dropout {}".format(attn_dropout))
            self.layers.append(nn.ModuleList([
                Attention_v2(dim = dim_in_out, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention_v2(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                Attention_v3(dim = dim, causal = False, dim_head = dim_head, heads = heads, rotary_emb = None),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention_v2(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention_v2(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    Attention_v3(dim = dim, causal = False, dim_head = dim_head, heads = heads, rotary_emb = None),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                Attention_v2(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention_v2(dim = dim, kv_dim=point_feature_dim, out_dim=dim_in_out, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                Attention_v3(dim = dim, causal = False, dim_head = dim_head, heads = heads, rotary_emb = None),
                FeedForward(dim = dim_in_out, out_dim=dim_in_out, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options



        self.cross_attn = cross_attn



    def forward(self, x, context=None):

        n, device = x.shape[1], x.device


        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        if self.cross_attn:
            #assert context is not None
            for idx, (self_attn, cross_attn, chanel_atten,ff) in enumerate(self.layers):
                #print("x1 shape: ", x.shape)
                if (idx==0 or idx==len(self.layers)-1) and not self.use_same_dims:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) # removing attn_bias for now
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now
                if FLAGS.tmp=='v0':
                    fuse_x=chanel_atten(x)+x
                    x = ff(fuse_x) + x
                    # print('---v0')
                if FLAGS.tmp=='v1':
                    fuse_x=chanel_atten(x)+x
                    x = ff(fuse_x) + fuse_x
                    # print('---v1')
                if FLAGS.tmp=='v2':
                    x=ff(x)+x
                    # print('---v2')







        out = self.norm(x)

        return out
        # return self.project_out(out)







class DiffusionNet_ddm(nn.Module):

    def __init__(
            self,
            dim=256,
            dim_in_out=None,
            num_timesteps = None,
            num_time_embeds = 1,
            cond = True,
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = True
        self.cond_dropout = True
        self.point_feature_dim = 128

        self.dim_in_out = default(dim_in_out, dim)
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query = nn.Parameter(torch.randn(self.dim_in_out))

        self.causal_transformer_latent = CausalTransformer_v2(dim = dim, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim)
        self.causal_transformer_ts = CausalTransformer_v2(dim = dim, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim+1)

        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        self.to_vec_latent=to_vec(dim)
        self.to_vec_ts=to_vec(dim)
        self.ts_linear=nn.Linear(2,dim)

        self.fc_inv = VNLinear(dim, dim)
        self.fc_out_so3 = VNLinear(dim, 1)
        self.fc_out_inv = nn.Linear(dim, 1)


    def forward_latent(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        if self.cond:
            assert type(data) is tuple
            data, cond = data # adding noise to cond_feature so doing this in diffusion.py

            #print("data, cond shape: ", data.shape, cond.shape) # B, dim_in_out; B, N, 3
            #print("pass cond: ", pass_cond)
            if self.cond_dropout:
                # classifier-free guidance: 20% unconditional
                mask=torch.bernoulli(torch.zeros_like(cond[:,0,0,0])+0.8)
                cond_feature=torch.einsum('b,b q r d-> b q r d',mask,cond)
            else:
                cond_feature = cond


        batch, dim, _,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds(diffusion_timesteps)

        data = data

        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)

        model_inputs = [time_embed, data, learned_queries]

        t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
        learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
        x_sample=torch.einsum('bqi,ri->brq',data,self.vs)[:,None,:,:]
        x = torch.cat([t_emb,x_sample,learned_emb],dim=1)



        cond_feature = None if not self.cond else cond_feature
        #print("tokens shape: ", tokens.shape, cond_feature.shape)
        tokens = self.causal_transformer_latent(x, context=cond_feature)
        tokens=self.to_vec_latent(tokens,self.vs)

        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :,:]

        return pred


    def forward_ts(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        assert type(data) is tuple
        latent,trans,scale, cond ,point= data # adding noise to cond_feature so doing this in diffusion.py



        batch, dim, _,device, dtype = *latent.shape, latent.device, latent.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds(diffusion_timesteps)


        learned_queries = repeat(self.learned_query, 'd -> b 1 d', b = batch)


        point_emb=torch.einsum('bqi,ri->bqr',point,self.vs)[:,:,:,None]
        cond_feature=torch.cat([point_emb,cond],dim=-1)

        t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
        learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
        latent_emb=torch.einsum('bqi,ri->brq',latent,self.vs)[:,None,:,:]
        trans_emb=torch.einsum('bi,ri->br',trans,self.vs)[:,None,:]
        scale_emb=scale[:,None,None].repeat(1,1,12)
        ts_emb=torch.cat([trans_emb,scale_emb],dim=1)
        ts_emb=self.ts_linear(ts_emb.permute(0,2,1))[:,None,:,:]
        x = torch.cat([t_emb, latent_emb,ts_emb, learned_emb],dim=1)


        tokens = self.causal_transformer_ts(x, context=cond_feature)
        tokens=self.to_vec_latent(tokens,self.vs)
        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        features = tokens[..., -1, :,:]
        dual_features=self.fc_inv(features)
        inv_fea= (features * dual_features).sum(-1)
        out_scale=self.fc_out_inv(inv_fea).squeeze(1)
        out_trans=self.fc_out_so3(features).squeeze(1)

        return out_trans,out_scale


class DiffusionNet_v3(nn.Module):

    def __init__(
            self,
            dim=256,
            dim_in_out=None,
            num_timesteps = None,
            num_time_embeds = 1,
            cond = True,
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = True
        self.cond_dropout = True
        self.point_feature_dim = 128

        self.dim_in_out = default(dim_in_out, dim)
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds_latent = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        self.to_time_embeds_ts = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query_latent = nn.Parameter(torch.randn(self.dim_in_out))
        self.learned_query_ts = nn.Parameter(torch.randn(self.dim_in_out))
        if FLAGS.mid_con:
            print('---using transformer_v3')
            self.causal_transformer_latent = CausalTransformer_v3(dim = dim, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim)
            self.causal_transformer_ts = CausalTransformer_v3(dim = dim, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim+1)
        else:
            self.causal_transformer_latent = CausalTransformer_v2(dim = dim, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim)
            self.causal_transformer_ts = CausalTransformer_v2(dim = dim, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim+1)

        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)

        self.to_vec_latent=to_vec(dim)
        self.to_vec_ts=to_vec(dim)

        self.ts_linear=nn.Linear(2,dim)
        self.fc_inv = VNLinear(dim, dim)
        self.fc_out_so3 = VNLinear(dim, 1)
        self.fc_out_inv = nn.Linear(dim, 1)


    def forward_latent(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        if self.cond:
            assert type(data) is tuple
            data, cond = data # adding noise to cond_feature so doing this in diffusion.py

            #print("data, cond shape: ", data.shape, cond.shape) # B, dim_in_out; B, N, 3
            #print("pass cond: ", pass_cond)
            if self.cond_dropout:
                # classifier-free guidance: 20% unconditional
                mask=torch.bernoulli(torch.zeros_like(cond[:,0,0,0])+0.8)
                cond_feature=torch.einsum('b,b q r d-> b q r d',mask,cond)
            else:
                cond_feature = cond


        batch, dim, _,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_latent(diffusion_timesteps)

        data = data

        learned_queries = repeat(self.learned_query_latent, 'd -> b 1 d', b = batch)

        model_inputs = [time_embed, data, learned_queries]

        t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
        learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
        x_sample=torch.einsum('bqi,ri->brq',data,self.vs)[:,None,:,:]
        x = torch.cat([t_emb,x_sample,learned_emb],dim=1)



        cond_feature = None if not self.cond else cond_feature
        #print("tokens shape: ", tokens.shape, cond_feature.shape)
        tokens = self.causal_transformer_latent(x, context=cond_feature)
        tokens = self.to_vec_latent(tokens,self.vs)

        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :,:]

        return pred


    def forward_ts(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        assert type(data) is tuple
        latent,trans,scale, cond ,point= data # adding noise to cond_feature so doing this in diffusion.py



        batch, dim, _,device, dtype = *latent.shape, latent.device, latent.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_ts(diffusion_timesteps)


        learned_queries = repeat(self.learned_query_ts, 'd -> b 1 d', b = batch)


        point_emb=torch.einsum('bqi,ri->bqr',point,self.vs)[:,:,:,None]
        cond_feature=torch.cat([point_emb,cond],dim=-1)

        t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
        learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
        latent_emb=torch.einsum('bqi,ri->brq',latent,self.vs)[:,None,:,:]
        trans_emb=torch.einsum('bi,ri->br',trans,self.vs)[:,None,:]
        scale_emb=scale[:,None,None].repeat(1,1,12)
        ts_emb=torch.cat([trans_emb,scale_emb],dim=1)

        ts_emb=self.ts_linear(ts_emb.permute(0,2,1))[:,None,:,:]
        x = torch.cat([t_emb, latent_emb,ts_emb, learned_emb],dim=1)


        tokens = self.causal_transformer_ts(x, context=cond_feature)
        tokens=self.to_vec_ts(tokens,self.vs)
        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        features = tokens[..., -1, :,:]
        dual_features=self.fc_inv(features)
        inv_fea= (features * dual_features).sum(-1)
        out_scale=self.fc_out_inv(inv_fea).squeeze(1)
        out_trans=self.fc_out_so3(features).squeeze(1)

        return out_trans,out_scale