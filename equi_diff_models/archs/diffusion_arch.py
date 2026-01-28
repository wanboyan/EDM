import math
import torch
import torch.nn.functional as F
from torch import nn, einsum 

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many, check_shape
from eqnet.transformer.utils import *
from rotary_embedding_torch import RotaryEmbedding
from diffusers.models.embeddings import Timesteps
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
            cross_attn=False,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            norm_in = False,
            norm_out = True,
            attn_dropout = 0.,
            ff_dropout = 0.,
            final_proj = True,
            normformer = False,
            rotary_emb = True,
            **kwargs
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross = None

        self.layers = nn.ModuleList([])

        dim_in_out = default(dim_in_out, dim)
        self.use_same_dims = (dim_in_out is None) or (dim_in_out==dim)
        point_feature_dim = kwargs.get('point_feature_dim', dim)

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


        self.norm = LayerNorm(dim_in_out, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim_in_out, dim_in_out, bias = False) if final_proj else nn.Identity()

        self.cross_attn = cross_attn

    def forward(self, x, time_emb=None, context=None):
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
        return self.project_out(out)


class to_vec(nn.Module):
    def __init__(self, dim_out,dim_in=None):
        super().__init__()
        self.dim_out=dim_out
        if dim_in is not None:
            self.dim_in=dim_in
        else:
            self.dim_in=dim_out
        self.fc=nn.Linear(self.dim_in, self.dim_in, bias=True)
        self.final_fc=VNLinear(self.dim_in,self.dim_out)
    def forward(self,x,vs):

        q=self.fc(x)
        atten=F.softmax(q,dim=2)
        x=atten*x
        x=x.permute(0,1,3,2)
        out=x.unsqueeze(-1).expand(-1,-1,-1,-1,3)*(vs.float())
        out=out.sum(-2)
        out=self.final_fc(out)
        return out






class DiffusionNet_occ(nn.Module):

    def __init__(
            self,
            dim=512,
            dim_in_out=768,
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
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        if FLAGS.use_fuse and not FLAGS.use_simple:
            self.point_feature_dim = sum(self.dim_list[1:])
        else:
            self.point_feature_dim = self.dim_list[-1]
        self.dim_in_out= dim_in_out
        self.dim_in_out_ts=256
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds_latent = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        self.to_time_embeds_ts = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out_ts * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out_ts), MLP(self.dim_in_out_ts, self.dim_in_out_ts * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query_latent = nn.Parameter(torch.randn(self.dim_in_out))
        self.learned_query_ts = nn.Parameter(torch.randn(self.dim_in_out_ts))


        self.causal_transformer_latent = CausalTransformer_v4(dim = dim, in_out_dim=self.dim_in_out, point_feature_dim=self.point_feature_dim)

        self.causal_transformer_ts = CausalTransformer_v3(dim =self.dim_in_out_ts, dim_in_out=self.dim_in_out_ts, point_feature_dim=self.point_feature_dim+1)



        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        if FLAGS.fea_type=='a5':
            faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
                   (1,2,4),(2,7,8),(3,7,9),(3,5,11),
                   (4,5,6),(2,4,10),(2,8,10),(7,8,9),
                   (3,9,11),(5,6,11),(4,6,10),(0,8,10),
                   (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

            face_normal=vs_[faces,:].sum(1)
            face_normal=F.normalize(face_normal,dim=-1)
            self.register_buffer('face_normal',face_normal)

            face_to_cube=[(1,4,0,2,3),(2,0,1,4,3),(3,1,0,4,2),(4,2,0,3,1),
                          (0,3,1,2,4),(3,2,0,4,1),(4,3,0,2,1),(0,4,1,2,3),
                          (1,0,2,4,3),(2,1,0,4,3),(4,0,1,3,2),(0,1,2,3,4),
                          (1,2,0,3,4),(2,3,0,1,4),(3,4,0,1,2),(1,3,0,2,4),
                          (0,2,1,3,4),(4,1,0,3,2),(3,0,1,4,2),(2,4,0,1,3)]
            face_to_cube=torch.from_numpy(np.array(face_to_cube))
            self.register_buffer('face_to_cube',face_to_cube)



        self.cls_linear_latent=nn.Linear(6,self.dim_in_out)
        self.cls_linear_ts=nn.Linear(6,self.dim_in_out)


        self.x_in_ts=MLP(self.dim_in_out,self.dim_in_out_ts)
        self.r_in_ts=MLP(self.dim_in_out,self.dim_in_out_ts)
        self.ts_linear=nn.Linear(2,self.dim_in_out_ts)



        self.to_vec_latent=to_vec(self.dim_in_out)


        self.fc_inv_latent = VNLinear(self.dim_in_out, self.dim_in_out)
        self.fc_out_so3_latent = VNLinear(self.dim_in_out, self.dim_in_out)
        self.fc_out_inv_latent = nn.Linear(self.dim_in_out, self.dim_in_out)






        self.fc_inv = VNLinear(128, 128)
        self.fc_out_so3 = VNLinear(128, 1)
        self.fc_out_inv = nn.Linear(128, 1)

        self.to_vec_ts=to_vec(128,self.dim_in_out_ts)


    def forward_latent(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):


        data, r_vector,cond,cls_vector = data
        cond_feature = cond


        batch, dim,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_latent(diffusion_timesteps)

        data = data

        learned_queries = repeat(self.learned_query_latent, 'd -> b 1 d', b = batch)

        if FLAGS.fea_type=='a12':
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            r_vector=torch.einsum('bqi,ri->brq',r_vector,self.vs)[:,None,:,:]
            x_sample=data[:,None,None,:].repeat(1,1,12,1)

        else:
            raise  NotImplementedError
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,5,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            x_sample=self.to_a5(data).permute(0,2,1)[:,None,:,:]


        if FLAGS.per_cat=='combine':
            cls_vector=self.cls_linear_latent(cls_vector)
            x = torch.cat([t_emb,cls_vector,x_sample,r_vector,learned_emb],dim=1)
        else:
            x = torch.cat([t_emb,x_sample,r_vector,learned_emb],dim=1)



        tokens = self.causal_transformer_latent(x, context=cond_feature)
        if FLAGS.fea_type=='a12':
            tokens = self.to_vec_latent(tokens,self.vs)
        else:
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=features



        features = tokens[..., -1, :,:]

        dual_features=self.fc_inv_latent(features)
        inv_fea= (features * dual_features).sum(-1)
        out_x=self.fc_out_inv_latent(inv_fea).squeeze(1)
        out_r=self.fc_out_so3_latent(features).squeeze(1)

        return out_x,out_r

    def to_a5(self,x):
        x_face=torch.einsum('f i , b n i -> b n f',self.face_normal,x)
        x_face=torch.tanh(x_face)
        color=torch.zeros_like(x_face)[:,:,:1].repeat(1,1,5)
        b,n,_=x_face.shape
        color.scatter_add_(-1,self.face_to_cube[:,0][None,None,:].repeat(b,n,1),x_face)
        return color

    def forward_ts(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        assert type(data) is tuple
        latent,r_vector,trans,scale, cond ,point,cls_vector= data # adding noise to cond_feature so doing this in diffusion.py



        batch, dim,device, dtype = *latent.shape, latent.device, latent.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_ts(diffusion_timesteps)


        learned_queries = repeat(self.learned_query_ts, 'd -> b 1 d', b = batch)

        if FLAGS.fea_type=='a12':
            point_emb=torch.einsum('bqi,ri->bqr',point,self.vs)[:,:,:,None]
            r_vector=torch.einsum('bqi,ri->brq',r_vector,self.vs)[:,None,:,:]
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
            latent_emb=latent[:,None,None,:].repeat(1,1,12,1)
            trans_emb=torch.einsum('bi,ri->br',trans,self.vs)[:,None,:]
            scale_emb=scale[:,None,None].repeat(1,1,12)


        elif FLAGS.fea_type=='a5':

            latent_emb=self.to_a5(latent).permute(0,2,1)[:,None,:,:]


            point_emb=self.to_a5(point)[:,:,:,None]


            trans_emb=self.to_a5(trans[:,None,:])

            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            scale_emb=scale[:,None,None].repeat(1,1,5)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,5,1)


        ts_emb=torch.cat([trans_emb,scale_emb],dim=1)


        ts_emb=self.ts_linear(ts_emb.permute(0,2,1))[:,None,:,:]

        latent_emb=self.x_in_ts(latent_emb)
        r_vector=self.r_in_ts(r_vector)


        cond_feature=torch.cat([point_emb,cond],dim=-1)

        if FLAGS.per_cat=='combine':
            cls_vector=self.cls_linear_ts(cls_vector)
            x = torch.cat([t_emb,cls_vector,latent_emb,r_vector,ts_emb,learned_emb],dim=1)
        else:
            x = torch.cat([t_emb,latent_emb,r_vector,ts_emb,learned_emb],dim=1)



        tokens = self.causal_transformer_ts(x, context=cond_feature)

        if FLAGS.fea_type=='a12':
            tokens=self.to_vec_ts(tokens,self.vs)
        elif FLAGS.fea_type=='a5':
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=features

        features = tokens[..., -1, :,:]


        dual_features=self.fc_inv(features)
        inv_fea= (features * dual_features).sum(-1)
        out_scale=self.fc_out_inv(inv_fea).squeeze(1)
        out_trans=self.fc_out_so3(features).squeeze(1)

        return out_trans,out_scale


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
            ff_dropout = 0.0,
            final_proj = True,
            normformer = False,
            rotary_emb = True,
            point_feature_dim =0,
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
                Attention_v3(dim = dim_in_out, causal = False, dim_head = dim_head, heads = heads, rotary_emb = None),
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



class CausalTransformer_v4(nn.Module):
    def __init__(
            self,
            dim,
            in_out_dim=None,
            depth=4,
            cross_attn=True,
            dim_head = 64,
            heads = 8,
            ff_mult = 4,
            norm_in = False,
            norm_out = True,
            attn_dropout = 0.,
            ff_dropout = 0.0,
            final_proj = True,
            normformer = False,
            rotary_emb = True,
            point_feature_dim =0,
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        if FLAGS.regular_transformer:
            self.rel_pos_bias = None
            rotary_emb = None
        else:
            self.rel_pos_bias = RelPosBias(heads = heads)
            rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None
        rotary_emb_cross =  None

        self.layers = nn.ModuleList([])
        if in_out_dim==None:
            in_out_dim=dim
        self.use_same_dim=(in_out_dim==dim)

        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)

        if cross_attn:
            #print("using CROSS ATTN, with dropout {}".format(attn_dropout))
            self.layers.append(nn.ModuleList([
                Attention_v2(dim = in_out_dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention_v2_cross(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                Attention_v3(dim = dim, causal = False, dim_head = dim_head, heads = heads, rotary_emb = None),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Attention_v2(dim = dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    Attention_v2_cross(dim = dim, kv_dim=point_feature_dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                    Attention_v3(dim = dim, causal = False, dim_head = dim_head, heads = heads, rotary_emb = None),
                    FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
                ]))
            self.layers.append(nn.ModuleList([
                Attention_v2(dim = dim, out_dim=dim, causal = True, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                Attention_v2_cross(dim = dim, kv_dim=point_feature_dim, out_dim=dim, causal = False, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb_cross),
                Attention_v3(dim = dim, out_dim=in_out_dim,causal = False, dim_head = dim_head, heads = heads, rotary_emb = None),
                FeedForward(dim = in_out_dim, out_dim=in_out_dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(in_out_dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options



        self.cross_attn = cross_attn



    def forward(self, x, context=None):

        n, device = x.shape[1], x.device


        x = self.init_norm(x)

        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(n, n + 1, device = device)
        else:
            attn_bias=None

        if self.cross_attn:
            #assert context is not None
            for idx, (self_attn, cross_attn, chanel_atten,ff) in enumerate(self.layers):
                if not self.use_same_dim and idx==0:
                    x = self_attn(x, attn_bias = attn_bias)
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now

                    fuse_x=chanel_atten(x)+x
                    x = ff(fuse_x) + fuse_x
                elif not self.use_same_dim and idx==len(self.layers)-1:
                    x = self_attn(x, attn_bias = attn_bias) + x
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now

                    fuse_x=chanel_atten(x)
                    x = ff(fuse_x) + fuse_x
                else:
                    x = self_attn(x, attn_bias = attn_bias) + x
                    x = cross_attn(x, context=context) + x  # removing attn_bias for now

                    fuse_x=chanel_atten(x)+x
                    x = ff(fuse_x) + fuse_x








        out = self.norm(x)

        return out



class CausalTransformer_v5(nn.Module):
    def __init__(
            self,
            dim,
            depth=12
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention_v4(dim = dim,init_scale=1),
                Attention_v5(dim = dim,init_scale=1),
                pointe_MLP(width=dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))





    def forward(self, x, context=None):

        n, device = x.shape[1], x.device

        for idx, (feature_attn, neuron_attn, ff,ln_1,ln_2,ln_3) in enumerate(self.layers):
            x = x + feature_attn(ln_1(x))
            x = x + neuron_attn(ln_2(x))
            x = x + ff(ln_3(x))










        return x





class CausalTransformer_v6(nn.Module):
    def __init__(
            self,
            dim,
            init_scale,
            heads=8,
            depth=6,
    ):
        super().__init__()
        self.encoder_layers = nn.ModuleList([])
        self.middle_layers = nn.ModuleList([])
        self.decoder_layers = nn.ModuleList([])
        for _ in range(depth):
            self.encoder_layers.append(nn.ModuleList([
                Attention_v4(dim = dim,init_scale=init_scale,heads=heads),
                Attention_v5(dim = dim,init_scale=init_scale,heads=heads),
                DIT_MLP(width=dim,init_scale=init_scale),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ]))
        self.middle_layers.append(nn.ModuleList([
            Attention_v4(dim = dim,init_scale=init_scale,heads=heads),
            Attention_v5(dim = dim,init_scale=init_scale,heads=heads),
            DIT_MLP(width=dim,init_scale=init_scale),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
            nn.LayerNorm(dim),
        ]))
        for _ in range(depth):
            tmp=nn.ModuleList([
                Attention_v4(dim = dim,init_scale=init_scale,heads=heads),
                Attention_v5(dim = dim,init_scale=init_scale,heads=heads),
                DIT_MLP(width=dim,init_scale=init_scale),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
                nn.LayerNorm(dim),
            ])
            linear = nn.Linear(dim * 2, dim)
            init_linear(linear, init_scale)
            layer_norm = nn.LayerNorm(dim)
            tmp.append(linear)
            tmp.append(layer_norm)
            self.decoder_layers.append(tmp)





    def forward(self, x, context=None):

        n, device = x.shape[1], x.device
        enc_outputs = []

        for idx, (feature_attn, neuron_attn, ff,ln_1,ln_2,ln_3) in enumerate(self.encoder_layers):
            x = x + feature_attn(ln_1(x))
            x = x + neuron_attn(ln_2(x))
            x = x + ff(ln_3(x))
            enc_outputs.append(x)
        for idx, (feature_attn, neuron_attn, ff,ln_1,ln_2,ln_3) in enumerate(self.middle_layers):
            x = x + feature_attn(ln_1(x))
            x = x + neuron_attn(ln_2(x))
            x = x + ff(ln_3(x))

        for idx, (feature_attn, neuron_attn, ff,ln_1,ln_2,ln_3,ff_2,ln_4) in enumerate(self.decoder_layers):
            x=torch.cat([enc_outputs.pop(),x],dim=-1)
            x = ff_2(x)
            x =ln_4(x)

            x = x + feature_attn(ln_1(x))
            x = x + neuron_attn(ln_2(x))
            x = x + ff(ln_3(x))

        return x




class DiffusionNet_v2(nn.Module):

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


        data, cond = data
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



class DiffusionNet_v4(nn.Module):

    def __init__(
            self,
            dim=512,
            dim_in_out=256,
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
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        if FLAGS.use_fuse and not FLAGS.use_simple:
            self.point_feature_dim = sum(self.dim_list[1:])
        else:
            self.point_feature_dim = self.dim_list[-1]
        self.dim_in_out= dim_in_out
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
        self.causal_transformer_latent = CausalTransformer_v4(dim = dim, in_out_dim=self.dim_in_out, point_feature_dim=self.point_feature_dim)
        if FLAGS.use_simple:
            self.causal_transformer_ts = CausalTransformer_v3(dim = self.dim_in_out, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim+1)
        else:
            raise NotImplementedError


        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        if FLAGS.fea_type=='a5':
            faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
                   (1,2,4),(2,7,8),(3,7,9),(3,5,11),
                   (4,5,6),(2,4,10),(2,8,10),(7,8,9),
                   (3,9,11),(5,6,11),(4,6,10),(0,8,10),
                   (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

            face_normal=vs_[faces,:].sum(1)
            face_normal=F.normalize(face_normal,dim=-1)
            self.register_buffer('face_normal',face_normal)

            face_to_cube=[(1,4,0,2,3),(2,0,1,4,3),(3,1,0,4,2),(4,2,0,3,1),
                          (0,3,1,2,4),(3,2,0,4,1),(4,3,0,2,1),(0,4,1,2,3),
                          (1,0,2,4,3),(2,1,0,4,3),(4,0,1,3,2),(0,1,2,3,4),
                          (1,2,0,3,4),(2,3,0,1,4),(3,4,0,1,2),(1,3,0,2,4),
                          (0,2,1,3,4),(4,1,0,3,2),(3,0,1,4,2),(2,4,0,1,3)]
            face_to_cube=torch.from_numpy(np.array(face_to_cube))
            self.register_buffer('face_to_cube',face_to_cube)




        self.x_in_latent=nn.Linear(256,self.dim_in_out)
        self.cls_linear_latent=nn.Linear(6,self.dim_in_out)
        self.cls_linear_ts=nn.Linear(6,self.dim_in_out)
        self.to_vec_latent=to_vec(self.dim_in_out)

        self.latent_out = nn.Sequential(VNLinear(self.dim_in_out,256),VNLayerNorm(256),
                                        VNReLU(256),VNLinear(256,256))


        if FLAGS.fea_type=='a5':
            self.latent_out_a5=VNLinear(self.dim_in_out,256)
            self.ts_out_a5=VNLinear(self.dim_in_out,128)

        self.x_in_ts=nn.Linear(256,self.dim_in_out)

        self.ts_linear=nn.Linear(2,self.dim_in_out)

        merging_mlp = [self.dim_in_out,256,128]
        merging_mlp_layers = []
        for k in range(len(merging_mlp) - 1):
            merging_mlp_layers.extend([
                VNLinear(merging_mlp[k],merging_mlp[k + 1]),
                VNLayerNorm(merging_mlp[k + 1]),
                VNReLU(merging_mlp[k + 1])
            ])
        self.merging_mlp = nn.Sequential(*merging_mlp_layers)
        self.fc_inv = VNLinear(128, 128)
        self.fc_out_so3 = VNLinear(128, 1)
        self.fc_out_inv = nn.Linear(128, 1)

        if not FLAGS.use_simple:
            self.to_vec_ts=to_vec(self.dim_in_out)

        else:
            self.to_vec_ts=to_vec(128,self.dim_in_out)


    def forward_latent(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):


        data, cond,cls_vector = data

        cond_feature = cond


        batch, dim, _,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_latent(diffusion_timesteps)

        data = data

        learned_queries = repeat(self.learned_query_latent, 'd -> b 1 d', b = batch)

        if FLAGS.fea_type=='a12':
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            x_sample=torch.einsum('bqi,ri->brq',data,self.vs)[:,None,:,:]

        else:
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,5,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            x_sample=self.to_a5(data).permute(0,2,1)[:,None,:,:]


        if FLAGS.per_cat=='combine':
            raise NotImplementedError
            cls_vector=self.cls_linear_latent(cls_vector)
            x = torch.cat([t_emb,cls_vector,x_sample,learned_emb],dim=1)
        else:
            x = torch.cat([t_emb,x_sample,learned_emb],dim=1)



        tokens = self.causal_transformer_latent(x, context=cond_feature)
        if FLAGS.fea_type=='a12':
            tokens = self.to_vec_latent(tokens,self.vs)
        else:
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=self.latent_out_a5(features)




        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :,:]
        if not FLAGS.use_simple:
            pred=self.latent_out(pred)

        return pred

    def to_a5(self,x):
        x_face=torch.einsum('f i , b n i -> b n f',self.face_normal,x)
        x_face=torch.tanh(x_face)
        color=torch.zeros_like(x_face)[:,:,:1].repeat(1,1,5)
        b,n,_=x_face.shape
        color.scatter_add_(-1,self.face_to_cube[:,0][None,None,:].repeat(b,n,1),x_face)
        return color

    def forward_ts(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        assert type(data) is tuple
        latent,trans,scale, cond ,point,cls_vector= data # adding noise to cond_feature so doing this in diffusion.py



        batch, dim, _,device, dtype = *latent.shape, latent.device, latent.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_ts(diffusion_timesteps)


        learned_queries = repeat(self.learned_query_ts, 'd -> b 1 d', b = batch)

        if FLAGS.fea_type=='a12':
            point_emb=torch.einsum('bqi,ri->bqr',point,self.vs)[:,:,:,None]
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
            latent_emb=torch.einsum('bqi,ri->brq',latent,self.vs)[:,None,:,:]
            trans_emb=torch.einsum('bi,ri->br',trans,self.vs)[:,None,:]
            scale_emb=scale[:,None,None].repeat(1,1,12)


        elif FLAGS.fea_type=='a5':

            latent_emb=self.to_a5(latent).permute(0,2,1)[:,None,:,:]


            point_emb=self.to_a5(point)[:,:,:,None]


            trans_emb=self.to_a5(trans[:,None,:])

            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            scale_emb=scale[:,None,None].repeat(1,1,5)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,5,1)


        ts_emb=torch.cat([trans_emb,scale_emb],dim=1)


        ts_emb=self.ts_linear(ts_emb.permute(0,2,1))[:,None,:,:]


        cond_feature=torch.cat([point_emb,cond],dim=-1)

        if FLAGS.per_cat=='combine':
            cls_vector=self.cls_linear_ts(cls_vector)
            x = torch.cat([t_emb,cls_vector,latent_emb,ts_emb,learned_emb],dim=1)
        else:
            x = torch.cat([t_emb,latent_emb,ts_emb,learned_emb],dim=1)



        tokens = self.causal_transformer_ts(x, context=cond_feature)

        if FLAGS.fea_type=='a12':
            tokens=self.to_vec_ts(tokens,self.vs)
        elif FLAGS.fea_type=='a5':
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=self.ts_out_a5(features)


        features = tokens[..., -1, :,:]
        if not FLAGS.use_simple:
            features=self.merging_mlp(features)

        dual_features=self.fc_inv(features)
        inv_fea= (features * dual_features).sum(-1)
        out_scale=self.fc_out_inv(inv_fea).squeeze(1)
        out_trans=self.fc_out_so3(features).squeeze(1)

        return out_trans,out_scale






class DiffusionNet_res(nn.Module):

    def __init__(
            self,
            dim=512,
            dim_in_out=256,
            num_timesteps = None,
            num_time_embeds = 1,
            cond = True,
            num_res=6,
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = True
        self.cond_dropout = True
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        if FLAGS.use_fuse:
            self.point_feature_dim = sum(self.dim_list[1:])
        else:
            self.point_feature_dim = self.dim_list[-1]

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


        self.res_latent =nn.ModuleList([])
        self.num_res=num_res
        for i in range(num_res):
            if i==0:
                self.res_latent.append(ResnetBlock(dim_in_out,dim,time_cond_dim=dim_in_out))
            elif i==num_res-1:
                self.res_latent.append(Attention_v3(dim = dim))
            else:
                self.res_latent.append(ResnetBlock(dim,dim,time_cond_dim=dim_in_out,cond_dim=self.point_feature_dim))



        self.res_ts = nn.ModuleList([])
        for i in range(num_res):
            if i==0:
                self.res_ts.append(ResnetBlock(dim_in_out+2,dim,time_cond_dim=dim_in_out))
            elif i==num_res-1:
                self.res_ts.append(Attention_v3(dim = dim))
            else:
                self.res_ts.append(ResnetBlock(dim,dim,time_cond_dim=dim_in_out,cond_dim=self.point_feature_dim+1))


        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        if FLAGS.fea_type=='a5':
            faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
                   (1,2,4),(2,7,8),(3,7,9),(3,5,11),
                   (4,5,6),(2,4,10),(2,8,10),(7,8,9),
                   (3,9,11),(5,6,11),(4,6,10),(0,8,10),
                   (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

            face_normal=vs_[faces,:].sum(1)
            face_normal=F.normalize(face_normal,dim=-1)
            self.register_buffer('face_normal',face_normal)

            face_to_cube=[(1,4,0,2,3),(2,0,1,4,3),(3,1,0,4,2),(4,2,0,3,1),
                          (0,3,1,2,4),(3,2,0,4,1),(4,3,0,2,1),(0,4,1,2,3),
                          (1,0,2,4,3),(2,1,0,4,3),(4,0,1,3,2),(0,1,2,3,4),
                          (1,2,0,3,4),(2,3,0,1,4),(3,4,0,1,2),(1,3,0,2,4),
                          (0,2,1,3,4),(4,1,0,3,2),(3,0,1,4,2),(2,4,0,1,3)]
            face_to_cube=torch.from_numpy(np.array(face_to_cube))
            self.register_buffer('face_to_cube',face_to_cube)





        self.cls_linear_latent=nn.Linear(6,dim)
        self.cls_linear_ts=nn.Linear(6,dim)
        self.to_vec_latent=to_vec(dim)

        self.latent_out = nn.Sequential(VNLinear(dim,256),VNLayerNorm(256),
                                        VNReLU(256),VNLinear(256,256))


        self.to_vec_ts=to_vec(dim)
        self.ts_linear=nn.Linear(2,dim)

        merging_mlp = [dim,256,128]
        merging_mlp_layers = []
        for k in range(len(merging_mlp) - 1):
            merging_mlp_layers.extend([
                VNLinear(merging_mlp[k],merging_mlp[k + 1]),
                VNLayerNorm(merging_mlp[k + 1]),
                VNReLU(merging_mlp[k + 1])
            ])
        self.merging_mlp = nn.Sequential(*merging_mlp_layers)


        self.fc_inv = VNLinear(128, 128)
        self.fc_out_so3 = VNLinear(128, 1)
        self.fc_out_inv = nn.Linear(128, 1)


    def forward_latent(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):


        data, cond,cls_vector = data
        cond_feature = cond


        batch, dim, _,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_latent(diffusion_timesteps)

        data = data


        if FLAGS.fea_type=='a12':
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            x_sample=torch.einsum('bqi,ri->brq',data,self.vs)[:,None,:,:]

        else:
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            x_sample=self.to_a5(data).permute(0,2,1)[:,None,:,:]

        for idx, res in enumerate(self.res_latent):
            if idx==0:
                x=res(x_sample,t_emb)
            elif idx==len(self.res_ts)-1:
                x=res(x)
            else:
                x=res(x,t_emb,cond_feature)

        tokens=x
        if FLAGS.fea_type=='a12':
            tokens = self.to_vec_latent(tokens,self.vs)
        else:
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=features


        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., -1, :,:]
        pred=self.latent_out(pred)

        return pred

    def to_a5(self,x):
        x_face=torch.einsum('f i , b n i -> b n f',self.face_normal,x)
        x_face=torch.tanh(x_face)
        color=torch.zeros_like(x_face)[:,:,:1].repeat(1,1,5)
        b,n,_=x_face.shape
        color.scatter_add_(-1,self.face_to_cube[:,0][None,None,:].repeat(b,n,1),x_face)
        return color

    def forward_ts(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        assert type(data) is tuple
        latent,trans,scale, cond ,point,cls_vector= data # adding noise to cond_feature so doing this in diffusion.py



        batch, dim, _,device, dtype = *latent.shape, latent.device, latent.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_ts(diffusion_timesteps)


        learned_queries = repeat(self.learned_query_ts, 'd -> b 1 d', b = batch)

        if FLAGS.fea_type=='a12':
            point_emb=torch.einsum('bqi,ri->bqr',point,self.vs)[:,:,:,None]
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            latent_emb=torch.einsum('bqi,ri->brq',latent,self.vs)[:,None,:,:]

            trans_emb=torch.einsum('bi,ri->br',trans,self.vs)[:,None,:]
            scale_emb=scale[:,None,None].repeat(1,1,12)


        elif FLAGS.fea_type=='a5':

            latent_emb=self.to_a5(latent).permute(0,2,1)[:,None,:,:]

            point_emb=self.to_a5(point)[:,:,:,None]


            trans_emb=self.to_a5(trans[:,None,:])

            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            scale_emb=scale[:,None,None].repeat(1,1,5)


        ts_emb=torch.cat([trans_emb,scale_emb],dim=1)
        ts_emb=ts_emb.permute(0,2,1)[:,None,:,:]
        x_sample=torch.cat([latent_emb,ts_emb],dim=-1)



        cond_feature=torch.cat([cond,point_emb],dim=-1)




        for idx, res in enumerate(self.res_ts):
            if idx==0:
                x=res(x_sample,t_emb)
            elif idx==len(self.res_ts)-1:
                x=res(x)
            else:
                x=res(x,t_emb,cond_feature)

        tokens=x
        if FLAGS.fea_type=='a12':
            tokens=self.to_vec_ts(tokens,self.vs)
        elif FLAGS.fea_type=='a5':
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=features

        features = tokens[..., -1, :,:]

        features=self.merging_mlp(features)

        dual_features=self.fc_inv(features)
        inv_fea= (features * dual_features).sum(-1)
        out_scale=self.fc_out_inv(inv_fea).squeeze(1)
        out_trans=self.fc_out_so3(features).squeeze(1)

        return out_trans,out_scale



class DiffusionNet_v5(nn.Module):

    def __init__(
            self,
            dim=512,
            dim_in_out=256,
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
        self.dim_list=[]
        for s in FLAGS.dim_list:
            self.dim_list.append(int(s))
        if FLAGS.use_fuse and not FLAGS.use_simple:
            self.point_feature_dim = sum(self.dim_list[1:])
        else:
            self.point_feature_dim = self.dim_list[-1]
        self.dim_in_out= dim_in_out
        #print("dim, in out, point feature dim: ", dim, dim_in_out, self.point_feature_dim)
        #print("cond dropout: ", self.cond_dropout)

        self.to_time_embeds_latent = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim), MLP(self.dim, self.dim * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        self.to_time_embeds_ts = nn.Sequential(
            nn.Embedding(num_timesteps, self.dim_in_out * num_time_embeds) if exists(num_timesteps) else nn.Sequential(SinusoidalPosEmb(self.dim_in_out), MLP(self.dim_in_out, self.dim_in_out * num_time_embeds)), # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n = num_time_embeds)
        )

        # last input to the transformer: "a final embedding whose output from the Transformer is used to predicted the unnoised CLIP image embedding"
        self.learned_query_latent = nn.Parameter(torch.randn(self.dim_in_out))
        self.learned_query_ts = nn.Parameter(torch.randn(self.dim_in_out))
        self.causal_transformer_latent = CausalTransformer_v5(dim = dim)
        # self.ln_post_latent = nn.LayerNorm(dim)
        # self.ln_post_ts = nn.LayerNorm(dim)

        self.causal_transformer_ts = CausalTransformer_v3(dim = self.dim_in_out, dim_in_out=self.dim_in_out, point_feature_dim=self.point_feature_dim+1)


        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        if FLAGS.fea_type=='a5':
            faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
                   (1,2,4),(2,7,8),(3,7,9),(3,5,11),
                   (4,5,6),(2,4,10),(2,8,10),(7,8,9),
                   (3,9,11),(5,6,11),(4,6,10),(0,8,10),
                   (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

            face_normal=vs_[faces,:].sum(1)
            face_normal=F.normalize(face_normal,dim=-1)
            self.register_buffer('face_normal',face_normal)

            face_to_cube=[(1,4,0,2,3),(2,0,1,4,3),(3,1,0,4,2),(4,2,0,3,1),
                          (0,3,1,2,4),(3,2,0,4,1),(4,3,0,2,1),(0,4,1,2,3),
                          (1,0,2,4,3),(2,1,0,4,3),(4,0,1,3,2),(0,1,2,3,4),
                          (1,2,0,3,4),(2,3,0,1,4),(3,4,0,1,2),(1,3,0,2,4),
                          (0,2,1,3,4),(4,1,0,3,2),(3,0,1,4,2),(2,4,0,1,3)]
            face_to_cube=torch.from_numpy(np.array(face_to_cube))
            self.register_buffer('face_to_cube',face_to_cube)




        self.x_in_latent=nn.Linear(256,dim)
        self.cond_in_latent=nn.Linear(256,dim)
        self.cls_linear_latent=nn.Linear(6,self.dim_in_out)
        self.cls_linear_ts=nn.Linear(6,self.dim_in_out)
        self.to_vec_latent=to_vec(self.dim_in_out,dim)

        self.latent_out = nn.Sequential(VNLinear(self.dim_in_out,256),VNLayerNorm(256),
                                        VNReLU(256),VNLinear(256,256))


        if FLAGS.fea_type=='a5':
            self.latent_out_a5=VNLinear(self.dim_in_out,256)
            self.ts_out_a5=VNLinear(self.dim_in_out,128)

        self.x_in_ts=nn.Linear(256,self.dim_in_out)

        self.ts_linear=nn.Linear(2,self.dim_in_out)

        merging_mlp = [self.dim_in_out,256,128]
        merging_mlp_layers = []
        for k in range(len(merging_mlp) - 1):
            merging_mlp_layers.extend([
                VNLinear(merging_mlp[k],merging_mlp[k + 1]),
                VNLayerNorm(merging_mlp[k + 1]),
                VNReLU(merging_mlp[k + 1])
            ])
        self.merging_mlp = nn.Sequential(*merging_mlp_layers)
        self.fc_inv = VNLinear(128, 128)
        self.fc_out_so3 = VNLinear(128, 1)
        self.fc_out_inv = nn.Linear(128, 1)

        self.to_vec_ts=to_vec(128,self.dim_in_out)


    def forward_latent(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):


        data, cond,cls_vector = data

        cond_feature = cond


        batch, dim, _,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_latent(diffusion_timesteps)

        data = data


        if FLAGS.fea_type=='a12':
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            x_sample=torch.einsum('bqi,ri->brq',data,self.vs)[:,None,:,:]
            x_sample=self.x_in_latent(x_sample)
            cond=self.cond_in_latent(cond)

        else:
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            x_sample=self.to_a5(data).permute(0,2,1)[:,None,:,:]


        if FLAGS.per_cat=='combine':
            raise NotImplementedError
            cls_vector=self.cls_linear_latent(cls_vector)
            x = torch.cat([t_emb,cls_vector,x_sample],dim=1)
        else:
            x = torch.cat([x_sample,t_emb,cond],dim=1)



        tokens = self.causal_transformer_latent(x)
        # tokens= self.ln_post_latent(tokens)

        if FLAGS.fea_type=='a12':
            tokens = self.to_vec_latent(tokens,self.vs)
        else:
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=self.latent_out_a5(features)




        # get learned query, which should predict the sdf layer embedding (per DDPM timestep)
        pred = tokens[..., 0, :,:]
        if not FLAGS.use_simple:
            pred=self.latent_out(pred)

        return pred

    def to_a5(self,x):
        x_face=torch.einsum('f i , b n i -> b n f',self.face_normal,x)
        x_face=torch.tanh(x_face)
        color=torch.zeros_like(x_face)[:,:,:1].repeat(1,1,5)
        b,n,_=x_face.shape
        color.scatter_add_(-1,self.face_to_cube[:,0][None,None,:].repeat(b,n,1),x_face)
        return color

    def forward_ts(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        assert type(data) is tuple
        latent,trans,scale, cond ,point,cls_vector= data # adding noise to cond_feature so doing this in diffusion.py



        batch, dim, _,device, dtype = *latent.shape, latent.device, latent.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.to_time_embeds_ts(diffusion_timesteps)


        learned_queries = repeat(self.learned_query_ts, 'd -> b 1 d', b = batch)

        if FLAGS.fea_type=='a12':
            point_emb=torch.einsum('bqi,ri->bqr',point,self.vs)[:,:,:,None]
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            t_emb=time_embed[:,:,None,:].repeat(1,1,12,1)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,12,1)
            latent_emb=torch.einsum('bqi,ri->brq',latent,self.vs)[:,None,:,:]
            trans_emb=torch.einsum('bi,ri->br',trans,self.vs)[:,None,:]
            scale_emb=scale[:,None,None].repeat(1,1,12)


        elif FLAGS.fea_type=='a5':

            latent_emb=self.to_a5(latent).permute(0,2,1)[:,None,:,:]


            point_emb=self.to_a5(point)[:,:,:,None]


            trans_emb=self.to_a5(trans[:,None,:])

            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            scale_emb=scale[:,None,None].repeat(1,1,5)
            learned_emb=learned_queries[:,:,None,:].repeat(1,1,5,1)


        ts_emb=torch.cat([trans_emb,scale_emb],dim=1)


        ts_emb=self.ts_linear(ts_emb.permute(0,2,1))[:,None,:,:]


        cond_feature=torch.cat([point_emb,cond],dim=-1)

        if FLAGS.per_cat=='combine':
            cls_vector=self.cls_linear_ts(cls_vector)
            x = torch.cat([t_emb,cls_vector,latent_emb,ts_emb,learned_emb],dim=1)
        else:
            x = torch.cat([t_emb,latent_emb,ts_emb,learned_emb],dim=1)



        tokens = self.causal_transformer_ts(x, context=cond_feature)

        if FLAGS.fea_type=='a12':
            tokens=self.to_vec_ts(tokens,self.vs)
        elif FLAGS.fea_type=='a5':
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=self.ts_out_a5(features)


        features = tokens[..., -1, :,:]
        if not FLAGS.use_simple:
            features=self.merging_mlp(features)

        dual_features=self.fc_inv(features)
        inv_fea= (features * dual_features).sum(-1)
        out_scale=self.fc_out_inv(inv_fea).squeeze(1)
        out_trans=self.fc_out_so3(features).squeeze(1)

        return out_trans,out_scale


class DiffusionNet_v6(nn.Module):

    def __init__(
            self,
            dim=512,
            dim_in_out=256,
            num_timesteps = None,
            num_time_embeds = 1,
            cond = True,
            init_scale=1,
    ):
        super().__init__()
        self.num_time_embeds = num_time_embeds
        self.dim = dim
        self.cond = cond
        self.cross_attn = True
        self.cond_dropout = True
        self.dim_list=[]

        self.dim_in_out= dim_in_out

        init_scale=init_scale * math.sqrt(1.0 / dim)



        self.time_embed=Timesteps(dim,flip_sin_to_cos=False,downscale_freq_shift=0)
        self.time_proj_latent = DIT_MLP( width=dim, init_scale=init_scale)
        self.time_proj_ts = DIT_MLP( width=dim, init_scale=init_scale)

        self.context_embed_latent = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, dim),
        )

        self.context_ts_norm = nn.LayerNorm(256)


        self.causal_transformer_latent = CausalTransformer_v6(dim = dim,depth=5,init_scale=init_scale)

        self.ln_post_latent = nn.LayerNorm(dim)
        self.ln_post_ts = nn.LayerNorm(dim)

        self.causal_transformer_ts = CausalTransformer_v6(dim = dim,depth=2,init_scale=init_scale)


        rotation_dict=torch.load(FLAGS.rotation_path)
        vs_=rotation_dict['vs'].float()
        self.register_buffer('vs',vs_)
        if FLAGS.fea_type=='a5':
            faces=[(1,2,7),(1,3,7),(1,3,5),(1,4,5),
                   (1,2,4),(2,7,8),(3,7,9),(3,5,11),
                   (4,5,6),(2,4,10),(2,8,10),(7,8,9),
                   (3,9,11),(5,6,11),(4,6,10),(0,8,10),
                   (0,6,10),(0,6,11),(0,9,11),(0,8,9)]

            face_normal=vs_[faces,:].sum(1)
            face_normal=F.normalize(face_normal,dim=-1)
            self.register_buffer('face_normal',face_normal)

            face_to_cube=[(1,4,0,2,3),(2,0,1,4,3),(3,1,0,4,2),(4,2,0,3,1),
                          (0,3,1,2,4),(3,2,0,4,1),(4,3,0,2,1),(0,4,1,2,3),
                          (1,0,2,4,3),(2,1,0,4,3),(4,0,1,3,2),(0,1,2,3,4),
                          (1,2,0,3,4),(2,3,0,1,4),(3,4,0,1,2),(1,3,0,2,4),
                          (0,2,1,3,4),(4,1,0,3,2),(3,0,1,4,2),(2,4,0,1,3)]
            face_to_cube=torch.from_numpy(np.array(face_to_cube))
            self.register_buffer('face_to_cube',face_to_cube)




        self.x_in_latent=nn.Linear(256,dim)
        self.x_in_ts=nn.Linear(256,dim)
        self.cls_linear_latent=nn.Linear(6,self.dim_in_out)
        self.cls_linear_ts=nn.Linear(6,self.dim_in_out)
        self.to_vec_latent=to_vec(self.dim_in_out,dim)

        if FLAGS.point_emb:
            self.point_in=nn.Linear(1,256)
            self.context_ts_lin = nn.Linear(512, dim)
        else:
            self.context_ts_lin = nn.Linear(257, dim)

        self.latent_out = nn.Sequential(VNLinear(self.dim_in_out,256),VNLayerNorm(256),
                                        VNReLU(256),VNLinear(256,256))


        if FLAGS.fea_type=='a5':
            self.latent_out_a5=VNLinear(self.dim_in_out,256)
            self.ts_out_a5=VNLinear(self.dim_in_out,128)

        self.x_in_ts=nn.Linear(256,dim)

        if FLAGS.use_simple:
            self.trans_in=nn.Linear(1,dim)
            self.scale_in=nn.Linear(1,dim)
        else:
            self.ts_linear=nn.Linear(2,dim)


        self.fc_inv = VNLinear(128, 128)
        self.fc_out_so3 = VNLinear(128, 1)
        self.fc_out_inv = nn.Linear(128, 1)


        if FLAGS.use_simple:
            self.to_vec_trans=to_vec(1,dim)
            self.out_lin_scale=nn.Linear(dim, 1)
        else:
            self.to_vec_ts=to_vec(128,dim)


    def forward_latent(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):


        data, cond,cls_vector = data

        cond_feature = cond


        batch, dim, _,device, dtype = *data.shape, data.device, data.dtype

        num_time_embeds = self.num_time_embeds
        time_embed = self.time_proj_latent(self.time_embed(diffusion_timesteps))

        data = data


        if FLAGS.fea_type=='a12':
            t_emb=time_embed[:,None,None,:].repeat(1,1,12,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            x_sample=torch.einsum('bqi,ri->brq',data,self.vs)[:,None,:,:]
            x_sample=self.x_in_latent(x_sample)
            cond=self.context_embed_latent(cond)


        else:
            raise NotImplementedError
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            x_sample=self.to_a5(data).permute(0,2,1)[:,None,:,:]


        if FLAGS.per_cat=='combine':
            raise NotImplementedError
            cls_vector=self.cls_linear_latent(cls_vector)
            x = torch.cat([t_emb,cls_vector,x_sample],dim=1)
        else:
            x = torch.cat([x_sample,t_emb,cond],dim=1)



        tokens = self.causal_transformer_latent(x)
        tokens=self.ln_post_latent(tokens)
        if FLAGS.fea_type=='a12':
            tokens = self.to_vec_latent(tokens,self.vs)
        else:
            raise NotImplementedError
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=self.latent_out_a5(features)

        pred = tokens[..., 0, :,:]


        return pred

    def to_a5(self,x):
        x_face=torch.einsum('f i , b n i -> b n f',self.face_normal,x)
        x_face=torch.tanh(x_face)
        color=torch.zeros_like(x_face)[:,:,:1].repeat(1,1,5)
        b,n,_=x_face.shape
        color.scatter_add_(-1,self.face_to_cube[:,0][None,None,:].repeat(b,n,1),x_face)
        return color

    def forward_ts(
            self,
            data,
            diffusion_timesteps,
            pass_cond=-1, # default -1, depends on prob; but pass as argument during sampling

    ):

        assert type(data) is tuple
        latent,trans,scale, cond ,point,cls_vector= data # adding noise to cond_feature so doing this in diffusion.py



        batch, dim, _,device, dtype = *latent.shape, latent.device, latent.dtype


        time_embed = self.time_proj_ts(self.time_embed(diffusion_timesteps))



        if FLAGS.fea_type=='a12':
            point_emb=torch.einsum('bqi,ri->bqr',point,self.vs)[:,:,:,None]
            cls_vector=cls_vector[:,None,None,:].repeat(1,1,12,1)
            t_emb=time_embed[:,None,None,:].repeat(1,1,12,1)
            latent_emb=torch.einsum('bqi,ri->brq',latent,self.vs)[:,None,:,:]
            trans_emb=torch.einsum('bi,ri->br',trans,self.vs)[:,None,:]
            scale_emb=scale[:,None,None].repeat(1,1,12)


        elif FLAGS.fea_type=='a5':
            latent_emb=self.to_a5(latent).permute(0,2,1)[:,None,:,:]


            point_emb=self.to_a5(point)[:,:,:,None]


            trans_emb=self.to_a5(trans[:,None,:])

            cls_vector=cls_vector[:,None,None,:].repeat(1,1,5,1)
            t_emb=time_embed[:,:,None,:].repeat(1,1,5,1)
            scale_emb=scale[:,None,None].repeat(1,1,5)


        latent_emb=self.x_in_ts(latent_emb)
        if FLAGS.use_simple:
            trans_emb=self.trans_in(trans_emb.permute(0,2,1))[:,None,:,:]
            scale_emb=self.scale_in(scale_emb.permute(0,2,1))[:,None,:,:]
        else:
            ts_emb=torch.cat([trans_emb,scale_emb],dim=1)
            ts_emb=self.ts_linear(ts_emb.permute(0,2,1))[:,None,:,:]

        cond=self.context_ts_norm(cond)
        if FLAGS.point_emb:
            point_emb=self.point_in(point_emb)
        cond_feature=torch.cat([point_emb,cond],dim=-1)
        cond_feature=self.context_ts_lin(cond_feature)


        if FLAGS.per_cat=='combine':
            cls_vector=self.cls_linear_ts(cls_vector)
            x = torch.cat([t_emb,cls_vector,latent_emb,ts_emb,cond_feature],dim=1)
        else:
            if FLAGS.use_simple:
                x = torch.cat([trans_emb,scale_emb,t_emb,latent_emb,cond_feature],dim=1)
            else:
                x = torch.cat([ts_emb,t_emb,latent_emb,cond_feature],dim=1)



        tokens = self.causal_transformer_ts(x)
        tokens=self.ln_post_ts(tokens)

        if FLAGS.fea_type=='a12':
            if FLAGS.use_simple:
                out_trans=self.to_vec_trans(tokens,self.vs)[:,0,0,:]
                out_scale=torch.mean(self.out_lin_scale(tokens)[:,1,:,0],dim=-1)

            else:
                tokens=self.to_vec_ts(tokens,self.vs)
        elif FLAGS.fea_type=='a5':
            features=tokens.permute(0,1,3,2)
            fea_face=features[:,:,:,self.face_to_cube[:,0]]*torch.relu(features[:,:,:,self.face_to_cube[:,0]]-features[:,:,:,self.face_to_cube[:,1]])
            features=torch.einsum('b q c n , n i->b q c i',fea_face,self.face_normal)
            tokens=self.ts_out_a5(features)

        if FLAGS.use_simple:
            pass
        else:
            features = tokens[..., 0, :,:]

            dual_features=self.fc_inv(features)
            inv_fea= (features * dual_features).sum(-1)
            out_scale=self.fc_out_inv(inv_fea).squeeze(1)
            out_trans=self.fc_out_so3(features).squeeze(1)

        return out_trans,out_scale




if __name__ == "__main__":
    def eval(argv):
        rotary_emb = RotaryEmbedding(dim = 32)
        rel_pos_bias = RelPosBias(heads = 8)

        atten=Attention_v2(dim = 512, out_dim=512, causal = True, dim_head = 64, heads = 8, rotary_emb = rotary_emb)
        cross_atten=Attention_v2_cross(dim = 512, out_dim=512, kv_dim=512, causal = False,dim_head = 64, heads = 8, rotary_emb = None)
        atten_chan=Attention_v3(dim = 512, out_dim=512,  causal = False,dim_head = 64, heads = 8, rotary_emb = None)
        ff=FeedForward(dim = 512, out_dim=512, mult = 4, dropout = 0.0, post_activation_norm = False)
        input_1=torch.randn(1, 3, 12,512)
        cond_1=torch.randn(1, 16, 12,512)
        cond_2=cond_1[:,:,(1,0,2,3,4,5,6,7,8,9,10,11),:]
        input_2=input_1[:,:,(1,0,2,3,4,5,6,7,8,9,10,11),:]
        n= input_1.shape[1]
        attn_bias = rel_pos_bias(n, n + 1,device = input_1.device)
        out_1 = ff(atten_chan(cross_atten(atten(input_1, attn_bias = attn_bias),context=cond_1)))
        out_2 = ff(atten_chan(cross_atten(atten(input_2, attn_bias = attn_bias),context=cond_2)))
        print(1)



    from absl import app

    from config.equi_diff_nocs.config import *

    FLAGS = flags.FLAGS
    app.run(eval)




