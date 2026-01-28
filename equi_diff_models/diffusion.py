import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from collections import namedtuple
from functools import partial
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

#from model.diffusion.model import * 
from diff_utils.helpers import * 

import numpy as np
import os
from statistics import mean
from tqdm.auto import tqdm
# import open3d as o3d
import absl.flags as flags
FLAGS = flags.FLAGS

# constants
ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


class DiffusionModel_occ(nn.Module):
    def __init__(
            self,
            model,
            timesteps = 1000, sampling_timesteps = None, beta_schedule = 'cosine',
            sample_pc_size = 128, perturb_pc = 'partial',  crop_percent=0.5,
            loss_type = 'l2', objective = 'pred_x0',
            data_scale = 1.0, data_shift = 0.0,
            p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k = 1,
            ddim_sampling_eta = 1.
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc
        self.crop_percent = crop_percent
        assert self.perturb_pc in [None, "partial", "noisy"]

        self.loss_fn = F.l1_loss if loss_type=='l1' else F.mse_loss

        # sampling related parameters
        self.sampling_timesteps = 50 # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # self.register_buffer('data_scale', torch.tensor(data_scale))
        # self.register_buffer('data_shift', torch.tensor(data_shift))

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))


        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )



    def ddim_sample_2(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False,
                      cond=None,cls_vector=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        interval=np.floor((total_timesteps-1)/(sampling_timesteps-1))
        times=np.arange(sampling_timesteps)*interval
        times=times.astype(np.int)
        times=times.tolist()
        if times[-1]<total_timesteps-1:
            times.append(total_timesteps-1)

        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = torch.randn(batch, dim,device = device)
        r_T = torch.randn(batch, dim,3,device = device)
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)
        cls_vector=cls_vector.repeat(batch,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        r_list=[]
        x_start_list=[]
        r_start_list=[]
        trans_list=[]
        scale_list=[]
        scale_start_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T,r_T, cond,cls_vector)
            model_in_ts=(x_T,r_T,trans_T,scale_T,cond,point,cls_vector)
            x_start,r_start = self.model.forward_latent(model_in, time_cond)

            trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)

            x_start_list.append(x_start)
            r_start_list.append(r_start)
            trans_list.append(trans_start)
            scale_start_list.append(scale_start)

            eta=0
            sigma =  ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()*eta
            c = 1 - alpha_next - sigma ** 2
            c=c/(1-alpha)
            a=alpha_next.sqrt()-(alpha*c).sqrt()
            b=c.sqrt()



            noise_1 = torch.randn_like(x_T) if time > 0 else 0
            noise_2 = torch.randn_like(trans_T) if time > 0 else 0
            noise_3 = torch.randn_like(scale_T) if time > 0 else 0
            noise_4 = torch.randn_like(r_T) if time > 0 else 0

            x_T = x_start * a + x_T*b+ sigma * noise_1
            r_T = r_start * a + r_T*b+ sigma * noise_4
            trans_T = trans_start * a + trans_T*b+ sigma * noise_2
            scale_T = scale_start * a + scale_T*b+ sigma * noise_3
            x_list.append(x_T)
            r_list.append(r_T)
            scale_list.append(scale_T)
            beta_list.append(1-alpha)
            sigma_list.append(sigma)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_start_list=torch.stack(x_start_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        r_list=torch.stack(r_list,dim=0)

        # trans_T=trans_list.mean(0)
        # scale_T=scale_list.mean(0)
        # x_T=x_list.mean(0)
        # x_T=x_start_list[0]
        # r_T=r_list[5]
        # if FLAGS.add_scale_noise==1:
        #     scale_T=scale_list[-3]

        # print(sigma_list)




        return x_T,r_T,trans_T,scale_T,x_list,beta_list

    def ddim_sample_3(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = default(noise, torch.randn(batch, dim,3, device = device))
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        trans_list=[]
        scale_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T, cond)
            model_in_ts=(x_T,trans_T,scale_T,cond,point)
            x_noise = self.model.forward_latent(model_in, time_cond)

            trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)

            x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
            trans_start=self.predict_start_from_noise(trans_T,time_cond,trans_noise)
            scale_start=self.predict_start_from_noise(scale_T,time_cond,scale_noise)

            x_T=alpha_next.sqrt()*(x_T-(1-alpha).sqrt()*x_noise)/(alpha.sqrt()) \
                +(1-alpha_next).sqrt()*x_noise
            trans_T=alpha_next.sqrt()*(trans_T-(1-alpha).sqrt()*trans_noise)/(alpha.sqrt()) \
                    +(1-alpha_next).sqrt()*trans_noise
            scale_T=alpha_next.sqrt()*(scale_T-(1-alpha).sqrt()*scale_noise)/(alpha.sqrt()) \
                    +(1-alpha_next).sqrt()*scale_noise


            x_list.append(x_T)
            beta_list.append(1-alpha)
            trans_list.append(trans_T)
            scale_list.append(scale_T)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        trans_T=trans_list.mean(0)
        scale_T=scale_list.mean(0)




        return x_T,trans_T,scale_T,x_list,beta_list

    def sample(self, dim, batch_size, point=None, noise=None, clip_denoised = True, traj=False, cond=None):

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        x_T = default(noise, torch.randn(batch, dim, 3,device = device))
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)

        first_x=None
        first_trans=None
        first_scale=None
        use_first=False
        if use_first:
            self.num_timesteps=1
        for t in reversed(range(0, self.num_timesteps)):

            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_in = (x_T, cond)
            model_in_ts=(x_T,trans_T,scale_T,cond,point)

            if FLAGS.pred=='noise':
                x_noise = self.model.forward_latent(model_in, time_cond)

                trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)

                x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
                trans_start=self.predict_start_from_noise(trans_T,time_cond,trans_noise)
                scale_start=self.predict_start_from_noise(scale_T,time_cond,scale_noise)
            else:
                x_start = self.model.forward_latent(model_in, time_cond)
                trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)
            if use_first:
                first_x=x_start
                first_trans=trans_start
                first_scale=scale_start
                use_first=False
            if clip_denoised:
                x_start.clamp_(-1., 1.)
                trans_start.clamp_(-1., 1.)
                scale_start.clamp_(-1., 1.)

            def get_next(x_start,x_T,time_cond):
                model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)
                noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0
                x_T = model_mean + (0.5 * model_log_variance).exp() * noise
                return x_T

            x_T=get_next(x_start,x_T,time_cond)
            trans_T=get_next(trans_start,trans_T,time_cond)
            scale_T=get_next(scale_start,scale_T,time_cond)
        if use_first:
            return first_x,first_trans,first_scale
        else:
            return x_T,trans_T,scale_T,None,None

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_lik(x_start))
        #noise = torch.clamp(noise, min=-6.0, max=6.0)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # main function for calculating loss
    def forward(self, x_start, r_start,trans_start, scale_start, t,
                ret_pred_x=False, noise = None, cond=None,point=None,cls_vector=None):
        '''
        x_start: [B, D]
        t: [B]
        '''

        noise = default(noise, lambda: torch.randn_like(x_start))
        noise_r=torch.randn_like(r_start)
        noise_trans = torch.randn_like(trans_start)
        noise_scale = torch.randn_like(scale_start)


        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_r=self.q_sample(x_start=r_start, t=t, noise=noise_r)
        x_tran = self.q_sample(x_start=trans_start, t=t, noise=noise_trans)
        x_scale = self.q_sample(x_start=scale_start, t=t, noise=noise_scale)

        model_in = (x, x_r, cond,cls_vector) if cond is not None else x
        model_in_ts=(x,x_r,x_tran,x_scale,cond,point,cls_vector)
        out_latent,out_r = self.model.forward_latent(model_in, t)
        out_trans,out_scale = self.model.forward_ts(model_in_ts, t)

        if FLAGS.pred == 'noise':
            target = noise
            trans_target=noise_trans
            scale_target=noise_scale
        elif FLAGS.pred == 'start':
            r_target=r_start
            target = x_start
            trans_target=trans_start
            scale_target=scale_start[:,None]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss_latent = self.loss_fn(out_latent, target, reduction = 'none')
        loss_r=self.loss_fn(out_r, r_target, reduction = 'none')
        loss_trans = self.loss_fn(out_trans, trans_target, reduction = 'none')
        loss_scale = self.loss_fn(out_scale, scale_target, reduction = 'none')
        #loss = reduce(loss, 'b ... -> b (...)', 'mean', b = x_start.shape[0]) # only one dim of latent so don't need this line

        loss_latent = loss_latent * extract(self.p2_loss_weight, t, loss_latent.shape)

        unreduced_loss_latent = loss_latent.detach().clone().mean(-1)
        unreduced_loss_r = loss_r.detach().clone().mean(-1)
        unreduced_loss_trans = loss_trans.detach().clone().mean(-1)
        unreduced_loss_scale = loss_scale.detach().clone().mean(-1)

        loss=loss_latent.mean()+loss_r.mean()+loss_trans.mean()+loss_scale.mean()
        return loss,unreduced_loss_latent,unreduced_loss_r,unreduced_loss_trans,unreduced_loss_scale

    def model_predictions(self, model_input, t):

        #model_output1 = self.model(model_input, t, pass_cond=0)
        #model_output2 = self.model(model_input, t, pass_cond=1)
        #model_output = model_output2*5 - model_output1*4
        model_output = self.model(model_input, t, pass_cond=1)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)



    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    def diffusion_model_from_latent_ts(self, x_start, r_vector, trans_start, scale_start,  cond=None,point=None,cls_vector=None):
        #if self.perturb_pc is None and cond is not None:
        #    print("check whether to pass condition!!!")

        # STEP 1: sample timestep
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # STEP 2: perturb condition

        # STEP 3: pass to forward function
        loss, unreduced_loss_latent,unreduced_loss_r,unreduced_loss_trans,unreduced_loss_scale = self(x_start,r_vector, trans_start, scale_start, t, cond=cond,point=point,
                                                                                     cls_vector=cls_vector,ret_pred_x=True)
        loss_100_latent = unreduced_loss_latent[t<100].mean().detach()
        loss_100_r = unreduced_loss_r[t<100].mean().detach()
        loss_100_trans = unreduced_loss_trans[t<100].mean().detach()
        loss_100_scale = unreduced_loss_scale[t<100].mean().detach()
        if torch.isnan(loss_100_r).any():
            loss_100_r=-1
        if torch.isnan(loss_100_latent).any():
            loss_100_latent=-1
        if torch.isnan(loss_100_trans).any():
            loss_100_trans=-1
        if torch.isnan(loss_100_scale).any():
            loss_100_scale=-1
        loss_1000_latent = unreduced_loss_latent[t>100].mean().detach()
        loss_1000_r = unreduced_loss_r[t>100].mean().detach()
        loss_1000_trans = unreduced_loss_trans[t>100].mean().detach()
        loss_1000_scale = unreduced_loss_scale[t>100].mean().detach()
        if torch.isnan(loss_1000_latent).any():
            loss_1000_latent=-1
        if torch.isnan(loss_1000_r).any():
            loss_1000_r=-1
        if torch.isnan(loss_1000_trans).any():
            loss_1000_trans=-1
        if torch.isnan(loss_1000_scale).any():
            loss_1000_scale=-1


        return loss, loss_100_latent,loss_100_r,loss_100_trans,loss_100_scale, \
               loss_1000_latent,loss_1000_r,loss_1000_trans,loss_1000_scale


    def generate_from_pc(self, pc, load_pc=False, batch=5, save_pc=False, return_pc=False, ddim=False, perturb_pc=True):
        self.eval()

        with torch.no_grad():
            if load_pc:
                pc = sample_pc(pc, self.pc_size).cuda().unsqueeze(0)

            if pc is None:
                input_pc = None
                save_pc = False
                full_perturbed_pc = None

            else:
                if perturb_pc:
                    full_perturbed_pc = perturb_point_cloud(pc, self.perturb_pc)
                    perturbed_pc = full_perturbed_pc[:, torch.randperm(full_perturbed_pc.shape[1])[:self.pc_size] ]
                    input_pc = perturbed_pc.repeat(batch, 1, 1)
                else:
                    full_perturbed_pc = pc
                    perturbed_pc = pc
                    input_pc = pc.repeat(batch, 1, 1)

            #print("shapes: ", pc.shape, self.pc_size, self.perturb_pc, perturbed_pc.shape, full_perturbed_pc.shape)
            #print("pc path: ", pc_path)

            #print("pc shape: ", perturbed_pc.shape, input_pc.shape)
            # if save_pc: # save perturbed pc ply file for visualization
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(perturbed_pc.cpu().numpy().squeeze())
            #     o3d.io.write_point_cloud("{}/input_pc.ply".format(save_pc), pcd)

            sample_fn = self.ddim_sample if ddim else self.sample
            samp,_ = sample_fn(dim=self.model.dim_in_out, batch_size=batch, traj=False, cond=input_pc)

        if return_pc:
            return samp, perturbed_pc
        return samp

    def generate_unconditional(self, num_samples):
        self.eval()
        with torch.no_grad():
            samp,_ = self.sample(dim=self.model.dim_in_out, batch_size=num_samples, traj=False, cond=None)

        return samp

    def generate_from_cond(self,cond,point,cls_vector,batch=1):
        self.eval()

        latent_pred,r_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_2(dim=768, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        return latent_pred,r_pred,trans_pred,scale_pred,x_list,beta_list





class DiffusionModel_v2(nn.Module):
    def __init__(
            self,
            model,
            timesteps = 1000, sampling_timesteps = None, beta_schedule = 'cosine',
            sample_pc_size = 128, perturb_pc = 'partial',  crop_percent=0.5,
            loss_type = 'l2', objective = 'pred_x0',
            data_scale = 1.0, data_shift = 0.0,
            p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k = 1,
            ddim_sampling_eta = 1.
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc
        self.crop_percent = crop_percent
        assert self.perturb_pc in [None, "partial", "noisy"]

        self.loss_fn = F.l1_loss if loss_type=='l1' else F.mse_loss

        # sampling related parameters
        self.sampling_timesteps = 10 # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # self.register_buffer('data_scale', torch.tensor(data_scale))
        # self.register_buffer('data_shift', torch.tensor(data_shift))

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))


        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )


    # def ddim_sample(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False, cond=None):
    #     batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
    #     times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
    #     times = list(reversed(times.int().tolist()))
    #     time_pairs = list(zip(times[:-1], times[1:]))
    #
    #     traj = []
    #
    #     x_T = default(noise, torch.randn(batch, dim,3, device = device))
    #     trans_T=torch.randn(batch,3,device = device)
    #     scale_T=torch.randn(batch,device = device)
    #     cond=cond.repeat(batch,1,1,1)
    #     point=point.repeat(batch,1,1)
    #
    #     sigma_list=[]
    #     for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
    #         alpha = self.alphas_cumprod[time]
    #         alpha_next = self.alphas_cumprod[time_next]
    #
    #         time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
    #
    #         model_in = (x_T, cond)
    #         model_in_ts=(x_T,trans_T,scale_T,cond,point)
    #         x_start = self.model.forward_latent(model_in, time_cond)
    #
    #         trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)
    #
    #         x_noise=self.predict_noise_from_start(x_T,time_cond,x_start)
    #         trans_noise=self.predict_noise_from_start(trans_T,time_cond,trans_start)
    #         scale_noise=self.predict_noise_from_start(scale_T,time_cond,scale_start)
    #
    #         if clip_denoised:
    #             x_start.clamp_(-1., 1.)
    #         eta=1
    #         sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    #         # sigma = 0
    #         c = ((1 - alpha_next) - sigma ** 2).sqrt()
    #
    #         noise_1 = torch.randn_like(x_T)
    #         noise_2 = torch.randn_like(trans_T)
    #         noise_3 = torch.randn_like(scale_T)
    #
    #         x_T = x_start * alpha_next.sqrt() + \
    #               c * x_noise + \
    #               sigma * noise_1
    #         trans_T = trans_start * alpha_next.sqrt() + \
    #               c * trans_noise + \
    #               sigma * noise_2
    #         scale_T = scale_start * alpha_next.sqrt() + \
    #               c * scale_noise + \
    #               sigma * noise_3
    #         sigma_list.append(sigma)
    #     print(sigma_list)
    #
    #
    #
    #
    #     return x_T,trans_T,scale_T

    def ddim_sample_2(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        interval=np.floor((total_timesteps-1)/(sampling_timesteps-1))
        times=np.arange(sampling_timesteps)*interval
        times=times.astype(np.int)
        times=times.tolist()
        if times[-1]<total_timesteps-1:
            times.append(total_timesteps-1)

        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = torch.randn(batch, dim,3,device = device)
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        x_start_list=[]
        trans_list=[]
        scale_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T, cond)
            model_in_ts=(x_T,trans_T,scale_T,cond,point)
            x_start = self.model.forward_latent(model_in, time_cond)

            trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)

            x_start_list.append(x_start)
            trans_list.append(trans_start)
            scale_list.append(scale_start)
            x_noise=self.predict_noise_from_start(x_T,time_cond,x_start)
            trans_noise=self.predict_noise_from_start(trans_T,time_cond,trans_start)
            scale_noise=self.predict_noise_from_start(scale_T,time_cond,scale_start)

            # if clip_denoised:
            #     x_start.clamp_(-1., 1.)
            #     trans_start.clamp_(-0.5, 0.5)
            #     scale_start.clamp_(0, 1)
            eta=1
            sigma =  ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()*eta
            c = 1 - alpha_next - sigma ** 2
            c=c/(1-alpha)
            a=alpha_next.sqrt()-(alpha*c).sqrt()
            b=c.sqrt()


            noise_1 = torch.randn_like(x_T) if time > 0 else 0
            noise_2 = torch.randn_like(trans_T) if time > 0 else 0
            noise_3 = torch.randn_like(scale_T) if time > 0 else 0

            x_T = x_start * a + x_T*b+ sigma * noise_1
            trans_T = trans_start * a + trans_T*b+ sigma * noise_2
            scale_T = scale_start * a + scale_T*b+ sigma * noise_3
            x_list.append(x_T)
            beta_list.append(1-alpha)
            sigma_list.append(sigma)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_start_list=torch.stack(x_start_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        trans_T=trans_list.mean(0)
        scale_T=scale_list.mean(0)
        x_T=x_start_list.mean(0)

        # print(sigma_list)




        return x_T,trans_T,scale_T,x_list,beta_list

    def ddim_sample_3(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = default(noise, torch.randn(batch, dim,3, device = device))
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        trans_list=[]
        scale_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T, cond)
            model_in_ts=(x_T,trans_T,scale_T,cond,point)
            x_noise = self.model.forward_latent(model_in, time_cond)

            trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)

            x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
            trans_start=self.predict_start_from_noise(trans_T,time_cond,trans_noise)
            scale_start=self.predict_start_from_noise(scale_T,time_cond,scale_noise)

            x_T=alpha_next.sqrt()*(x_T-(1-alpha).sqrt()*x_noise)/(alpha.sqrt())\
                +(1-alpha_next).sqrt()*x_noise
            trans_T=alpha_next.sqrt()*(trans_T-(1-alpha).sqrt()*trans_noise)/(alpha.sqrt()) \
                +(1-alpha_next).sqrt()*trans_noise
            scale_T=alpha_next.sqrt()*(scale_T-(1-alpha).sqrt()*scale_noise)/(alpha.sqrt()) \
                +(1-alpha_next).sqrt()*scale_noise


            x_list.append(x_T)
            beta_list.append(1-alpha)
            trans_list.append(trans_T)
            scale_list.append(scale_T)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        # trans_T=trans_list.mean(0)
        # scale_T=scale_list.mean(0)




        return x_T,trans_T,scale_T,x_list,beta_list

    def sample(self, dim, batch_size, point=None, noise=None, clip_denoised = True, traj=False, cond=None):

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        x_T = default(noise, torch.randn(batch, dim, 3,device = device))
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)

        first_x=None
        first_trans=None
        first_scale=None
        use_first=False
        if use_first:
            self.num_timesteps=1
        for t in reversed(range(0, self.num_timesteps)):

            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_in = (x_T, cond)
            model_in_ts=(x_T,trans_T,scale_T,cond,point)

            if FLAGS.pred=='noise':
                x_noise = self.model.forward_latent(model_in, time_cond)

                trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)

                x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
                trans_start=self.predict_start_from_noise(trans_T,time_cond,trans_noise)
                scale_start=self.predict_start_from_noise(scale_T,time_cond,scale_noise)
            else:
                x_start = self.model.forward_latent(model_in, time_cond)
                trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)
            if use_first:
                first_x=x_start
                first_trans=trans_start
                first_scale=scale_start
                use_first=False
            if clip_denoised:
                x_start.clamp_(-1., 1.)
                trans_start.clamp_(-1., 1.)
                scale_start.clamp_(-1., 1.)

            def get_next(x_start,x_T,time_cond):
                model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)
                noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0
                x_T = model_mean + (0.5 * model_log_variance).exp() * noise
                return x_T

            x_T=get_next(x_start,x_T,time_cond)
            trans_T=get_next(trans_start,trans_T,time_cond)
            scale_T=get_next(scale_start,scale_T,time_cond)
        if use_first:
            return first_x,first_trans,first_scale
        else:
            return x_T,trans_T,scale_T,None,None

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_lik(x_start))
        #noise = torch.clamp(noise, min=-6.0, max=6.0)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # main function for calculating loss
    def forward(self, x_start, trans_start, scale_start, t,  ret_pred_x=False, noise = None, cond=None,point=None):
        '''
        x_start: [B, D]
        t: [B]
        '''

        noise = default(noise, lambda: torch.randn_like(x_start))
        noise_trans = torch.randn_like(trans_start)
        noise_scale = torch.randn_like(scale_start)


        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_tran = self.q_sample(x_start=trans_start, t=t, noise=noise_trans)
        x_scale = self.q_sample(x_start=scale_start, t=t, noise=noise_scale)

        model_in = (x, cond) if cond is not None else x
        model_in_ts=(x,x_tran,x_scale,cond,point)
        model_out = self.model.forward_latent(model_in, t)
        out_trans,out_scale = self.model.forward_ts(model_in_ts, t)

        if FLAGS.pred == 'noise':
            target = noise
            trans_target=noise_trans
            scale_target=noise_scale
        elif FLAGS.pred == 'start':
            target = x_start
            trans_target=trans_start
            scale_target=scale_start

        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss_latent = self.loss_fn(model_out, target, reduction = 'none')
        loss_trans = self.loss_fn(out_trans, trans_target, reduction = 'none')
        loss_scale = self.loss_fn(out_scale, scale_target, reduction = 'none')
        #loss = reduce(loss, 'b ... -> b (...)', 'mean', b = x_start.shape[0]) # only one dim of latent so don't need this line

        loss_latent = loss_latent * extract(self.p2_loss_weight, t, loss_latent.shape)

        unreduced_loss_latent = loss_latent.detach().clone().mean(-1).mean(-1)
        unreduced_loss_trans = loss_trans.detach().clone().mean(-1)
        unreduced_loss_scale = loss_scale.detach().clone()

        loss=loss_latent.mean()+loss_trans.mean()+loss_scale.mean()
        return loss,unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale

    def model_predictions(self, model_input, t):

        #model_output1 = self.model(model_input, t, pass_cond=0)
        #model_output2 = self.model(model_input, t, pass_cond=1)
        #model_output = model_output2*5 - model_output1*4
        model_output = self.model(model_input, t, pass_cond=1)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)



    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    def diffusion_model_from_latent_ts(self, x_start, trans_start, scale_start,  cond=None,point=None):
        #if self.perturb_pc is None and cond is not None:
        #    print("check whether to pass condition!!!")

        # STEP 1: sample timestep
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # STEP 2: perturb condition

        # STEP 3: pass to forward function
        loss, unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale = self(x_start, trans_start, scale_start, t, cond=cond,point=point,ret_pred_x=True)
        loss_100_latent = unreduced_loss_latent[t<100].mean().detach()
        loss_100_trans = unreduced_loss_trans[t<100].mean().detach()
        loss_100_scale = unreduced_loss_scale[t<100].mean().detach()
        if torch.isnan(loss_100_latent).any():
            loss_100_latent=-1
        if torch.isnan(loss_100_trans).any():
            loss_100_trans=-1
        if torch.isnan(loss_100_scale).any():
            loss_100_scale=-1
        loss_1000_latent = unreduced_loss_latent[t>100].mean().detach()
        loss_1000_trans = unreduced_loss_trans[t>100].mean().detach()
        loss_1000_scale = unreduced_loss_scale[t>100].mean().detach()
        if torch.isnan(loss_1000_latent).any():
            loss_1000_latent=-1
        if torch.isnan(loss_1000_trans).any():
            loss_1000_trans=-1
        if torch.isnan(loss_1000_scale).any():
            loss_1000_scale=-1


        return loss, loss_100_latent,loss_100_trans,loss_100_scale,\
               loss_1000_latent,loss_1000_trans,loss_1000_scale


    def generate_from_pc(self, pc, load_pc=False, batch=5, save_pc=False, return_pc=False, ddim=False, perturb_pc=True):
        self.eval()

        with torch.no_grad():
            if load_pc:
                pc = sample_pc(pc, self.pc_size).cuda().unsqueeze(0)

            if pc is None:
                input_pc = None
                save_pc = False
                full_perturbed_pc = None

            else:
                if perturb_pc:
                    full_perturbed_pc = perturb_point_cloud(pc, self.perturb_pc)
                    perturbed_pc = full_perturbed_pc[:, torch.randperm(full_perturbed_pc.shape[1])[:self.pc_size] ]
                    input_pc = perturbed_pc.repeat(batch, 1, 1)
                else:
                    full_perturbed_pc = pc
                    perturbed_pc = pc
                    input_pc = pc.repeat(batch, 1, 1)

            #print("shapes: ", pc.shape, self.pc_size, self.perturb_pc, perturbed_pc.shape, full_perturbed_pc.shape)
            #print("pc path: ", pc_path)

            #print("pc shape: ", perturbed_pc.shape, input_pc.shape)
            # if save_pc: # save perturbed pc ply file for visualization
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(perturbed_pc.cpu().numpy().squeeze())
            #     o3d.io.write_point_cloud("{}/input_pc.ply".format(save_pc), pcd)

            sample_fn = self.ddim_sample if ddim else self.sample
            samp,_ = sample_fn(dim=self.model.dim_in_out, batch_size=batch, traj=False, cond=input_pc)

        if return_pc:
            return samp, perturbed_pc
        return samp

    def generate_unconditional(self, num_samples):
        self.eval()
        with torch.no_grad():
            samp,_ = self.sample(dim=self.model.dim_in_out, batch_size=num_samples, traj=False, cond=None)

        return samp

    def generate_from_cond(self,cond,point,batch=1):
        self.eval()
        # latent_pred,trans_pred,scale_pred,x_list,beta_list = self.sample(dim=self.model.dim_in_out, batch_size=batch, point=point,traj=False, cond=cond)
        if FLAGS.pred=='noise':
            latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_3(dim=self.model.dim_in_out, batch_size=batch, point=point,traj=False, cond=cond)
        else:
            latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_2(dim=self.model.dim_in_out, batch_size=batch, point=point,traj=False, cond=cond)
        return latent_pred,trans_pred,scale_pred,x_list,beta_list





class DiffusionModel_v3(nn.Module):
    def __init__(
            self,
            model,
            timesteps = 1000, sampling_timesteps = None, beta_schedule = 'cosine',
            sample_pc_size = 128, perturb_pc = 'partial',  crop_percent=0.5,
            loss_type = 'l2', objective = 'pred_x0',
            data_scale = 1.0, data_shift = 0.0,
            p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k = 1,
            ddim_sampling_eta = 1.
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc
        self.crop_percent = crop_percent
        assert self.perturb_pc in [None, "partial", "noisy"]

        self.loss_fn = F.l1_loss if loss_type=='l1' else F.mse_loss

        # sampling related parameters
        self.sampling_timesteps = 20# default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # self.register_buffer('data_scale', torch.tensor(data_scale))
        # self.register_buffer('data_shift', torch.tensor(data_shift))

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))


        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )



    def ddim_sample_2(self, dim, batch_size, point=None,noise=None, clip_denoised = True,
                      traj=False, cond=None,cls_vector=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        interval=np.floor((total_timesteps-2)/(sampling_timesteps-1))
        times=np.arange(sampling_timesteps)*interval+1
        times=times.astype(np.int)
        times=times.tolist()
        # if times[-1]<total_timesteps-1:
        #     times.append(total_timesteps-1)

        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = torch.randn(batch, dim,3,device = device)
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)
        cls_vector=cls_vector.repeat(batch,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        x_start_list=[]
        trans_list=[]
        scale_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T, cond,cls_vector)
            model_in_ts=(x_T,trans_T,scale_T,cond,point,cls_vector)

            if FLAGS.pred=='start':
                x_start = self.model.forward_latent(model_in, time_cond)

            else:
                x_noise=self.model.forward_latent(model_in, time_cond)
                x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
            trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)


            x_start_list.append(x_start)
            trans_list.append(trans_start)
            scale_list.append(scale_start)


            # if clip_denoised:
            #     x_start.clamp_(-1., 1.)
            #     trans_start.clamp_(-0.5, 0.5)
            #     scale_start.clamp_(0, 1)
            eta=1
            sigma =  ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()*eta
            c = 1 - alpha_next - sigma ** 2
            c=c/(1-alpha)
            a=alpha_next.sqrt()-(alpha*c).sqrt()
            b=c.sqrt()

            # a=0.8
            # b=0.2


            noise_1 = torch.randn_like(x_T) if time > 0 else 0
            noise_2 = torch.randn_like(trans_T) if time > 0 else 0
            noise_3 = torch.randn_like(scale_T) if time > 0 else 0

            x_T = x_start * a + x_T*b+ sigma * noise_1
            trans_T = trans_start * a + trans_T*b+ sigma * noise_2
            scale_T = scale_start * a + scale_T*b+ sigma * noise_3
            x_list.append(x_T)
            beta_list.append(1-alpha)
            sigma_list.append(sigma)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_start_list=torch.stack(x_start_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        trans_T=trans_list.mean(0)
        scale_T=scale_list.mean(0)
        # x_T=x_start_list.mean(0)
        # x_T=x_start_list[0]
        # x_list=x_start_list

        # print(sigma_list)




        return x_T,trans_T,scale_T,x_list,beta_list

    def ddim_sample_3(self, dim, batch_size, point=None,noise=None, clip_denoised = True,
                      traj=False, cond=None,cls_vector=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        interval=np.floor((total_timesteps-2)/(sampling_timesteps-1))
        times=np.arange(sampling_timesteps)*interval+1
        times=times.astype(np.int)
        times=times.tolist()
        # if times[-1]<total_timesteps-1:
        #     times.append(total_timesteps-1)

        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = torch.randn(batch, dim,3,device = device)
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)
        cls_vector=cls_vector.repeat(batch,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        x_start_list=[]
        trans_list=[]
        scale_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T, cond,cls_vector)
            model_in_ts=(x_T,trans_T,scale_T,cond,point,cls_vector)
            x_noise = self.model.forward_latent(model_in, time_cond)

            trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)



            x_T=alpha_next.sqrt()*(x_T-(1-alpha).sqrt()*x_noise)/(alpha.sqrt()) \
                +(1-alpha_next).sqrt()*x_noise
            trans_T=alpha_next.sqrt()*(trans_T-(1-alpha).sqrt()*trans_noise)/(alpha.sqrt()) \
                    +(1-alpha_next).sqrt()*trans_noise
            scale_T=alpha_next.sqrt()*(scale_T-(1-alpha).sqrt()*scale_noise)/(alpha.sqrt()) \
                    +(1-alpha_next).sqrt()*scale_noise


            x_list.append(x_T)
            beta_list.append(1-alpha)
            trans_list.append(trans_T)
            scale_list.append(scale_T)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        # trans_T=trans_list.mean(0)
        # scale_T=scale_list.mean(0)




        return x_T,trans_T,scale_T,x_list,beta_list

    def sample(self, dim, batch_size, point=None, noise=None, clip_denoised = True, traj=False, cond=None,cls_vector=None,):

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        x_T = default(noise, torch.randn(batch, dim, 3,device = device))
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)
        cls_vector=cls_vector.repeat(batch,1)

        first_x=None
        first_trans=None
        first_scale=None
        use_first=False
        if use_first:
            self.num_timesteps=1
        for t in reversed(range(0, self.num_timesteps)):

            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_in = (x_T, cond,cls_vector)
            model_in_ts=(x_T,trans_T,scale_T,cond,point,cls_vector)

            if FLAGS.pred=='noise':
                x_noise = self.model.forward_latent(model_in, time_cond)

                trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)

                x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
                trans_start=self.predict_start_from_noise(trans_T,time_cond,trans_noise)
                scale_start=self.predict_start_from_noise(scale_T,time_cond,scale_noise)
            else:
                x_start = self.model.forward_latent(model_in, time_cond)
                trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)
            if use_first:
                first_x=x_start
                first_trans=trans_start
                first_scale=scale_start
                use_first=False
            if clip_denoised:
                x_start.clamp_(-1., 1.)
                trans_start.clamp_(-1., 1.)
                scale_start.clamp_(-1., 1.)

            def get_next(x_start,x_T,time_cond):
                model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)
                noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0
                x_T = model_mean + (0.5 * model_log_variance).exp() * noise
                return x_T

            x_T=get_next(x_start,x_T,time_cond)
            trans_T=get_next(trans_start,trans_T,time_cond)
            scale_T=get_next(scale_start,scale_T,time_cond)
        if use_first:
            return first_x,first_trans,first_scale
        else:
            return x_T,trans_T,scale_T,None,None

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_lik(x_start))
        #noise = torch.clamp(noise, min=-6.0, max=6.0)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # main function for calculating loss
    def forward(self, x_start, trans_start, scale_start, t,  ret_pred_x=False, noise = None, cond=None,
                point=None,cls_vector=None):
        '''
        x_start: [B, D]
        t: [B]
        '''

        noise = default(noise, lambda: torch.randn_like(x_start))
        noise_trans = torch.randn_like(trans_start)
        noise_scale = torch.randn_like(scale_start)


        x = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_tran = self.q_sample(x_start=trans_start, t=t, noise=noise_trans)
        x_scale = self.q_sample(x_start=scale_start, t=t, noise=noise_scale)

        model_in = (x, cond,cls_vector) if cond is not None else x
        model_in_ts=(x,x_tran,x_scale,cond,point,cls_vector)
        model_out = self.model.forward_latent(model_in, t)
        out_trans,out_scale = self.model.forward_ts(model_in_ts, t)

        target = x_start
        trans_target=trans_start
        scale_target=scale_start

        pred_model=model_out
        pred_tran=out_trans
        pred_scale=out_scale

        if FLAGS.pred == 'epsilnoise':
            target=noise
            trans_target=trans_target
            scale_target=scale_target






        loss_latent = self.loss_fn(pred_model, target, reduction = 'none')
        loss_trans = self.loss_fn(pred_tran, trans_target, reduction = 'none')
        loss_scale = self.loss_fn(pred_scale, scale_target, reduction = 'none')
        #loss = reduce(loss, 'b ... -> b (...)', 'mean', b = x_start.shape[0]) # only one dim of latent so don't need this line

        loss_latent = loss_latent * extract(self.p2_loss_weight, t, loss_latent.shape)

        unreduced_loss_latent = loss_latent.detach().clone().mean(-1).mean(-1)
        unreduced_loss_trans = loss_trans.detach().clone().mean(-1)
        unreduced_loss_scale = loss_scale.detach().clone()

        loss=loss_latent.mean()+loss_trans.mean()+loss_scale.mean()
        return loss,unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale

    def model_predictions(self, model_input, t):

        #model_output1 = self.model(model_input, t, pass_cond=0)
        #model_output2 = self.model(model_input, t, pass_cond=1)
        #model_output = model_output2*5 - model_output1*4
        model_output = self.model(model_input, t, pass_cond=1)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)



    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    def diffusion_model_from_latent_ts(self, x_start, trans_start, scale_start,  cond=None,point=None,cls_vector=None):
        #if self.perturb_pc is None and cond is not None:
        #    print("check whether to pass condition!!!")

        # STEP 1: sample timestep
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # STEP 2: perturb condition

        # STEP 3: pass to forward function
        loss, unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale = self(x_start, trans_start, scale_start, t, cond=cond,point=point,
                                                                                     cls_vector=cls_vector,ret_pred_x=True)
        loss_100_latent = unreduced_loss_latent[t<100].mean().detach()
        loss_100_trans = unreduced_loss_trans[t<100].mean().detach()
        loss_100_scale = unreduced_loss_scale[t<100].mean().detach()
        if torch.isnan(loss_100_latent).any():
            # print('nan !!')
            loss_100_latent=-1
        if torch.isnan(loss_100_trans).any():
            # print('nan !!')
            loss_100_trans=-1
        if torch.isnan(loss_100_scale).any():
            # print('nan !!')
            loss_100_scale=-1
        loss_1000_latent = unreduced_loss_latent[t>100].mean().detach()
        loss_1000_trans = unreduced_loss_trans[t>100].mean().detach()
        loss_1000_scale = unreduced_loss_scale[t>100].mean().detach()
        if torch.isnan(loss_1000_latent).any():
            loss_1000_latent=-1
        if torch.isnan(loss_1000_trans).any():
            loss_1000_trans=-1
        if torch.isnan(loss_1000_scale).any():
            loss_1000_scale=-1


        return loss, loss_100_latent,loss_100_trans,loss_100_scale, \
               loss_1000_latent,loss_1000_trans,loss_1000_scale


    def generate_from_pc(self, pc, load_pc=False, batch=5, save_pc=False, return_pc=False, ddim=False, perturb_pc=True):
        self.eval()

        with torch.no_grad():
            if load_pc:
                pc = sample_pc(pc, self.pc_size).cuda().unsqueeze(0)

            if pc is None:
                input_pc = None
                save_pc = False
                full_perturbed_pc = None

            else:
                if perturb_pc:
                    full_perturbed_pc = perturb_point_cloud(pc, self.perturb_pc)
                    perturbed_pc = full_perturbed_pc[:, torch.randperm(full_perturbed_pc.shape[1])[:self.pc_size] ]
                    input_pc = perturbed_pc.repeat(batch, 1, 1)
                else:
                    full_perturbed_pc = pc
                    perturbed_pc = pc
                    input_pc = pc.repeat(batch, 1, 1)

            #print("shapes: ", pc.shape, self.pc_size, self.perturb_pc, perturbed_pc.shape, full_perturbed_pc.shape)
            #print("pc path: ", pc_path)

            #print("pc shape: ", perturbed_pc.shape, input_pc.shape)
            # if save_pc: # save perturbed pc ply file for visualization
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(perturbed_pc.cpu().numpy().squeeze())
            #     o3d.io.write_point_cloud("{}/input_pc.ply".format(save_pc), pcd)

            sample_fn = self.ddim_sample if ddim else self.sample
            samp,_ = sample_fn(dim=self.model.dim_in_out, batch_size=batch, traj=False, cond=input_pc)

        if return_pc:
            return samp, perturbed_pc
        return samp

    def generate_unconditional(self, num_samples):
        self.eval()
        with torch.no_grad():
            samp,_ = self.sample(dim=self.model.dim_in_out, batch_size=num_samples, traj=False, cond=None)

        return samp

    def generate_from_cond(self,cond,point,cls_vector,batch=1):
        self.eval()

        # latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_2(dim=256, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        if FLAGS.pred is not 'noise':
            latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_2(dim=256, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        else:
            latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_3(dim=256, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        # latent_pred,trans_pred,scale_pred,x_list,beta_list = self.sample(dim=256, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        return latent_pred,trans_pred,scale_pred,x_list,beta_list



def random_time(min_time,
                max_time,batch):


    time = (min_time/max_time + torch.rand(batch) * (1-min_time/max_time))*max_time   # Shape: (1,)
    #time = torch.exp(torch.rand_like(max_time) * (torch.log(max_time)-torch.log(min_time)) + torch.log(min_time))
    return time




class DiffusionModel_score(nn.Module):
    def __init__(
            self,
            model,
            timesteps = 1000, sampling_timesteps = None, beta_schedule = 'cosine',
            sample_pc_size = 128, perturb_pc = 'partial',  crop_percent=0.5,
            loss_type = 'l2', objective = 'pred_x0',
            data_scale = 1.0, data_shift = 0.0,
            p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
            p2_loss_weight_k = 1,
            ddim_sampling_eta = 1.
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        self.min_time=0.003
        self.max_time=1
        self.mult=2.5
        betas = linear_beta_schedule(timesteps) if beta_schedule == 'linear' else cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.pc_size = sample_pc_size
        self.perturb_pc = perturb_pc
        self.crop_percent = crop_percent
        assert self.perturb_pc in [None, "partial", "noisy"]

        self.loss_fn = F.l1_loss if loss_type=='l1' else F.mse_loss

        # sampling related parameters
        self.sampling_timesteps = 20 # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # self.register_buffer('data_scale', torch.tensor(data_scale))
        # self.register_buffer('data_shift', torch.tensor(data_shift))

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))


        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        register_buffer('posterior_variance', posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (x0 - extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )



    def ddim_sample_2(self, dim, batch_size, point=None,noise=None, clip_denoised = True,
                      traj=False, cond=None,cls_vector=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        interval=np.floor((total_timesteps-1)/(sampling_timesteps-1))
        times=np.arange(sampling_timesteps)*interval
        times=times.astype(np.int)
        times=times.tolist()
        if times[-1]<total_timesteps-1:
            times.append(total_timesteps-1)

        times = list(reversed(times))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = torch.randn(batch, dim,3,device = device)
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)
        cls_vector=cls_vector.repeat(batch,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        x_start_list=[]
        trans_list=[]
        scale_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T, cond,cls_vector)
            model_in_ts=(x_T,trans_T,scale_T,cond,point,cls_vector)
            x_start = self.model.forward_latent(model_in, time_cond)

            trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)

            x_start_list.append(x_start)
            trans_list.append(trans_start)
            scale_list.append(scale_start)
            x_noise=self.predict_noise_from_start(x_T,time_cond,x_start)
            trans_noise=self.predict_noise_from_start(trans_T,time_cond,trans_start)
            scale_noise=self.predict_noise_from_start(scale_T,time_cond,scale_start)

            # if clip_denoised:
            #     x_start.clamp_(-1., 1.)
            #     trans_start.clamp_(-0.5, 0.5)
            #     scale_start.clamp_(0, 1)
            eta=1
            sigma =  ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()*eta
            c = 1 - alpha_next - sigma ** 2
            c=c/(1-alpha)
            a=alpha_next.sqrt()-(alpha*c).sqrt()
            b=c.sqrt()

            # a=0.8
            # b=0.2


            noise_1 = torch.randn_like(x_T) if time > 0 else 0
            noise_2 = torch.randn_like(trans_T) if time > 0 else 0
            noise_3 = torch.randn_like(scale_T) if time > 0 else 0

            x_T = x_start * a + x_T*b+ sigma * noise_1
            trans_T = trans_start * a + trans_T*b+ sigma * noise_2
            scale_T = scale_start * a + scale_T*b+ sigma * noise_3
            x_list.append(x_T)
            beta_list.append(1-alpha)
            sigma_list.append(sigma)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_start_list=torch.stack(x_start_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        trans_T=trans_list.mean(0)
        scale_T=scale_list.mean(0)
        # x_T=x_start_list.mean(0)
        # x_T=x_start_list[0]

        # print(sigma_list)




        return x_T,trans_T,scale_T,x_list,beta_list

    def ddim_sample_3(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False, cond=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = batch_size, self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(0., total_timesteps, steps = sampling_timesteps + 2)[:-1]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        traj = []

        x_T = default(noise, torch.randn(batch, dim,3, device = device))
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        trans_list=[]
        scale_list=[]
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)

            model_in = (x_T, cond)
            model_in_ts=(x_T,trans_T,scale_T,cond,point)
            x_noise = self.model.forward_latent(model_in, time_cond)

            trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)

            x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
            trans_start=self.predict_start_from_noise(trans_T,time_cond,trans_noise)
            scale_start=self.predict_start_from_noise(scale_T,time_cond,scale_noise)

            x_T=alpha_next.sqrt()*(x_T-(1-alpha).sqrt()*x_noise)/(alpha.sqrt()) \
                +(1-alpha_next).sqrt()*x_noise
            trans_T=alpha_next.sqrt()*(trans_T-(1-alpha).sqrt()*trans_noise)/(alpha.sqrt()) \
                    +(1-alpha_next).sqrt()*trans_noise
            scale_T=alpha_next.sqrt()*(scale_T-(1-alpha).sqrt()*scale_noise)/(alpha.sqrt()) \
                    +(1-alpha_next).sqrt()*scale_noise


            x_list.append(x_T)
            beta_list.append(1-alpha)
            trans_list.append(trans_T)
            scale_list.append(scale_T)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        trans_T=trans_list.mean(0)
        scale_T=scale_list.mean(0)




        return x_T,trans_T,scale_T,x_list,beta_list

    def sample(self, dim, batch_size, point=None, noise=None, clip_denoised = True, traj=False, cond=None,cls_vector=None,):

        batch, device, objective = batch_size, self.betas.device, self.objective

        traj = []

        x_T = default(noise, torch.randn(batch, dim, 3,device = device))
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)
        cls_vector=cls_vector.repeat(batch,1)

        first_x=None
        first_trans=None
        first_scale=None
        use_first=False
        if use_first:
            self.num_timesteps=1
        for t in reversed(range(0, self.num_timesteps)):

            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_in = (x_T, cond,cls_vector)
            model_in_ts=(x_T,trans_T,scale_T,cond,point,cls_vector)

            if FLAGS.pred=='noise':
                x_noise = self.model.forward_latent(model_in, time_cond)

                trans_noise,scale_noise = self.model.forward_ts(model_in_ts, time_cond)

                x_start=self.predict_start_from_noise(x_T,time_cond,x_noise)
                trans_start=self.predict_start_from_noise(trans_T,time_cond,trans_noise)
                scale_start=self.predict_start_from_noise(scale_T,time_cond,scale_noise)
            else:
                x_start = self.model.forward_latent(model_in, time_cond)
                trans_start,scale_start = self.model.forward_ts(model_in_ts, time_cond)
            if use_first:
                first_x=x_start
                first_trans=trans_start
                first_scale=scale_start
                use_first=False
            if clip_denoised:
                x_start.clamp_(-1., 1.)
                trans_start.clamp_(-1., 1.)
                scale_start.clamp_(-1., 1.)

            def get_next(x_start,x_T,time_cond):
                model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x_T, t = time_cond)
                noise = torch.randn_like(x_T) if t > 0 else 0. # no noise if t == 0
                x_T = model_mean + (0.5 * model_log_variance).exp() * noise
                return x_T

            x_T=get_next(x_start,x_T,time_cond)
            trans_T=get_next(trans_start,trans_T,time_cond)
            scale_T=get_next(scale_start,scale_T,time_cond)
        if use_first:
            return first_x,first_trans,first_scale
        else:
            return x_T,trans_T,scale_T,None,None

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped


    # "nice property": return x_t given x_0, noise, and timestep
    def q_sample(self, x_start, t, noise=None):

        noise = default(noise, lambda: torch.randn_lik(x_start))
        #noise = torch.clamp(noise, min=-6.0, max=6.0)

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # main function for calculating loss
    def forward(self, x_start, trans_start, scale_start, t,  ret_pred_x=False, noise = None, cond=None,
                point=None,cls_vector=None):
        '''
        x_start: [B, D]
        t: [B]
        '''
        t_norm=t/1.0 * 10000.0
        sigma_min=0.01
        sigma_max=90
        std=sigma_min * (sigma_max / sigma_min) ** t
        noise = torch.randn_like(x_start)*std[:,None,None]
        noise_trans = torch.randn_like(trans_start)*std[:,None]
        noise_scale = torch.randn_like(scale_start)*std

        bs=x_start.shape[0]
        x = x_start+noise
        x_tran = trans_start + noise_trans
        x_scale = scale_start + noise_scale

        model_in = (x, cond,cls_vector) if cond is not None else x
        model_in_ts=(x,x_tran,x_scale,cond,point,cls_vector)
        model_out = self.model.forward_latent(model_in, t_norm)
        out_trans,out_scale = self.model.forward_ts(model_in_ts, t_norm)


        target = -noise/std[:,None,None]**2
        trans_target= -noise_trans/std[:,None]**2
        scale_target= -noise_scale/std**2


        loss_weighting = std ** 2
        loss_latent = torch.mean((loss_weighting[:,None,None] * (model_out/ (std+1e-7)[:,None,None] - target)**2).reshape(bs, -1), dim=-1)
        loss_trans= torch.mean((loss_weighting[:,None] * (out_trans/ (std+1e-7)[:,None]  - trans_target)**2).reshape(bs, -1), dim=-1)
        loss_scale= torch.mean((loss_weighting * (out_scale/(std+1e-7)- scale_target)**2).reshape(bs, -1), dim=-1)




        unreduced_loss_latent = loss_latent.detach().clone()
        unreduced_loss_trans = loss_trans.detach().clone()
        unreduced_loss_scale = loss_scale.detach().clone()

        loss=loss_latent.mean()+loss_trans.mean()+loss_scale.mean()
        return loss,unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale

    def model_predictions(self, model_input, t):

        #model_output1 = self.model(model_input, t, pass_cond=0)
        #model_output2 = self.model(model_input, t, pass_cond=1)
        #model_output = model_output2*5 - model_output1*4
        model_output = self.model(model_input, t, pass_cond=1)

        x = model_input[0] if type(model_input) is tuple else model_input

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, model_output)

        elif self.objective == 'pred_x0':
            pred_noise = self.predict_noise_from_start(x, t, model_output)
            x_start = model_output

        return ModelPrediction(pred_noise, x_start)



    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    def diffusion_model_from_latent_ts(self, x_start, trans_start, scale_start,  cond=None,point=None,cls_vector=None):
        #if self.perturb_pc is None and cond is not None:
        #    print("check whether to pass condition!!!")

        # STEP 1: sample timestep
        eps=1e-5
        t= torch.rand(x_start.shape[0], device=x_start.device) * (1. - eps) + eps

        # STEP 2: perturb condition

        # STEP 3: pass to forward function
        loss, unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale = self(x_start, trans_start, scale_start, t, cond=cond,point=point,
                                                                                     cls_vector=cls_vector,ret_pred_x=True)
        loss_100_latent = unreduced_loss_latent[t<0.1].mean().detach()
        loss_100_trans = unreduced_loss_trans[t<0.1].mean().detach()
        loss_100_scale = unreduced_loss_scale[t<0.1].mean().detach()
        if torch.isnan(loss_100_latent).any():
            # print('nan !!')
            loss_100_latent=-1
        if torch.isnan(loss_100_trans).any():
            # print('nan !!')
            loss_100_trans=-1
        if torch.isnan(loss_100_scale).any():
            # print('nan !!')
            loss_100_scale=-1
        loss_1000_latent = unreduced_loss_latent[t>0.1].mean().detach()
        loss_1000_trans = unreduced_loss_trans[t>0.1].mean().detach()
        loss_1000_scale = unreduced_loss_scale[t>0.1].mean().detach()
        if torch.isnan(loss_1000_latent).any():
            loss_1000_latent=-1
        if torch.isnan(loss_1000_trans).any():
            loss_1000_trans=-1
        if torch.isnan(loss_1000_scale).any():
            loss_1000_scale=-1


        return loss, loss_100_latent,loss_100_trans,loss_100_scale, \
               loss_1000_latent,loss_1000_trans,loss_1000_scale


    def generate_from_pc(self, pc, load_pc=False, batch=5, save_pc=False, return_pc=False, ddim=False, perturb_pc=True):
        self.eval()

        with torch.no_grad():
            if load_pc:
                pc = sample_pc(pc, self.pc_size).cuda().unsqueeze(0)

            if pc is None:
                input_pc = None
                save_pc = False
                full_perturbed_pc = None

            else:
                if perturb_pc:
                    full_perturbed_pc = perturb_point_cloud(pc, self.perturb_pc)
                    perturbed_pc = full_perturbed_pc[:, torch.randperm(full_perturbed_pc.shape[1])[:self.pc_size] ]
                    input_pc = perturbed_pc.repeat(batch, 1, 1)
                else:
                    full_perturbed_pc = pc
                    perturbed_pc = pc
                    input_pc = pc.repeat(batch, 1, 1)

            #print("shapes: ", pc.shape, self.pc_size, self.perturb_pc, perturbed_pc.shape, full_perturbed_pc.shape)
            #print("pc path: ", pc_path)

            #print("pc shape: ", perturbed_pc.shape, input_pc.shape)
            # if save_pc: # save perturbed pc ply file for visualization
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(perturbed_pc.cpu().numpy().squeeze())
            #     o3d.io.write_point_cloud("{}/input_pc.ply".format(save_pc), pcd)

            sample_fn = self.ddim_sample if ddim else self.sample
            samp,_ = sample_fn(dim=self.model.dim_in_out, batch_size=batch, traj=False, cond=input_pc)

        if return_pc:
            return samp, perturbed_pc
        return samp

    def generate_unconditional(self, num_samples):
        self.eval()
        with torch.no_grad():
            samp,_ = self.sample(dim=self.model.dim_in_out, batch_size=num_samples, traj=False, cond=None)

        return samp

    def generate_from_cond(self,cond,point,cls_vector,batch=1):
        self.eval()

        latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_2(dim=256, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        # latent_pred,trans_pred,scale_pred,x_list,beta_list = self.sample(dim=256, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        return latent_pred,trans_pred,scale_pred,x_list,beta_list




class DiffusionModel_v4(nn.Module):
    def __init__(
            self,
            model,
            timesteps = 1000

    ):
        super().__init__()
        self.loss_fn=F.mse_loss
        self.model = model
        self.noise_scheduler=DDPMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,beta_schedule='scaled_linear',
                                           variance_type='fixed_small',clip_sample=False)
        if FLAGS.pred=='noise':
            prediction_type='epsilon'
        else:
            prediction_type='sample'
        self.denoise_scheduler=DDIMScheduler(num_train_timesteps=1000,beta_start=0.00085,beta_end=0.012,beta_schedule='scaled_linear',
                                             clip_sample=False,set_alpha_to_one=False,steps_offset=1,prediction_type=prediction_type)



    def forward(self, x_start, trans_start, scale_start, t,  ret_pred_x=False, noise = None, cond=None,
                point=None,cls_vector=None):
        '''
        x_start: [B, D]
        t: [B]
        '''

        noise = default(noise, lambda: torch.randn_like(x_start))
        noise_trans = torch.randn_like(trans_start)
        noise_scale = torch.randn_like(scale_start)

        x=self.noise_scheduler.add_noise(x_start,noise,t)
        x_tran=self.noise_scheduler.add_noise(trans_start,noise_trans,t)
        x_scale=self.noise_scheduler.add_noise(scale_start,noise_scale,t)


        model_in = (x, cond,cls_vector)
        model_in_ts=(x,x_tran,x_scale,cond,point,cls_vector)
        model_out = self.model.forward_latent(model_in, t)
        out_trans,out_scale = self.model.forward_ts(model_in_ts, t)

        target = x_start
        trans_target=trans_start
        scale_target=scale_start

        pred_model=model_out
        pred_tran=out_trans
        pred_scale=out_scale

        if FLAGS.pred == 'noise':
            target=noise
            trans_target=noise_trans
            scale_target=noise_scale






        loss_latent = self.loss_fn(pred_model, target, reduction = 'none')
        loss_trans = self.loss_fn(pred_tran, trans_target, reduction = 'none')
        loss_scale = self.loss_fn(pred_scale, scale_target, reduction = 'none')
        #loss = reduce(loss, 'b ... -> b (...)', 'mean', b = x_start.shape[0]) # only one dim of latent so don't need this line


        unreduced_loss_latent = loss_latent.detach().clone().mean(-1).mean(-1)
        unreduced_loss_trans = loss_trans.detach().clone().mean(-1)
        unreduced_loss_scale = loss_scale.detach().clone()

        loss=loss_latent.mean()+loss_trans.mean()+loss_scale.mean()
        return loss,unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale





    # a wrapper function that only takes x_start (clean modulation vector) and condition
    # does everything including sampling timestep and returns loss, loss_100, loss_1000, prediction
    def diffusion_model_from_latent_ts(self, x_start, trans_start, scale_start,  cond=None,point=None,cls_vector=None):
        #if self.perturb_pc is None and cond is not None:
        #    print("check whether to pass condition!!!")

        # STEP 1: sample timestep
        t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (x_start.shape[0],), device=x_start.device).long()

        # STEP 2: perturb condition

        # STEP 3: pass to forward function
        loss, unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale = self(x_start, trans_start, scale_start, t, cond=cond,point=point,
                                                                                     cls_vector=cls_vector,ret_pred_x=True)
        loss_100_latent = unreduced_loss_latent[t<100].mean().detach()
        loss_100_trans = unreduced_loss_trans[t<100].mean().detach()
        loss_100_scale = unreduced_loss_scale[t<100].mean().detach()
        if torch.isnan(loss_100_latent).any():
            # print('nan !!')
            loss_100_latent=-1
        if torch.isnan(loss_100_trans).any():
            # print('nan !!')
            loss_100_trans=-1
        if torch.isnan(loss_100_scale).any():
            # print('nan !!')
            loss_100_scale=-1
        loss_1000_latent = unreduced_loss_latent[t>100].mean().detach()
        loss_1000_trans = unreduced_loss_trans[t>100].mean().detach()
        loss_1000_scale = unreduced_loss_scale[t>100].mean().detach()
        if torch.isnan(loss_1000_latent).any():
            loss_1000_latent=-1
        if torch.isnan(loss_1000_trans).any():
            loss_1000_trans=-1
        if torch.isnan(loss_1000_scale).any():
            loss_1000_scale=-1


        return loss, loss_100_latent,loss_100_trans,loss_100_scale, \
               loss_1000_latent,loss_1000_trans,loss_1000_scale


    def ddim_sample_3(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False, cond=None,
                      cls_vector=None):
        batch, device = batch_size, cond.device
        steps=50
        eta=0

        self.denoise_scheduler.set_timesteps(steps)
        timesteps = self.denoise_scheduler.timesteps.to(device)
        x_T = torch.randn(batch, dim,3, device = device)
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)
        cls_vector=cls_vector.repeat(batch,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        trans_list=[]
        scale_list=[]
        for t in tqdm(timesteps, desc = 'sampling loop time step'):


            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)

            model_in = (x_T, cond,cls_vector)
            model_in_ts=(x_T,trans_T,scale_T,cond,point,cls_vector)
            x_pred = self.model.forward_latent(model_in, time_cond)

            trans_pred,scale_pred = self.model.forward_ts(model_in_ts, time_cond)

            x_T = self.denoise_scheduler.step(
                x_pred, t, x_T, eta=eta,generator=None,
            ).prev_sample

            trans_T = self.denoise_scheduler.step(
                trans_pred, t, trans_T, eta=eta,generator=None,
            ).prev_sample

            scale_T = self.denoise_scheduler.step(
                scale_pred, t, scale_T, eta=eta,generator=None,
            ).prev_sample

            alpha=self.denoise_scheduler.alphas_cumprod[t]

            x_list.append(x_T)
            beta_list.append(1-alpha)
            trans_list.append(trans_T)
            scale_list.append(scale_T)
        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        # trans_T=trans_list.mean(0)
        # scale_T=scale_list.mean(0)
        # x_T=x_list[193]



        return x_T,trans_T,scale_T,x_list,beta_list

    def generate_from_cond(self,cond,point,cls_vector,batch=1):
        self.eval()

        latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_3(dim=256, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        return latent_pred,trans_pred,scale_pred,x_list,beta_list



