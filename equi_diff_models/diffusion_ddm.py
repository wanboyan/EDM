import math
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from collections import namedtuple
from functools import partial

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



class DiffusionModel_ddm(nn.Module):
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
        self.sampling_timesteps = 5 # default num sampling timesteps to number of timesteps at training
        assert self.sampling_timesteps <= timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # self.register_buffer('data_scale', torch.tensor(data_scale))
        # self.register_buffer('data_shift', torch.tensor(data_shift))

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        self.eps = 1e-3
        self.sigma_min = 1e-3
        self.sigma_max = 1

    def q_sample(self, x_start, noise, t, C):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x_noisy = x_start + C * time + time * noise
        return x_noisy


    def pred_x0_from_xt(self, xt, noise, C, t):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        x0 = xt - C * time - time * noise
        return x0


    def pred_xtms_from_xt(self, xt, noise, C, t, s):
        time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
        mean = xt + C * (time-s) - C * time - s / torch.sqrt(time) * noise
        epsilon = torch.randn_like(mean, device=xt.device)
        sigma = torch.sqrt(s * (time-s) / time)
        xtms = mean + sigma * epsilon
        return xtms

    def ddim_sample_2(self, dim, batch_size, point=None,noise=None, clip_denoised = True, traj=False,
                      cond=None,cls_vector=None):
        batch, device,  = batch_size, cond.device
        cls_vector=cls_vector.repeat(batch,1)
        sampling_timesteps=self.sampling_timesteps
        step = 1. / self.sampling_timesteps
        step_indices = torch.arange(sampling_timesteps, dtype=torch.float32, device=device)
        t_steps = (self.sigma_max + step_indices / (sampling_timesteps - 1) * (
                step - self.sigma_max))
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        time_steps = -torch.diff(t_steps)
        traj = []

        x_T = torch.randn(batch, dim,3,device = device)
        trans_T=torch.randn(batch,3,device = device)
        scale_T=torch.randn(batch,device = device)
        cond=cond.repeat(batch,1,1,1)
        point=point.repeat(batch,1,1)

        sigma_list=[]
        beta_list=[]
        x_list=[]
        trans_list=[]
        scale_list=[]
        cur_time = torch.ones((batch,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            model_in = (x_T, cond,cls_vector)
            model_in_ts=(x_T,trans_T,scale_T,cond,point,cls_vector)

            C_pose_ = self.model.forward_latent(model_in, cur_time.log())
            sigma = cur_time.reshape(x_T.shape[0], 1, 1)
            c_skip = (sigma - 1) / (sigma ** 2 + (sigma - 1) ** 2)
            c_out = sigma / (sigma ** 2 + (sigma - 1) ** 2).sqrt()
            C_pose = c_skip * x_T + c_out * C_pose_
            noise_pose = (x_T - (sigma - 1) * C_pose) / sigma
            # x_T = self.pred_xtms_from_xt(x_T, noise_pose, C_pose, cur_time, s)
            x0_pose = - C_pose
            grad = (x_T - x0_pose) / cur_time   #  TODO: shape match
            x_T = x0_pose + grad * (cur_time - s)

            C_trans_, C_scale_ = self.model.forward_ts(model_in_ts, cur_time.log())

            sigma = cur_time.reshape(x_T.shape[0], 1)
            # c_skip = (sigma - 1) / (sigma ** 2 - sigma + 1)
            # c_out = (sigma / (sigma ** 2 - sigma + 1)).sqrt()
            C_trans = c_skip * trans_T + c_out * C_trans_
            noise_trans = (trans_T - (sigma - 1) * C_trans) / sigma.sqrt()
            # trans_T = self.pred_xtms_from_xt(trans_T, noise_trans, C_trans, cur_time, s)
            x0_trans = - C_trans
            grad = (trans_T - x0_trans) / cur_time   #  TODO: shape match
            trans_T = x0_trans + grad * (cur_time - s)

            sigma = cur_time.reshape(x_T.shape[0])
            # c_skip = (sigma - 1) / (sigma ** 2 - sigma + 1)
            # c_out = (sigma / (sigma ** 2 - sigma + 1)).sqrt()
            C_scale = c_skip * scale_T + c_out * C_scale_
            noise_scale = (scale_T - (sigma - 1) * C_scale) / sigma.sqrt()
            # scale_T = self.pred_xtms_from_xt(scale_T, noise_scale, C_scale, cur_time, s)
            x0_scale = - C_scale
            grad = (scale_T - x0_scale) / cur_time   #  TODO: shape match
            scale_T = x0_scale + grad * (cur_time - s)

            x_list.append(x_T)
            beta_list.append(1)
            trans_list.append(trans_T)
            scale_list.append(scale_T)
            cur_time = cur_time - s

        trans_list=torch.stack(trans_list,dim=0)
        scale_list=torch.stack(scale_list,dim=0)
        x_list=torch.stack(x_list,dim=0)
        # trans_T=trans_list.mean(0)
        # scale_T=scale_list.mean(0)
        # x_T=x_list.mean(0)
        # print(sigma_list)




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


    # main function for calculating loss
    def forward(self, x_start, trans_start, scale_start, t,  ret_pred_x=False, noise = None, cond=None,point=None,cls_vector=None):
        '''
        x_start: [B, D]
        t: [B]
        '''
        noise = default(noise, lambda: torch.randn_like(x_start))
        noise_trans = torch.randn_like(trans_start)
        noise_scale = torch.randn_like(scale_start)

        C_pose = - x_start
        x_pose = self.q_sample(x_start=x_start, t=t, noise=noise, C=C_pose)
        C_tran = - trans_start
        x_tran = self.q_sample(x_start=trans_start, t=t, noise=noise_trans, C=C_tran)
        C_scale = - scale_start
        x_scale = self.q_sample(x_start=scale_start, t=t, noise=noise_scale, C=C_scale)

        ## precondition
        sigma = t.reshape(x_pose.shape[0], 1, 1)
        c_skip = (sigma - 1) / (sigma ** 2 + (sigma - 1) ** 2)
        c_out = sigma / (sigma ** 2 + (sigma - 1) ** 2).sqrt()

        model_in = (x_pose, cond,cls_vector) if cond is not None else x_pose
        model_in_ts = (x_pose, x_tran, x_scale, cond, point,cls_vector)
        C_pose_pred_ = self.model.forward_latent(model_in, t.log())

        C_pose_pred = c_skip * x_pose + c_out * C_pose_pred_

        noise_pose_pred = (x_pose - (t.reshape(x_pose.shape[0], 1, 1) - 1) * C_pose_pred) / t.reshape(x_pose.shape[0], 1, 1)

        C_trans_pred_, C_scale_pred_ = self.model.forward_ts(model_in_ts, t.log())

        C_trans_pred = c_skip[:,0,:] * x_tran + c_out[:,0,:] * C_trans_pred_
        C_scale_pred = c_skip[:,0,0] * x_scale + c_out[:,0,0]  * C_scale_pred_


        noise_trans_pred = (x_tran - (t.reshape(x_pose.shape[0], 1) - 1) * C_trans_pred) / t.reshape(x_pose.shape[0], 1)
        noise_scale_pred = (x_scale - (t.reshape(x_pose.shape[0]) - 1) * C_scale_pred) / t.reshape(x_pose.shape[0])

        simple_weight1 = ((sigma - 1) / sigma) ** 2 + 1
        simple_weight2 = (t / (1 - t + self.eps)) ** 2 + 1  # eps prevents div 0

        loss_latent = simple_weight1 * self.loss_fn(C_pose_pred, C_pose, reduction = 'none') #+ \
                      # simple_weight2 * self.loss_fn(noise_pose_pred, noise, reduction = 'none')

        loss_trans = simple_weight1[:,0,:] * self.loss_fn(C_trans_pred, C_tran, reduction = 'none') #+ \
                      # simple_weight2[:,0,:] * self.loss_fn(noise_trans_pred, noise_trans, reduction = 'none')

        loss_scale = simple_weight1[:,0,0] * self.loss_fn(C_scale_pred, C_scale, reduction = 'none') #+ \
                     # simple_weight2[:,0,0] * self.loss_fn(noise_scale_pred, noise_scale, reduction = 'none')


        unreduced_loss_latent = loss_latent.detach().clone().mean(-1).mean(-1)
        unreduced_loss_trans = loss_trans.detach().clone().mean(-1)
        unreduced_loss_scale = loss_scale.detach().clone()

        loss=loss_latent.sum()+loss_trans.sum()+loss_scale.sum()
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
        # t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device).long()
        eps = self.eps  # smallest time step
        t = torch.rand(x_start.shape[0], device=x_start.device) * (1. - eps) + eps
        # STEP 2: perturb condition

        # STEP 3: pass to forward function
        loss, unreduced_loss_latent,unreduced_loss_trans,unreduced_loss_scale = self(x_start, trans_start, scale_start, t, cond=cond,point=point,cls_vector=cls_vector,ret_pred_x=True)
        loss_100_latent = unreduced_loss_latent[t<0.1].mean().detach()
        loss_100_trans = unreduced_loss_trans[t<0.1].mean().detach()
        loss_100_scale = unreduced_loss_scale[t<0.1].mean().detach()
        if torch.isnan(loss_100_latent).any():
            loss_100_latent=-1
        if torch.isnan(loss_100_trans).any():
            loss_100_trans=-1
        if torch.isnan(loss_100_scale).any():
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

    def generate_from_cond(self,cond,point,cls_vector,batch=1):
        self.eval()
        # latent_pred,trans_pred,scale_pred,x_list,beta_list = self.sample(dim=self.model.dim_in_out, batch_size=batch, point=point,traj=False, cond=cond)

        latent_pred,trans_pred,scale_pred,x_list,beta_list = self.ddim_sample_2(dim=self.model.dim_in_out, batch_size=batch, point=point,traj=False, cond=cond,cls_vector=cls_vector)
        return latent_pred,trans_pred,scale_pred,x_list,beta_list