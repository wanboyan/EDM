import numpy as np
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class VADLogVar(nn.Module):
    def __init__(self, N, dim):
        super(VADLogVar, self).__init__()
        self.N = N
        self.dim = dim
        self.weight_mu = nn.Parameter(torch.Tensor(N, dim))
        self.weight_logvar = nn.Parameter(torch.Tensor(N, dim))
        self.reset_parameters()
        print('[VADLogVar Embedding] #entries: {}; #dims: {};'.format(self.N, self.dim))
        
    def reset_parameters(self):
        mu_init_std = 1.0 / np.sqrt(self.dim)
        torch.nn.init.normal_(
            self.weight_mu.data,
            0.0,
            mu_init_std,
        )
        logvar_init_std = 1.0 / np.sqrt(self.dim)
        torch.nn.init.normal_(
            self.weight_logvar.data,
            0,
            logvar_init_std,
        )

    def forward(self, idx, **kwargs):
        num_augment_pts = kwargs['num_augment_pts']
        mu = self.weight_mu[idx]
        logvar = self.weight_logvar[idx]
        logvar = logvar.detach()
        std = torch.exp(0.5*logvar)
        if self.training:
            eps = torch.randn_like(std)
            batch_latent = mu + eps*std
            batch_latent_aug = batch_latent
            return {'latent_code': batch_latent, 'latent_code_augment': batch_latent_aug, 'mu': mu, 'logvar': logvar, 'std': std}
        else:
            print('[VADLogVar Embedding] Test mode forward')
            batch_latent = mu
            return {'latent_code': batch_latent, 'mu': mu, 'logvar': logvar, 'std': std}

class AD(nn.Module):
    def __init__(self, cfg, N, dim):
        super(AD, self).__init__()
        self.cfg = cfg
        self.N = N
        self.dim = dim
        self.embed_params = nn.Parameter(torch.Tensor(N, dim))
        self.reset_parameters()
        print('[AD Embedding] #entries: {}; #dims: {}; cfg: {}'.format(self.N, self.dim, self.cfg))

    def reset_parameters(self):
        if self.cfg.init_std is None:
            init_std = 1.0 / np.sqrt(self.dim)
        else:
            init_std = self.cfg.init_std
        torch.nn.init.normal_(
            self.embed_params.data,
            0.0,
            init_std,
        )
        
    def _normalize(self, idx):
        batch_embed = self.embed_params[idx].detach()
        batch_norms = torch.sqrt(torch.sum(batch_embed.data**2, dim=-1, keepdim=True))
        batch_scale_factors = torch.clamp(batch_norms, self.cfg.max_norm, None)
        batch_embed_normalized = batch_embed / batch_scale_factors
        self.embed_params.data[idx] = batch_embed_normalized
        
    def forward(self, idx, **kwargs):
        if getattr(self.cfg, 'max_norm', None):
            self._normalize(idx)
        batch_embed = self.embed_params[idx]
        
        return {'latent_code': batch_embed}
        
