import torch
import torch.nn as nn
import numpy as np

class RPE(nn.Module):
    def __init__(self, nhead, point_cloud_range, quan_size=0.02):
        '''
        Args:
            point_cloud_range: (6), [xmin, ymin, zmin, xmax, ymax, zmax]
        '''
        super().__init__()

        self.nhead = nhead
        point_cloud_range = np.array(point_cloud_range)
        point_cloud_range = point_cloud_range[3:6] - point_cloud_range[0:3]
        point_cloud_range = (point_cloud_range**2).sum()**0.5
        self.max_len = int(point_cloud_range // quan_size + 1)
        self.grid_size = quan_size

        self.pos_embed = nn.Embedding(self.max_len, self.nhead)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, batch_rel_coords,pred_scale):
        """
        Args:
            batch_rel_coords: (B, N, 3)
        Returns
            pos_embedding: (B, N, nhead)
        """
        pred_scale=pred_scale.reshape(-1,1)
        dist = torch.norm(batch_rel_coords, dim=-1)  # (B, N)

        if pred_scale is not None:
            dist=dist/pred_scale

        dist = dist / self.grid_size

        idx1 = dist.long()
        idx2 = idx1 + 1
        w1 = idx2.type_as(dist) - dist
        w2 = dist - idx1.type_as(dist)

        idx1[idx1 >= self.max_len] = self.max_len - 1
        idx2[idx2 >= self.max_len] = self.max_len - 1

        embed1 = self.pos_embed(idx1)  # (B, N, nhead)
        embed2 = self.pos_embed(idx2)  # (B, N, nhead)

        embed = embed1 * w1.unsqueeze(-1) + embed2 * w2.unsqueeze(-1)  # (B, N, nhead)

        return embed

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(SinusoidalPositionalEmbedding, self).__init__()
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        div_indices = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(div_indices * (-np.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)

    def forward(self, emb_indices):
        r"""Sinusoidal Positional Embedding.

        Args:
            emb_indices: torch.Tensor (*)

        Returns:
            embeddings: torch.Tensor (*, D)
        """
        input_shape = emb_indices.shape
        omegas = emb_indices.view(-1, 1, 1) * self.div_term.view(1, -1, 1)  # (-1, d_model/2, 1)
        sin_embeddings = torch.sin(omegas)
        cos_embeddings = torch.cos(omegas)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=2)  # (-1, d_model/2, 2)
        embeddings = embeddings.view(*input_shape, self.d_model)  # (*, d_model)
        embeddings = embeddings.detach()
        return embeddings







class DirRPE(nn.Module):
    def __init__(self, nhead):
        '''
        Args:
            point_cloud_range: (6), [xmin, ymin, zmin, xmax, ymax, zmax]
        '''
        super().__init__()
        hidden_dim=16
        self.nhead = nhead
        self.sigma_d = 0.002
        self.sigma_a = 1
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.embedding = SinusoidalPositionalEmbedding(hidden_dim)
        self.proj_d = nn.Linear(hidden_dim, nhead)
        self.proj_a = nn.Linear(hidden_dim, nhead)
        self.reduction_a = 'max'
    def forward(self, rel_d,rel_a) :
        dist = torch.norm(rel_d, dim=-1)
        dist=dist/self.sigma_d
        angular=rel_a*self.factor_a

        d_embeddings = self.embedding(dist)
        d_embeddings = self.proj_d(d_embeddings)

        a_embeddings = self.embedding(angular)
        a_embeddings = self.proj_a(a_embeddings)

        if self.reduction_a == 'max':
            a_embeddings = a_embeddings.max(dim=2)[0]
        else:
            a_embeddings = a_embeddings.mean(dim=2)
        embed=a_embeddings+d_embeddings
        return embed