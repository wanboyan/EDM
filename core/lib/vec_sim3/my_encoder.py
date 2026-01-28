import torch

from layers_equi import *



def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def channel_equi_vec_normalize(x):
    # B,C,3,...
    assert x.ndim >= 3, "x shape [B,C,3,...]"
    x_dir = F.normalize(x, dim=2)
    dim=torch.tensor(x.shape[1]).float()
    x_norm = x.norm(dim=2, keepdim=True)
    x_normalized_norm = F.normalize(x_norm, dim=1)  # normalize across C
    y = x_dir * x_normalized_norm*torch.sqrt(dim)
    return y
class VNN_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, meta_output=None):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        self.scale=10.0
        self.conv_pos = VNLinearLeakyReLU(3, 128, negative_slope=0.2, share_nonlinearity=False, use_batchnorm=False)
        self.fc_pos = VNLinear(128, 2*hidden_dim)
        self.block_0 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.2, share_nonlinearity=False)
        self.pool = meanpool
        self.fc_mu = VNLinear(c_dim, c_dim)  # for plane features resolution 64x64, spatial resolution is 2x2 after the last encoder layer
        self.fc_var = nn.Linear(c_dim, c_dim)
        self.fc_inv = VNLinear(c_dim, c_dim)
        self.fc_inv_2 = VNLinear(c_dim, c_dim)
        self.kl_std=0.25
    def forward(self, p):
        batch_size = p.size(0)
        p = p.unsqueeze(1)*self.scale
        #mean = get_graph_mean(p, k=self.k)
        #mean = p_trans.mean(dim=-1, keepdim=True).expand(p_trans.size())
        feat = get_graph_feature_cross(p, k=self.k)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)

        net = self.fc_pos(net)

        net = self.block_0(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))

        c=channel_equi_vec_normalize(c)
        c_inv_dual = self.fc_inv(c[..., None]).squeeze(-1)
        c_inv = (channel_equi_vec_normalize(c_inv_dual) * c).sum(-1)
        mu = self.fc_mu(c)
        log_var = self.fc_var(c_inv).unsqueeze(-1).repeat(1,1,3)

        z_so3=self.reparameterize(mu,log_var)
        z_inv_dual = self.fc_inv_2(z_so3[..., None]).squeeze(-1)
        z_inv = (z_inv_dual * z_so3).sum(-1)
        kl_loss=self.loss_function(mu,log_var)
        return z_so3,z_inv,kl_loss

    def get_inv(self, z_so3):
        z_inv_dual = self.fc_inv_2(z_so3[..., None]).squeeze(-1)
        z_inv = (z_inv_dual * z_so3).sum(-1)
        return z_so3,z_inv

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def loss_function(self,mu,log_var):
            std = torch.exp(0.5 * log_var)
            gt_dist = torch.distributions.normal.Normal( torch.zeros_like(mu), torch.ones_like(std)*self.kl_std )
            sampled_dist = torch.distributions.normal.Normal( mu, std )
            #gt_dist = normal_dist.sample(log_var.shape)
            #print("gt dist shape: ", gt_dist.shape)

            kl = torch.distributions.kl.kl_divergence(sampled_dist, gt_dist) # reversed KL
            kl_loss = reduce(kl, 'b ... -> b (...)', 'mean').mean()

            return kl_loss