import torch
import torch.utils.data 
from torch.nn import functional as F
import pytorch_lightning as pl

# add paths in model/__init__.py for new models
from equi_diff_models import *
import open3d as o3d
class CombinedModel(pl.LightningModule):
    def __init__(self, specs):
        super().__init__()
        self.specs = specs

        self.task = specs['training_task'] # 'combined' or 'modulation' or 'diffusion'

        if self.task in ('combined', 'modulation'):
            self.sdf_model = SdfModel(specs=specs) 

            feature_dim = specs["SdfModelSpecs"]["latent_dim"] # latent dim of pointnet 
            modulation_dim = feature_dim*3 # latent dim of modulation
            latent_std = specs.get("latent_std", 0.25) # std of target gaussian distribution of latent space
            hidden_dims = [modulation_dim, modulation_dim, modulation_dim, modulation_dim, modulation_dim]
            self.vae_model = BetaVAE(in_channels=feature_dim*3, latent_dim=modulation_dim, hidden_dims=hidden_dims, kl_std=latent_std)


    def training_step(self, x, idx):

        if self.task == 'combined':
            return self.train_combined(x)
        elif self.task == 'modulation':
            return self.train_modulation(x)
        elif self.task == 'diffusion':
            return self.train_diffusion(x)
        

    def configure_optimizers(self):

        if self.task == 'combined':
            params_list = [
                    { 'params': list(self.sdf_model.parameters()) + list(self.vae_model.parameters()), 'lr':self.specs['sdf_lr'] },
                    { 'params': self.diffusion_model.parameters(), 'lr':self.specs['diff_lr'] }
                ]
        elif self.task == 'modulation':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['sdf_lr'] }
                ]
        elif self.task == 'diffusion':
            params_list = [
                    { 'params': self.parameters(), 'lr':self.specs['diff_lr'] }
                ]

        optimizer = torch.optim.Adam(params_list)
        return {
                "optimizer": optimizer,
                # "lr_scheduler": {
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=50000, threshold=0.0002, min_lr=1e-6, verbose=False),
                # "monitor": "total"
                # }
        }


    #-----------different training steps for sdf modulation, diffusion, combined----------

    def train_modulation(self, x):

        # xyz = x['xyz'] # (B, N, 3)
        xyz=torch.cat([x['points.nss'],x['points.uni']],dim=1)
        # gt = x['gt_sdf'] # (B, N)
        gt=torch.cat([x['points.nss.value'],x['points.uni.value']],dim=1)
        # pc = x['point_cloud'] # (B, 1024, 3)
        pc=x['inputs']


        # STEP 1: obtain reconstructed plane feature and latent code 
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)
        out = self.vae_model(original_features) # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]

        # STEP 2: pass recon back to GenSDF pipeline 
        pred_sdf = self.sdf_model.forward_with_plane_features(reconstructed_plane_feature, xyz)
        
        # STEP 3: losses for VAE and SDF
        # we only use the KL loss for the VAE; no reconstruction loss
        try:
            vae_loss = self.vae_model.loss_function(*out, M_N=self.specs["kld_weight"] )
        except:
            print("vae loss is nan at epoch {}...".format(self.current_epoch))
            return None # skips this batch

        sdf_loss = F.l1_loss(pred_sdf.squeeze(), gt.squeeze(), reduction='none')
        sdf_loss = reduce(sdf_loss, 'b ... -> b (...)', 'mean').mean()

        loss = sdf_loss + vae_loss

        loss_dict =  {"sdf": sdf_loss, "vae": vae_loss}
        self.log_dict(loss_dict, prog_bar=True, enable_graph=False)

        return loss


    def generate(self,x,ckp_index,latent_dir=None):
        xyz=torch.cat([x['points.nss'],x['points.uni']],dim=1).cuda()
        # gt = x['gt_sdf'] # (B, N)
        gt=torch.cat([x['points.nss.value'],x['points.uni.value']],dim=1).cuda()
        # pc = x['point_cloud'] # (B, 1024, 3)
        pc=x['inputs'].cuda()
        model_id=x['model_id'][0]


        # STEP 1: obtain reconstructed plane feature and latent code
        plane_features = self.sdf_model.pointnet.get_plane_features(pc)
        original_features = torch.cat(plane_features, dim=1)

        if latent_dir:
            z=self.vae_model.get_latent(original_features)
            np.save(os.path.join(latent_dir,f'{model_id}.npy'),z[0].detach().cpu().numpy())
            return

        out = self.vae_model(original_features)

        # out = [self.decode(z), input, mu, log_var, z]
        reconstructed_plane_feature, latent = out[0], out[-1]
        vertices,faces=self.sdf_model.generate_mesh_2(reconstructed_plane_feature)

        pcd=o3d.geometry.PointCloud()
        pcd.points=o3d.utility.Vector3dVector(pc[0].detach().cpu().numpy())
        mesh=o3d.geometry.TriangleMesh.create_coordinate_frame()
        mesh.vertices=o3d.utility.Vector3dVector(vertices)
        mesh.triangles=o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        frame=o3d.geometry.TriangleMesh.create_coordinate_frame()
        frame.scale(1, center=(0,0,0))
        o3d.visualization.draw_geometries([mesh,frame,pcd],mesh_show_back_face=True)
    def get_mesh_from_latent(self,latent,thred=0.02):
        reconstructed_plane_feature = self.vae_model.decode(latent)

        vertices,faces,normals=self.sdf_model.generate_mesh_2(reconstructed_plane_feature,thred)

        return vertices,faces,normals



