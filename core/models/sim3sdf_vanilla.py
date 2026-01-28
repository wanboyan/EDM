from core.lib.vec_sim3.vec_layers import VecLinear
from .model_base import ModelBase
import torch
import copy
import trimesh
from torch import nn
from scipy.spatial.transform import Rotation
import absl.flags as flags
FLAGS = flags.FLAGS

from core.lib.implicit_func.onet_decoder import Decoder, DecoderCBatchNorm, DecoderCat,Decoder_v2,Decoder_v3
from core.lib.vec_sim3.vec_dgcnn import VecDGCNN, VecDGCNN_v2
from core.lib.vec_sim3.vec_dgcnn_atten import VecDGCNN_att
from core.lib.vec_sim3.my_encoder import VNN_ResnetPointnet
import time
import logging
from .utils.occnet_utils import get_generator as get_mc_extractor
from .utils.ndf_utils.pcl_extractor import get_generator as get_udf_extractor
from .utils.misc import cfg_with_default, count_param
from torch import distributions as dist
import numpy as np

from core.models.utils.oflow_eval.evaluator import MeshEvaluator
from core.models.utils.oflow_common import eval_iou


class VaeModel(ModelBase):
    def __init__(self, cfg):
        network = SIM3Recon(cfg)
        super().__init__(cfg, network)
        self.cfg=cfg

        self.nss_th = cfg_with_default(cfg, ["model", "loss_th"], 1.0)

        self.use_udf = cfg_with_default(cfg, ["model", "use_udf"], False)



    def generate_mesh(self, embedding):
        self.mesh_extractor = get_mc_extractor(self.cfg)
        assert not self.use_udf
        net = self.network.module if self.__dataparallel_flag__ else self.network

        mesh = self.mesh_extractor.generate_from_latent(c=embedding, F=net.decode)
        if mesh.vertices.shape[0] == 0:
            mesh = trimesh.primitives.Box(extents=(1.0, 1.0, 1.0))
            logging.warning("Mesh extraction fail, replace by a place holder")
        return mesh
    def generate_mesh_2(self, embedding,thres=0.01):
        N=128
        max_batch=int(2 ** 14)
        voxel_origin = [-1, -1, -1]
        voxel_size = 2.0 / (N - 1)

        overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
        samples = torch.zeros(N ** 3, 4)
        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.long() / N) % N
        samples[:, 0] = ((overall_index.long() / N) / N) % N

        # transform first 3 columns
        # to be the x, y, z coordinate
        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        num_samples = N ** 3

        samples.requires_grad = False

        head = 0

        while head < num_samples:
            with torch.no_grad():
                sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()
                samples[head : min(head + max_batch, num_samples), 3]=self.network.decode(sample_subset.unsqueeze(0), None, embedding, return_sdf=True)
                head += max_batch

        sdf_values = samples[:, 3]
        sdf_values = sdf_values.reshape(N, N, N)
        import skimage.measure

        numpy_3d_sdf_tensor = sdf_values.data.cpu().numpy()
        verts, faces, normals, values = skimage.measure.marching_cubes(
            numpy_3d_sdf_tensor, level=thres, spacing=[voxel_size] * 3,allow_degenerate=True,
        )

        # transform from voxel coordinates to camera coordinates
        # note x and y are flipped in the output of marching_cubes
        mesh_points = np.zeros_like(verts)
        mesh_points[:, 0] = voxel_origin[0] + verts[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + verts[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + verts[:, 2]
        return mesh_points,faces,normals

    def generate_dense_surface_pts(self, embedding):
        net = self.network.module if self.__dataparallel_flag__ else self.network

        for param in self.network.network_dict["decoder"].parameters():
            param.requires_grad = False

        pcl = self.pcl_extractor.generate_from_latent(c=embedding, F=net.decode)
        if pcl is None:
            logging.warning("Dense PCL extraction fail, replace by a point at origin")
            pcl = np.array([[0.0, 0.0, 0.0]])

        for param in self.network.network_dict["decoder"].parameters():
            param.requires_grad = True

        return pcl

    def _postprocess_after_optim(self, batch):
        if "occ_hat_iou" in batch.keys() and not self.use_udf:
            # IOU is only directly computable when using sdf
            report = {}
            occ_pred = batch["occ_hat_iou"].unsqueeze(1).detach().cpu().numpy()
            occ_gt = batch["model_input"]["eval.points.occ"].unsqueeze(1).detach().cpu().numpy()
            iou = eval_iou(occ_gt, occ_pred, threshold=self.iou_threshold)  # B,T_all
            # make metric tensorboard
            batch["iou"] = iou.mean()
            batch["iou_i"] = torch.from_numpy(iou).reshape(-1)
            # make report
            report["iou"] = iou.mean(axis=1).tolist()
            batch["running_metric_report"] = report
        if "df_hat" in batch.keys():
            df_gt = batch["model_input"]["eval.points.value"].unsqueeze(1).detach().cpu().numpy()
            df_gt = abs(df_gt.squeeze(1))
            df_hat = abs(batch["df_hat"]).detach().cpu().numpy()
            df_error = abs(df_gt - df_hat)
            df_correct = df_error < self.df_acc_th
            nss_mask = (df_gt < self.nss_th).astype(np.float)
            far_mask = 1.0 - nss_mask
            df_acc = df_correct.sum(-1) / df_correct.shape[1]
            df_acc_nss = (df_correct * nss_mask).sum(-1) / (nss_mask.sum(-1) + 1e-6)
            df_acc_far = (df_correct * far_mask).sum(-1) / (far_mask.sum(-1) + 1e-6)
            batch["acc_i"], batch["acc"] = df_acc, df_acc.mean()
            batch["acc_nss_i"], batch["acc_nss"] = df_acc_nss, df_acc_nss.mean()
            batch["acc_far_i"], batch["acc_far"] = df_acc_far, df_acc_far.mean()

        if "z_so3" in batch.keys():
            self.network.eval()
            phase = batch["model_input"]["phase"]
            n_batch = batch["z_so3"].shape[0]
            # TEST_RESULT = {}
            with torch.no_grad():
                batch["mesh"] = []
                rendered_fig_list = []
                for bid in range(n_batch):
                    start_t = time.time()
                    embedding = {
                        "z_so3": batch["z_so3"][bid : bid + 1],
                        "z_inv": batch["z_inv"][bid : bid + 1],
                        "s": batch["s"][bid : bid + 1],
                        "t": batch["t"][bid : bid + 1],
                    }

                    if self.use_udf:  # generate dense pcl
                        from .utils.viz_udf_render import viz_input_and_recon

                        dense_pcl = self.generate_dense_surface_pts(embedding=embedding)
                        batch["dense_pcl"] = dense_pcl
                        rendered_fig = viz_input_and_recon(
                            input=batch["input"][bid].detach().cpu().numpy(), output=dense_pcl
                        )
                        # imageio.imsave("./debug/dbg.png", rendered_fig)
                        rendered_fig_list.append(rendered_fig.transpose(2, 0, 1)[None, ...])
                        # print()
                        # todo: render this pcl to viz, tensorboard is bad
                    else:  # generate mesh
                        mesh = self.generate_mesh(embedding=embedding)
                        batch["mesh"].append(mesh)
                    if self.viz_one and not phase.startswith("test"):
                        break
                if len(rendered_fig_list) > 0:
                    batch["rendered_fig_list"] = torch.Tensor(
                        np.concatenate(rendered_fig_list, axis=0)
                    )  # B,3,H,W
        return batch


class SIM3Recon(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)

        self.encoder_64_flag = cfg_with_default(cfg, ["model", "encoder_64"], True)
        # assert self.encoder_64_flag, "should use 64"

        self.decoder_type = cfg_with_default(cfg, ["model", "decoder_type"], "decoder")
        decoder_class = {"decoder": Decoder, "cbatchnorm": DecoderCBatchNorm, "inner": DecoderCat}[
            self.decoder_type
        ]

        self.encoder_type = cfg_with_default(cfg, ["model", "encoder_type"], "sim3pointres")
        encoder_class = {
            "vecdgcnn": VecDGCNN,
            "vecdgcnn2": VecDGCNN_v2,
            "vecdgcnn_atten": VecDGCNN_att,
        }

        if FLAGS.use_vae:
            encoder=VNN_ResnetPointnet(c_dim=256,hidden_dim=256)
        else:
            encoder = encoder_class[self.encoder_type](**cfg["model"]["encoder"])
        if self.encoder_64_flag:
            encoder = encoder.double()
        self.network_dict = torch.nn.ModuleDict()
        self.network_dict['encoder']=encoder
        if FLAGS.decoder_version=='v1':
            self.network_dict['decoder']=decoder_class(**cfg["model"]["decoder"])
        elif FLAGS.decoder_version=='v2':
            self.network_dict['decoder']=Decoder_v2()
        elif FLAGS.decoder_version=='v3':
            self.network_dict['decoder']=Decoder_v3()

        self.decoder_use_pe = cfg_with_default(cfg, ["model", "use_pe"], False)
        if self.decoder_use_pe:
            self.pe_src = cfg["model"]["pe_src"]
            self.pe_pow = cfg["model"]["pe_pow"]
            self.pe_sigma = np.pi * torch.pow(2, torch.linspace(0, self.pe_pow - 1, self.pe_pow))
            self.network_dict["pe_projector"] = VecLinear(
                cfg["model"]["encoder"]["c_dim"], self.pe_src
            )

        self.use_cls = cfg_with_default(cfg, ["model", "use_cls"], False)
        if self.use_cls:
            self.network_dict["cls_head"] = nn.Sequential(
                nn.Linear(cfg["model"]["encoder"]["c_dim"], cfg["model"]["encoder"]["c_dim"]),
                nn.Sigmoid(),
                nn.Linear(cfg["model"]["encoder"]["c_dim"], cfg["model"]["encoder"]["c_dim"]),
                nn.Sigmoid(),
                nn.Linear(cfg["model"]["encoder"]["c_dim"], cfg["model"]["num_cates"]),
            )
            self.w_cls = cfg_with_default(cfg["model"], ["w_cls"], 1.0)
            self.criterion_cls = nn.CrossEntropyLoss()

        self.w_s = cfg_with_default(cfg["model"], ["w_s"], 0.0)
        self.w_t = cfg_with_default(cfg["model"], ["w_t"], 0.0)
        self.w_recon = cfg_with_default(cfg["model"], ["w_recon"], 1.0)

        self.sdf2occ_factor = cfg_with_default(cfg, ["model", "sdf2occ_factor"], -1.0)
        self.w_uni = cfg_with_default(cfg, ["model", "w_uni"], 1.0)
        self.w_nss = cfg_with_default(cfg, ["model", "w_nss"], 1.0)

        self.loss_th = cfg_with_default(cfg, ["model", "loss_th"], 1.0)
        self.loss_near_lambda = cfg_with_default(cfg, ["model", "loss_near_lambda"], 1.0)
        self.loss_far_lambda = cfg_with_default(cfg, ["model", "loss_far_lambda"], 0.1)

        self.training_centroid_aug_std = cfg["model"]["center_aug_std"]

        self.use_udf = cfg_with_default(cfg, ["model", "use_udf"], False)
        if self.use_udf:
            logging.info("Use UDF instead of SDF")

        self.rot_aug = cfg_with_default(cfg, ["model", "rot_aug"], False)
        if self.rot_aug:
            logging.warning(f"Use Rot Aug, this should only happen for ablation study!")

        count_param(self.network_dict)

        return

    def forward(self, input_pack, viz_flag=False):
        output = {}
        phase= input_pack["phase"]

        # prepare inputs
        input_pcl = input_pack["inputs"].transpose(2, 1)
        query = torch.cat([input_pack["points.uni"], input_pack["points.nss"]], dim=1)
        B, _, N = input_pcl.shape
        device = input_pcl.device

        # encoding
        if self.encoder_64_flag:
            input_pcl = input_pcl.double()
        encoder_ret = self.network_dict["encoder"](input_pcl)
        if FLAGS.use_vae==1:
            pred_so3_feat, pred_inv_feat,kl_loss = encoder_ret
            output['kl_loss']=kl_loss
        else:
            pred_so3_feat, pred_inv_feat = encoder_ret
        if self.encoder_64_flag:

            pred_so3_feat, pred_inv_feat = pred_so3_feat.float(), pred_inv_feat.float()



        embedding = {
            "z_so3": pred_so3_feat,
            "z_inv": pred_inv_feat,
        }

        if phase.startswith("test") or viz_flag:
            output["z_so3"] = pred_so3_feat
            output["z_inv"] = pred_inv_feat
            output["input"] = input_pack["inputs"]
        if phase.startswith("test"):
            return output

        N_uni = input_pack["points.uni"].shape[1]
        sdf_hat = self.decode(  # SDF must have nss sampling
            query,
            None,
            embedding,
            return_sdf=True,
        )
        sdf_gt = torch.cat([input_pack["points.uni.value"], input_pack["points.nss.value"]], dim=1)
        if self.use_udf:
            sdf_gt, sdf_hat = abs(sdf_gt), abs(sdf_hat)

        sdf_error_i = abs(sdf_hat - sdf_gt)
        sdf_near_mask = (sdf_error_i < self.loss_th).float().detach()
        sdf_loss_i = (
            sdf_error_i * sdf_near_mask * self.loss_near_lambda
            + sdf_error_i * (1.0 - sdf_near_mask) * self.loss_far_lambda
        )


        output["batch_loss"] = (
            sdf_loss_i.mean()
        )

        return output

    def positional_encoder(self, x):
        device = x.device
        y = torch.cat(
            [
                x[..., None],
                torch.sin(x[:, :, :, None] * self.pe_sigma[None, None, None].to(device)),
                torch.cos(x[:, :, :, None] * self.pe_sigma[None, None, None].to(device)),
            ],
            dim=-1,
        )
        return y

    def decode(self, query, z_none, c, return_sdf=False):
        B, M, _ = query.shape
        z_so3, z_inv = c["z_so3"], c["z_inv"]
        if FLAGS.decoder_version!='v1':
            sdf = self.network_dict["decoder"](z_so3,z_inv,query)
        else:
            q = query

            inner = (q.unsqueeze(1) * z_so3.unsqueeze(2)).sum(dim=-1)  # B,C,N
            length = q.norm(dim=-1).unsqueeze(1)
            inv_query = torch.cat([inner, length], 1).transpose(2, 1)  # B,N,D

            if self.decoder_use_pe:
                coordinate = self.network_dict["pe_projector"](z_so3)  # B,PE_C,3
                pe_inner = (q.unsqueeze(1) * coordinate.unsqueeze(2)).sum(dim=-1)  # B,PE_C,N
                pe_query = self.positional_encoder(pe_inner)
                pe_query = pe_query.transpose(-2, -1).reshape(B, -1, M)
                inv_query = torch.cat([inv_query, pe_query.transpose(2, 1)], 2)

            if self.decoder_type == "inner":
                input = torch.cat([inv_query, z_inv[:, None, :].expand(-1, M, -1)], -1)
                sdf = self.network_dict["decoder"](input)
            else:
                sdf = self.network_dict["decoder"](inv_query, None, z_inv)

        if return_sdf:
            return sdf
        else:
            return dist.Bernoulli(logits=self.sdf2occ_factor * sdf)
