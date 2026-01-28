from functools import partial
from easydict import EasyDict

import copy
import torch
from torch import nn
import torch.nn.functional as F
from eqnet.models import support_processor
from eqnet.utils.config import cfg, cfg_from_yaml_file, merge_new_config
from eqnet.utils import attention_helper
from eqnet.positional_encoding import crpe, rpe
from eqnet import transformer
from eqnet.transformer.utils import *
import absl.flags as flags
import os
from e3nn import o3
FLAGS = flags.FLAGS

class QNet_equi_v5(nn.Module):
    def __init__(self, cfg_file, qnet_handcrafted_cfg=None):
        super().__init__()
        cfg_from_yaml_file(cfg_file, cfg)
        qnet_handcrafted_cfg = EasyDict() if qnet_handcrafted_cfg is None else qnet_handcrafted_cfg
        model_cfg = merge_new_config(cfg, qnet_handcrafted_cfg)
        self.model_cfg = model_cfg

        # 1. Support feature preprocessing module.
        self.support_feature_processor_cfg = self.model_cfg.get('SUPPORT_FEATURE_PROCESSOR')
        self.support_feature_processor = support_processor.__all__ [self.support_feature_processor_cfg.NAME](
            self.support_feature_processor_cfg)

        # 2. Define Q-Net.
        # hierarchical feature num.
        self.num_levels = len(self.support_feature_processor.target_chn)

        # q-encoder & q-decoder num in each level.
        self.num_q_layers = self.model_cfg.get('NUM_Q_LAYERS')


        # head_num, dropout_rate, dim_feedforward and neighbor num of local-wise attention in each level.
        self.q_head_per_level = self._get_per_level_param('Q_HEAD_PER_LEVEL')
        self.q_dim_feedforward_per_level = self._get_per_level_param('Q_DIM_FEEDFORWARD_PER_LEVEL')
        self.q_dropout_per_level = self._get_per_level_param('Q_DROPOUT_PER_LEVEL')
        self.q_local_size_per_level = self._get_per_level_param('Q_LOCAL_SIZE_PER_LEVEL')
        # attention & rpe version.
        self.q_version = self.model_cfg.get('Q_VERSION', 'v2')
        # rpe setting.
        self.rpe_type = self.model_cfg.get('RPE_TYPE', 'CRPE')
        self.rpe_cfg = self.model_cfg.get(f'{self.rpe_type}_CONFIG')
        self.rpe_point_cloud_range = self.rpe_cfg.get('POINT_CLOUD_RANGE')
        self.rpe_quan_size = self.rpe_cfg.get('QUANTIZE_SIZE')


        # target output channel for query features.
        self.q_target_chn = self.model_cfg.get('Q_TARGET_CHANNEL')

        # define rpe_layer
        if self.rpe_type == 'RPE':
            if FLAGS.use_dir_rpe:
                self.dir_rpe_layer=rpe.DirRPE(self.q_head_per_level[0])
            self.rpe_layer = rpe.RPE(
                self.q_head_per_level[0],
                point_cloud_range=self.rpe_point_cloud_range,
                quan_size=self.rpe_quan_size
            )

        # define decoder and encoder function,
        self.q_target_chn_per_level = self.support_feature_processor.target_chn
        # q-encoder and related modules.
        self.q_encoder = nn.ModuleList()
        # q-decoder and related modules.
        self.q_decoder = nn.ModuleList()
        for i in range(self.num_levels):

            decoder_layers = nn.ModuleList([
                transformer.TransformerDecoderLayer_equi_v4(
                    self.q_version, self.q_target_chn_per_level[i], self.q_head_per_level[i],
                    self.q_dim_feedforward_per_level[i], self.q_dropout_per_level[i])
                for layer_idx in range(self.num_q_layers)
            ])
            self.q_decoder.append(decoder_layers)

            encoder_layers = nn.ModuleList([
                transformer.TransformerEncoderLayer_equi_v4(
                    self.q_version, self.q_target_chn_per_level[i], self.q_head_per_level[i],
                    self.q_dim_feedforward_per_level[i], self.q_dropout_per_level[i])
                for layer_idx in range(self.num_q_layers - 1)
            ])
            self.q_encoder.append(encoder_layers)



        num_features_before_fusion = sum(self.q_target_chn_per_level)
        merging_mlp = self.model_cfg.get('MERGING_MLP', [])
        merging_mlp = [num_features_before_fusion] + merging_mlp + [self.q_target_chn]
        merging_mlp_layers = []
        for k in range(len(merging_mlp) - 1):
            merging_mlp_layers.extend([
                VNLinear(merging_mlp[k],merging_mlp[k + 1]),
                VNLayerNorm2(merging_mlp[k + 1]),
                VNReLU(merging_mlp[k + 1])
            ])
        self.merging_mlp = nn.Sequential(*merging_mlp_layers)

        self.irreps_sh=o3.Irreps('1x0e+1x1e')
        self.sh = o3.SphericalHarmonics(irreps_out = self.irreps_sh, normalize = True, normalization='component')
        self.length_range=(0.001,0.005,None,None)



    def _get_per_level_param(self, key):
        param = self.model_cfg.get(key)
        if not isinstance(param, list):
            param = [param] * self.num_levels
        assert len(param) == self.num_levels
        return param

    def _compute_relative_pos(self, pos1, pos2):
        # pos1: A float tensor with shape [n, 3]
        # pos2: A float tensor with shape [n, m, 3]
        relative_pos = pos2[:, :, :] - pos1[:, None, :]  # n, m, 3
        return relative_pos




    def qnet(self, data_dict, query_features, query_pos,
             support_features, support_pos, support_mask, level_idx,pred_scale=None):
        """
        :param data_dict:
        :param query_features: A float tensor with shape [bs, query_num, c]
        :param query_pos: A float tensor with shape [bs, query_num, 3]

        :param support_features: A float tensor with shape [bs, key_num, c]
        :param support_pos: A float tensor with shape [bs, key_num, 3]
        :param support_mask: A bool tensor with shape [bs, key_num], 0 valid / 1 padding.
        :param level_idx
        :return:
        """
        if FLAGS.use_pred_scale:
            assert pred_scale is not None
        batch_size = data_dict['batch_size']

        # generate index pair for cross-attention layer.
        # cross-attention: query & support.
        (query_pos, query_features, query_batch_cnt, support_pos, support_features, support_batch_cnt,
         ca_index_pair, ca_index_pair_batch, support_key_pos) = attention_helper.ca_attention_mapper(
            query_pos, query_features, support_pos, support_features, support_mask,
            self.q_local_size_per_level[level_idx])

        # print(ca_index_pair)

        # self-attention: support & support.
        sa_index_pair, sa_index_pair_batch, sa_key_pos = attention_helper.sa_attention_mapper(
            support_pos,support_batch_cnt, self.q_local_size_per_level[level_idx])

        support_cnt=support_batch_cnt[0]
        feat_dim=support_features.shape[1]
        query_cnt=query_batch_cnt[0]

        if FLAGS.use_global==1:

            support_pos_=support_pos.reshape(batch_size,support_cnt,-1)
            support_features_=support_features.reshape(batch_size,support_cnt,-1)
            mean_support_pos=torch.mean(support_pos_,dim=1,keepdim=True)
            mean_support_features=torch.mean(support_features_,dim=1,keepdim=True)
            neighbors=self.q_local_size_per_level[level_idx]
            support_pos_=torch.cat([support_pos_,mean_support_pos],dim=1)
            support_features_=torch.cat([support_features_,mean_support_features],dim=1)
            support_key_pos_=support_key_pos.reshape(batch_size,query_cnt,neighbors,3)
            support_key_pos_=torch.cat([support_key_pos_,mean_support_pos.unsqueeze(1).repeat(1,query_cnt,1,1)],dim=2)
            ca_index_pair_=ca_index_pair.reshape(batch_size,-1,neighbors)
            pad=(0,1)
            ca_index_pair_=F.pad(ca_index_pair_,pad,'constant',support_cnt)


            sa_index_pair_=sa_index_pair.reshape(batch_size,-1,neighbors)
            sa_key_pos_=sa_key_pos.reshape(batch_size,support_cnt,neighbors,3)
            sa_key_pos_=torch.cat([sa_key_pos_,mean_support_pos.unsqueeze(1).repeat(1,support_cnt,1,1)],dim=2)
            sa_key_pos_=torch.cat([sa_key_pos_,mean_support_pos.unsqueeze(1).repeat(1,1,neighbors+1,1)],dim=1)
            pad=(0,1,0,1)
            sa_index_pair_=F.pad(sa_index_pair_,pad,'constant',support_cnt)

            support_pos=support_pos_.reshape(-1,3)
            support_features=support_features_.reshape(-1,feat_dim)
            ca_index_pair=ca_index_pair_.reshape(-1,neighbors+1)
            sa_index_pair=sa_index_pair_.reshape(-1,neighbors+1)
            support_key_pos=support_key_pos_.reshape(-1,neighbors+1,3)
            sa_key_pos=sa_key_pos_.reshape(-1,neighbors+1,3)
            support_batch_cnt=support_batch_cnt+1
            sa_index_pair_batch=torch.arange(batch_size,device=sa_index_pair_batch.device,dtype=torch.int32).unsqueeze(-1).repeat(1,support_cnt+1).reshape(-1)
            support_cnt+=1





        ca_relpos = self._compute_relative_pos(query_pos, support_key_pos)
        ca_length=ca_relpos.norm(dim=-1, p=2)
        pred_scale_q=pred_scale.reshape(-1,1,1).repeat(1,query_cnt,1)
        ca_rpe_weights = self.rpe_layer(ca_relpos,pred_scale_q)

        ca_cutoff_nonscalar = soft_square_cutoff_2(x=ca_length.reshape(-1), ranges=self.length_range)
        ca_edge_sh=self.sh(ca_relpos.reshape(-1,3))
        ca_edge_sh = cutoff_irreps(f=ca_edge_sh,
                                edge_cutoff=None,
                                cutoff_scalar=None,
                                cutoff_nonscalar=ca_cutoff_nonscalar,
                                irreps=self.irreps_sh)




        sa_relpos = self._compute_relative_pos(support_pos, sa_key_pos)
        sa_length=sa_relpos.norm(dim=-1, p=2)

        pred_scale_s=pred_scale.reshape(-1,1,1).repeat(1,support_cnt,1)
        sa_rpe_weights = self.rpe_layer(sa_relpos,pred_scale_s)





        sa_cutoff_nonscalar = soft_square_cutoff_2(x=sa_length.reshape(-1), ranges=self.length_range)
        sa_edge_sh=self.sh(sa_relpos.reshape(-1,3))

        sa_edge_sh = cutoff_irreps(f=sa_edge_sh,
                                   edge_cutoff=None,
                                   cutoff_scalar=None,
                                   cutoff_nonscalar=sa_cutoff_nonscalar,
                                   irreps=self.irreps_sh)







        # 3. Do transformer.
        aux_ret = []
        num_q_layers = self.num_q_layers
        for i in range(num_q_layers):
            # do q-decoder layer
            # support_key_features=attention_helper.ca_attention_mapper_v3(ca_index_pair,support_features,query_cnt,support_cnt)
            # sa_key_features=attention_helper.sa_attention_mapper_v3(sa_index_pair,support_features,support_cnt)

            query_features = self.q_decoder[level_idx][i](
                query_features, support_features,ca_index_pair,query_cnt,support_cnt,ca_edge_sh,ca_rpe_weights)


            # do q-encoder layer: self-attention on support features.
            # Only do these self-attention on the not last layer.
            if i != (num_q_layers - 1):
                support_features = self.q_encoder[level_idx][i](
                    support_features,sa_index_pair,support_cnt,sa_edge_sh,sa_rpe_weights)
                # print(1)



        # 4. Revert window representation back.
        return query_features.view(batch_size, -1, self.q_target_chn_per_level[level_idx],3)

    def forward(self, data_dict,pred_scale=None):
        batch_size = data_dict['batch_size']

        # 1. preprocess support features & get support features / points / mask.
        support_dict = self.support_feature_processor(data_dict)
        support_features = support_dict['support_features']
        support_points = support_dict['support_points']
        support_mask = support_dict['support_mask']



        query_positions = data_dict['query_positions']  # batch_size, query_num, 3
        # print(query_positions)
        # 2. Hierarchical inference.
        query_feature_list = []
        aux_feature_list = []
        for i in range(self.num_levels):
            # 3. generate target path.
            cur_support_features = support_features[i]
            cur_support_points = support_points[i]
            cur_support_mask = support_mask[i]

            query_features = cur_support_features.new_zeros(
                (batch_size, query_positions.shape[1], self.q_target_chn_per_level[i]*3))

            # query_features: b, num_query, c
            # aux_query_features: a list of [b, num_query, aux_c]

            query_features= self.qnet(
                data_dict, query_features, query_positions,
                cur_support_features, cur_support_points, cur_support_mask, i,pred_scale)
            # print(query_features[0][0])

            query_feature_list.append(query_features)

        # --- c. merge different features.
        # Aggregate result from different path.
        query_features = torch.cat(query_feature_list, dim=-2)  # b, n, c1 + c2 + ...
        data_dict['query_features_before_fusion'] = query_features

        relu_list=[]
        for layer in self.merging_mlp:
            query_features=layer(query_features)
            if layer._get_name()=='VNReLU':
                relu_list.append(query_features)
        relu_fea=torch.cat(relu_list,dim=2)


        # query_features = self.merging_mlp(query_features)


        data_dict['query_features'] = query_features
        data_dict['relu_fea'] = relu_fea
        return data_dict