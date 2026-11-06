import torch.nn as nn
import torch
import time
import os
import torch.nn.functional as F
import torch.utils.checkpoint as gradient_checkpoint
from easydict import EasyDict
from externals.pointconvformer.model_architecture import get_default_configs
from externals.pointconvformer.layers import (
    PointConvStridePE_custom,
    PointConvTransposePE_custom,
)


def build_pointconv_interaction_nets(cfg, phase):

    if cfg['model']['type'] == 'Interaction_PointConv':
        model = Interaction_PointConv(
            cfg['pointcloud']['input_dim'],
            cfg['pointcloud']['knn'],
            cfg['pointcloud']['knn_k_decay_factor'],
            cfg['pointcloud']['scaling'],
            cfg['pointcloud']['downsampling_layer_num'],
            cfg['model']['input_frame_num'],
            cfg['model']['pointcloud_feat'],
            cfg['model']['dist_threshold'],
            cfg['model']['pointwise_prediction'],
            phase
        )
    else:
        raise NotImplementedError
    
    if phase == 'test':        
        model_path = os.path.join(cfg['exp_dir'], cfg['exp_name'], 'epoch_%02d/model.pt' % cfg['epoch_sel'])
        chkpt = torch.load(model_path)
        model.load_state_dict(chkpt['pointconv_internaction_network'])
        model.eval()

    return model.to(cfg['device'])


def build_multi_frame_mlp(cfg, phase):

    pred_dim = cfg['model']['output_dim']
    model = Multi_Frame_MLP(
        cfg['model']['pred_mlp_input_feat_dim'],
        1, # single frame mlp (right now, multiple frames are taken care by the backbone)
        cfg['model']['pred_mlp_hidden_dim'],
        pred_dim
    )
    if phase == 'test':
        model_path = os.path.join(cfg['exp_dir'], cfg['exp_name'], 'epoch_%02d/model.pt' % cfg['epoch_sel'])
        chkpt = torch.load(model_path)
        model.load_state_dict(chkpt['predictor'])
        model.eval()

    return model.to(cfg['device'])


class Interaction_PointConv(nn.Module):

    def __init__(
        self,
        point_dim,
        knn,
        knn_k_decay_factor,
        scaling,
        block_num,
        input_history_len,
        feature_dim,
        dist_threshold,
        pointwise_prediction,
        phase,
        relu_slope=0.2
    ):
        super(Interaction_PointConv, self).__init__()
        self.point_dim = point_dim
        self.knn = knn
        self.knn_k_decay_factor = knn_k_decay_factor
        self.block_num = block_num
        self.phase = phase
        # group_num = 1
        self.conv1 = nn.Linear(input_history_len*(point_dim), feature_dim)
        # self.gn1 = nn.GroupNorm(group_num, feature_dim)
        self.bn1 = nn.BatchNorm1d(feature_dim, momentum=0.1)
        self.conv2 = nn.Linear(feature_dim, feature_dim)
        # self.gn2 = nn.GroupNorm(group_num, feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim, momentum=0.1)
        self.conv3 = nn.Linear(feature_dim, feature_dim)
        # self.gn3 = nn.GroupNorm(group_num, feature_dim)
        self.bn3 = nn.BatchNorm1d(feature_dim, momentum=0.1)
        self.relu_slope = relu_slope
        self.dist_threshold = dist_threshold * scaling
        self.pointwise_prediction = pointwise_prediction
        self.point_modeling = nn.ModuleList()
        unet_feature_dim = feature_dim

        # First assignment-free spatial block without downsampling.
        self.point_modeling.append(
            Spatial_PointConv(unet_feature_dim, unet_feature_dim, point_dim)
        )
        unet_feature_dim = 2*unet_feature_dim

        # Downsampling spatial blocks.
        for _ in range(self.block_num):
            self.point_modeling.append(
                Spatial_PointConv(
                    int(unet_feature_dim // 2),
                    unet_feature_dim,
                    point_dim,
                )
            )
            unet_feature_dim = 2*unet_feature_dim
        
        # Bottleneck spatial blocks.
        unet_feature_dim = int(unet_feature_dim // 2)
        for _ in range(self.block_num):
            self.point_modeling.append(
                Spatial_PointConv(unet_feature_dim, unet_feature_dim, point_dim)
            )

        # Upsampling spatial blocks.
        for _ in range(self.block_num):
            self.point_modeling.append(
                Spatial_PointConv_Transpose(
                    unet_feature_dim,
                    int(unet_feature_dim // 2),
                    point_dim,
                )
            )
            unet_feature_dim = int(unet_feature_dim // 2)
        
        # Final assignment-free spatial block without upsampling.
        self.point_modeling.append(
            Spatial_PointConv(unet_feature_dim, unet_feature_dim, point_dim)
        )

    def forward(self, xyz_delta, xyz_sets, idx_info):

        assert xyz_delta.shape[1] == self.conv1.in_features, (
            "Concatenated motion history does not match input_frame_num * "
            "pointcloud.input_dim."
        )
        assert len(xyz_sets) == self.block_num + 1
        assert all(points.shape[1] == self.point_dim for points in xyz_sets)

        xyz_delta = torch.unsqueeze(xyz_delta, dim=0).contiguous() # [1, B, C]
        xyz_sets_conv = []
        nn_idx_conv_forward = []
        nn_idx_conv_propagate = []
        for l in range(len(xyz_sets)):
            xyz_sets_conv.append(torch.unsqueeze(xyz_sets[l], dim=0).contiguous())
            if l != len(xyz_sets) - 1:
                nn_idx_conv_forward.append(
                    torch.unsqueeze(idx_info['nn_idx_forward'][l], dim=0)
                )
                nn_idx_conv_propagate.append(
                    torch.unsqueeze(idx_info['nn_idx_propagate'][l], dim=0)
                )

        # Recompute same-level spatial neighborhoods from the current positions.
        # This keeps the graph valid as independently moving points approach or
        # separate from one another during a rollout.
        nn_idx_spatial_self = []
        curr_k = self.knn
        if self.phase == 'test':
            assert len(idx_info['batch_idx'][0]) == 1, "Testing with batch size > 1 is not supported."
            for l in range(len(xyz_sets_conv)):
                curr_k = curr_k if l == 0 else int(curr_k // self.knn_k_decay_factor)
                assert curr_k >= 1
                nn_idx_spatial_self.append(
                    self._knn_point(curr_k, xyz_sets_conv[l], xyz_sets_conv[l])
                )
        else:
            for i in range(len(xyz_sets_conv)):
                curr_k = curr_k if i == 0 else int(curr_k // self.knn_k_decay_factor)
                assert curr_k >= 1
                nn_idx_spatial_self_list = []
                level_l = torch.split(
                    xyz_sets_conv[i], idx_info['batch_idx'][i], dim=1
                )
                offset = 0
                for level_l_pcd in level_l:
                    nn_idx_spatial_self_list.append(
                        self._knn_point(curr_k, level_l_pcd, level_l_pcd) + offset
                    )
                    offset += level_l_pcd.shape[1]
                nn_idx_spatial_self.append(
                    torch.cat(nn_idx_spatial_self_list, dim=1)
                )

        # point wise encoding
        features = self.conv1(xyz_delta)         
        features = F.leaky_relu(
            # self.gn1(features.permute(1, 2, 0)).permute(2, 0, 1),            
            self.bn1(features.permute(0, 2, 1)).permute(0, 2, 1),
            negative_slope=self.relu_slope
        )
        features = self.conv2(features)
        features = F.leaky_relu(
            # self.gn2(features.permute(1, 2, 0)).permute(2, 0, 1),
            self.bn2(features.permute(0, 2, 1)).permute(0, 2, 1),
            negative_slope=self.relu_slope
        )

        # First assignment-free spatial block without downsampling.
        features_all = []
        ct = 0
        layer = 0
        features = gradient_checkpoint.checkpoint(
            self.point_modeling[layer],
            self.dist_threshold,
            xyz_sets_conv[ct],
            features,
            nn_idx_spatial_self[ct],
            use_reentrant = False
        )
        features_all.append(features)

        # Downsampling spatial blocks.
        layer += 1        
        for _ in range(self.block_num):
            features = gradient_checkpoint.checkpoint(
                self.point_modeling[layer],
                self.dist_threshold,
                xyz_sets_conv[ct],
                features,
                nn_idx_conv_forward[ct],
                xyz_sets_conv[ct + 1],
                use_reentrant = False
            )
            features_all.append(features)
            layer += 1
            ct += 1

        # Bottleneck spatial blocks.
        ct = len(xyz_sets_conv) - 1
        for _ in range(self.block_num):
            features = gradient_checkpoint.checkpoint(
                self.point_modeling[layer],
                self.dist_threshold,
                xyz_sets_conv[ct],
                features,
                nn_idx_spatial_self[ct],
                use_reentrant = False
            )
            layer += 1
        
        # Upsampling spatial blocks.
        ct = len(xyz_sets_conv) - 1
        for _ in range(self.block_num):
            features = gradient_checkpoint.checkpoint(
                self.point_modeling[layer],
                self.dist_threshold,
                xyz_sets_conv[ct],
                features,
                nn_idx_conv_propagate[ct - 1],
                xyz_sets_conv[ct - 1],
                features_all[ct - 1],
                use_reentrant = False
            )
            layer += 1
            ct -= 1
        
        # Final assignment-free spatial block.
        ct = 0
        features = gradient_checkpoint.checkpoint(
            self.point_modeling[layer],
            self.dist_threshold,
            xyz_sets_conv[ct],
            features,
            nn_idx_spatial_self[ct],
            use_reentrant = False
        )
        features = torch.squeeze(features, dim=0)

        return features

    def _knn_point(self, nsample, xyz, new_xyz):

        sqrdists = torch.cdist(new_xyz, xyz)
        _, group_idx = torch.topk(
            sqrdists, nsample, dim = -1, largest=False, sorted=False
        )

        return group_idx


class Spatial_PointConv(nn.Module):

    def __init__(self, channel_in, channel_out, point_dim, is_res=True):
        
        super(Spatial_PointConv, self).__init__()
        # Pass the coordinate width explicitly so this block can later operate
        # on native 2D points instead of silently reverting to XYZ.
        cfg = get_default_configs(EasyDict({'point_dim': point_dim}))
        cfg.USE_VI = False
        cfg.USE_PE = True
        cfg.num_heads = 1
        cfg.mid_dim = 4

        weightnet_input_dim = cfg.point_dim
        weightnet = [weightnet_input_dim, cfg.mid_dim]
        self.pointconv_res_block = PointConvStridePE_custom(
            channel_in, channel_out, cfg, weightnet, is_res
        )

    def forward(
            self,
            dist,
            dense_xyz,
            dense_feats,
            nn_idx,
            sparse_xyz=None,
        ):
        
        features, _ = self.pointconv_res_block(
            dist, dense_xyz, dense_feats, nn_idx, sparse_xyz
        )

        return features


class Spatial_PointConv_Transpose(nn.Module):

    def __init__(self, channel_in, channel_out, point_dim):
        
        super(Spatial_PointConv_Transpose, self).__init__()
        cfg = get_default_configs(EasyDict({'point_dim': point_dim}))
        cfg.USE_VI = False
        cfg.num_heads = 1
        cfg.mid_dim = 4
        weightnet_input_dim = cfg.point_dim
        weightnet = [weightnet_input_dim, cfg.mid_dim]
        mlp2 = [channel_out, channel_out]

        self.pointconv_res_block = PointConvTransposePE_custom(
            channel_in, channel_out, cfg, weightnet, mlp2
        )

    def forward(
            self,
            dist,
            sparse_xyz,
            sparse_feats,
            nn_idx,
            dense_xyz,
            dense_feats,
        ):
        
        features, _ = self.pointconv_res_block(
            dist, sparse_xyz, sparse_feats, nn_idx, dense_xyz, dense_feats
        )

        return features


class MLP(nn.Module):

    def __init__(self, in_feat_dim, hidden_size, out_feat_dim):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_feat_dim, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_feat_dim),
        )

    def forward(self, x):
    
        return self.layers(x)


class Multi_Frame_MLP(nn.Module):

    def __init__(self, features_dim, frame_num, hidden_size, states_dim):

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features_dim*frame_num, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, states_dim),
        )

    def forward(self, x):
    
        return self.layers(x)
