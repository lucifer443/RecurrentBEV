from typing import List, Sequence
from torch import Tensor

import kornia
from mmengine.model import BaseModule
from mmengine.runner import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F

from bev.registry import MODELS
from bev.utils.typing_utils import OptConfigType
from bev.utils.misc import inverse_trans_matrix, cast_data
from bev.models.middle_encoders.depth_encoders import CamEncoder
from ..utils import multi_apply, gen_dx_bx


@MODELS.register_module()
class TransformAwareGRU(BaseModule):

    def __init__(self,
                 grid_config: dict,
                 transform_aware_config: dict,
                 encoder_config: dict,
                 used_prev_frames: int = 1,
                 prev_bev_channels: int = 256,
                 empty_padding: str = 'zero',
                 init_cfg: OptConfigType = None,):
        super().__init__(init_cfg=init_cfg)
        self.grid_config = grid_config
        self.used_prev_frames = used_prev_frames

        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.register_buffer('dx', dx)
        self.register_buffer('bx', bx)
        self.register_buffer('nx', nx)

        self.prev_bev_channels = prev_bev_channels

        padding_func = dict(
            zero=self.zero_padding,
            self=self.self_padding
        )
        self.ta_block = TransformAwareBlock(**transform_aware_config)
        self.encoder = ConvGRU(**encoder_config)
        self.padding_func = padding_func[empty_padding]

    @autocast(enabled=False, cache_enabled=False)
    def spacial_align(self,
                      transforms: dict,
                      prev_bevs_list: List[Sequence[Tensor]],
                      prev_transforms_list: List[dict]) -> List[Sequence[Tensor]]:
        """Spacial align bev features.

        Reserved bevs list length is used_prev_frames
        """
        prev_bevs_list = cast_data(prev_bevs_list, torch.float32)
        aligned_prev_bevs_list = []
        for t in range(self.used_prev_frames):
            bev_feats = prev_bevs_list[t]
            grid = self.generate_grid(transforms, prev_transforms_list[t])

            # multi_apply for multi-level features
            aligned_bev_feats = multi_apply(
                F.grid_sample,
                bev_feats,
                grid=grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True)

            transform_embedding = _compute_transform_embedding(transforms, prev_transforms_list[t])
            aligned_bev_feats = multi_apply(
                self.ta_block,
                aligned_bev_feats,
                transform_embedding=transform_embedding,
            )

            aligned_prev_bevs_list.append(aligned_bev_feats)
        return aligned_prev_bevs_list

    def forward(self,
                bev_feats: Sequence[Tensor],
                transforms: dict,
                prev_bevs_list: List[Sequence[Tensor]],
                prev_transforms_list: List[dict]) -> Sequence[Tensor]:
        assert len(prev_bevs_list) == len(prev_transforms_list)
        # Pad fake frames with zero features and random transforms
        # Avoid modifyind origin prev_bevs_list and prev_transforms_list
        prev_bevs_list = prev_bevs_list.copy()
        prev_transforms_list = prev_transforms_list.copy()
        while len(prev_bevs_list) < self.used_prev_frames:
            prev_bevs_list.insert(0, multi_apply(self.padding_func, bev_feats))
            prev_transforms_list.insert(0, transforms)

        prev_bevs_list = prev_bevs_list[-self.used_prev_frames:]
        prev_transforms_list = prev_transforms_list[-self.used_prev_frames:]

        # if self.conv_on_prev_bev:
        #     # Use list comprehension to avoid modifying origin prev_bevs_list
        #     # multi_apply for multi-level features
        #     prev_bevs_list = [multi_apply(self.conv, bev_feats)
        #                       for bev_feats in prev_bevs_list]

        prev_bevs_list = self.spacial_align(transforms,
                                            prev_bevs_list,
                                            prev_transforms_list)
        all_bevs = prev_bevs_list + [bev_feats]

        # convert [(feat_a0, feat_a1, ...), (feat_b0, feat_b1, ...), ...]
        # to [(feat_a0, feat_b0, ...), (feat_a1, feat_b1, ...), ...]
        all_bevs = list(zip(*all_bevs))
        outs = multi_apply(torch.cat, all_bevs, dim=1)

        outs = multi_apply(self.encoder, outs)
        return outs

    def generate_grid(self, transforms: dict, prev_transforms: dict) -> Tensor:
        B = transforms['lidar2global'].shape[0]
        Y = self.nx[1].int().item()
        X = self.nx[0].int().item()

        curr_l2g = transforms['lidar2global']
        curr_l2la = transforms['lidar2lidar_aug']
        adj_l2g = prev_transforms['lidar2global']
        adj_l2la = prev_transforms['lidar2lidar_aug']

        # Prepare transformation curr_lidar_aug_2_adj_lidar_aug
        curr2adj = (
                adj_l2la @
                inverse_trans_matrix(adj_l2g) @
                curr_l2g @
                inverse_trans_matrix(curr_l2la))

        # Get bev coors of (x, y, 0, 1)
        pixel_coors = kornia.utils.create_meshgrid(
            height=Y,  # Y
            width=X,  # X
            normalized_coordinates=False,
            device=curr_l2g.device)  # (1, Y, X, 2)
        curr_bev_coors = self.bx[:2] + pixel_coors * self.dx[:2]
        curr_bev_coors = F.pad(curr_bev_coors, [0, 1], "constant", 0.0)
        curr_bev_coors = F.pad(curr_bev_coors, [0, 1], "constant", 1.0)

        # Transform coors
        curr_bev_coors = curr_bev_coors.reshape(1, Y, X, 4, 1)
        curr2adj = curr2adj.reshape(B, 1, 1, 4, 4)
        adj_bev_coors = curr2adj @ curr_bev_coors
        adj_bev_coors = adj_bev_coors[:, :, :, :2, 0]

        # Normalize between [-1, 1]
        grid = (adj_bev_coors - self.bx[:2]) \
               / ((self.nx[:2] - 1) * self.dx[:2])
        min_n = -1.0
        max_n = 1.0
        grid = grid * (max_n - min_n) + min_n

        return grid

    def self_padding(self, feat):
        repeat_num = self.prev_bev_channels // feat.size(1)
        return feat.repeat(1, repeat_num, 1, 1).detach()

    def zero_padding(self, feat):
        B, _, H, W = feat.shape
        C = self.prev_bev_channels
        zero_feat = torch.zeros(
            (B, C, H, W),
            dtype=feat.dtype,
            layout=feat.layout,
            device=feat.device)
        return zero_feat


class TransformAwareBlock(BaseModule):

    def __init__(self,
                 transform_channels,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 init_cfg=None):
        super(TransformAwareBlock, self).__init__(init_cfg)

        self.cam_encoder = CamEncoder(transform_channels, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, x, transform_embedding):
        scale = self.cam_encoder(transform_embedding)
        x = x * scale[..., None, None]
        x = self.conv(x)
        return x


class ConvGRU(BaseModule):

    def __init__(self, curr_channels, hidden_channels, kernel_size):
        super(ConvGRU, self).__init__()
        padding = kernel_size // 2
        self.curr_channels = curr_channels
        self.hidden_channels = hidden_channels
        self.reset_gate = nn.Conv2d(curr_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(curr_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(curr_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def init_weights(self):
        nn.init.orthogonal_(self.reset_gate.weight)
        nn.init.orthogonal_(self.update_gate.weight)
        nn.init.orthogonal_(self.out_gate.weight)
        nn.init.constant_(self.reset_gate.bias, 0.)
        nn.init.constant_(self.update_gate.bias, 0.)
        nn.init.constant_(self.out_gate.bias, 0.)

    def forward(self, feat):
        prev, curr = feat.split([self.hidden_channels, self.curr_channels], dim=1)

        update = torch.sigmoid(self.update_gate(feat))
        reset = torch.sigmoid(self.reset_gate(feat))
        out_inputs = torch.tanh(self.out_gate(torch.cat([curr, prev * reset], dim=1)))
        new_state = prev * (1 - update) + out_inputs * update
        return new_state


def _compute_transform_embedding(transforms, prev_transforms):
    curr_l2g = transforms['lidar2global']
    curr_l2la = transforms['lidar2lidar_aug']
    adj_l2g = prev_transforms['lidar2global']
    adj_l2la = prev_transforms['lidar2lidar_aug']

    # Prepare transformation curr_lidar_aug_2_adj_lidar_aug
    curr2adj = (
        adj_l2la @
        inverse_trans_matrix(adj_l2g) @
        curr_l2g @
        inverse_trans_matrix(curr_l2la))

    transforms_embedding = torch.cat([curr2adj[:, :3, :3].flatten(1, 2),
                                      curr2adj[:, :3, 3]], dim=1)
    return transforms_embedding
