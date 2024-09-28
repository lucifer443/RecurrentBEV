from typing import List, Sequence, Optional

import torch
from torch import Tensor
from mmengine.model import BaseModule

from bev.registry import MODELS
from bev.structures import BEVDataSample
from bev.utils.typing_utils import ConfigType, OptConfigType
from ..utils.misc import multi_apply

from mmdet.models.layers import ResLayer
from mmdet.models.backbones.resnet import BasicBlock


@MODELS.register_module()
class ViewTransformer(BaseModule):
    """View transformer that transforms img features to bev features
    (perspective view to bev view)
    """
    def __init__(self,
                 shared_encoder: OptConfigType = None,
                 img_encoder: OptConfigType = None,
                 depth_encoder: OptConfigType = None,
                 view_trans: ConfigType = None,
                 post_process: OptConfigType = None,
                 loss_depth: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        """
        Args:
            grid_config (dict): BEV map grid config.
                Default:{
                    'xbound': [-51.2, 51.2, 0.8],
                    'ybound': [-51.2, 51.2, 0.8],
                    'zbound': [-10.0, 10.0, 20.0],
                    'dbound': [1.0, 60.0],
                    'dbins': 59,
                    'dmode': 'UD'}
            input_size (tuple): Input size of the entire model
                (before backbone). Default: (256, 704).
            downsample (int): Downsample between backbone input and view
                transformer input. Default: 16.
            shared_encoder (dict): Config of middle encoder applied on the input
                feature inseide view transformer. Default: None.
                It's in_channels should be same as out_channels of img neck.
            img_encoder (dict): Config of img encoder inside view transformer.
                It's in_channels should be same as out_channels of img neck or
                shared_encdoer when specified.
            depth_encoder (dict): Config of depth encoder. It's in_channels
                should be same as out_channels of img neck or shared_encdoer
                when specified.
            loss_depth (dict): Loss config for depth. When specified, bev
                detector will extraly compute a depth loss.
            pool_method (str): Voxel pooling method to be used.
                Default: 'bevfusion'.
        """
        super().__init__(init_cfg=init_cfg)

        if shared_encoder:
            self.shared_encoder = MODELS.build(shared_encoder)
        if img_encoder:
            self.img_encoder = MODELS.build(img_encoder)
        if depth_encoder:
            self.depth_encoder = MODELS.build(depth_encoder)
        if view_trans:
            self.view_trans = MODELS.build(view_trans)
        if post_process:
            post_process['block'] = BasicBlock
            self.post_process = ResLayer(**post_process)
        if loss_depth is not None:
            self.loss_depth = MODELS.build(loss_depth)

    @property
    def with_shared_encoder(self):
        """bool: Whether the view transformer has a shared encoder
        both used by img feature and depth feature."""
        return hasattr(self,
                       'shared_encoder') and self.shared_encoder is not None

    @property
    def with_img_encoder(self):
        """bool: Whether the view transformer has a img encoder
        to further encoding img feature."""
        return hasattr(self,
                       'img_encoder') and self.img_encoder is not None

    @property
    def with_depth_encoder(self):
        """bool: Whether the view transformer has a depth encoder
        that extracts depth distribution from img feature."""
        return hasattr(self,
                       'depth_encoder') and self.depth_encoder is not None

    @property
    def with_view_trans(self):
        """bool: Whether the view transformer has a depth encoder
        that extracts depth distribution from img feature."""
        return hasattr(self,
                       'view_trans') and self.view_trans is not None

    @property
    def with_post_process(self):
        return hasattr(self,
                       'post_process') and self.post_process is not None

    @property
    def with_loss_depth(self):
        """bool: Whether the view transformer has a depth loss."""
        return hasattr(self,
                       'loss_depth') and self.loss_depth is not None

    def forward(self,
                feats: Sequence[Tensor],
                transforms: dict,
                batch_input_metas: List[dict],
                prev_frames: Optional[List[Sequence[Tensor]]] = None,
                ):
        """
        Args:
            feats (Sequence[Tensor]): Multi-level img features with shape
                (B, N, C, H, W).
            transforms (dict): Transformation matrixes with shape (B, N, d, d)
                or (B, d, d).
            batch_input_metas (List[dict])
            prev_frames (Optional[List[Sequence[Tensor]]]): When None, it
                means do not use prev frames. When a list, it means use prev
                frames. If the list is empty or shorter than required, it will
                copy the most nearest feature to meet the required length.
                Default: None.
        """
        if self.with_shared_encoder:
            feats = multi_apply(self.shared_encoder, feats)

        if self.with_img_encoder:
            img_feats = self.img_encoder(feats, transforms)
        else:
            img_feats = feats

        if self.with_depth_encoder:
            depth_feats = self.depth_encoder(feats, transforms)
            self.D = depth_feats[0].shape[1]
        else:
            depth_feats = None

        assert self.with_view_trans
        bev_feats = self.view_trans(
            img_feats,
            depth_feats,
            transforms,
            batch_input_metas)

        if self.with_post_process:
            bev_feats = multi_apply(self.post_process, bev_feats)

        return bev_feats, depth_feats

    def loss(self,
             depth_feats: Sequence[Tensor],
             batch_data_samples: List[BEVDataSample]):
        assert len(depth_feats) == 1, 'Now we only support one level depth_feats'
        depth_feat = depth_feats[0]

        batch_gt_depth_dist = []
        batch_gt_depth_valid_mask = []
        for data_sample in batch_data_samples:
            batch_gt_depth_dist.append(data_sample.gt_depth.depth_dist)
            batch_gt_depth_valid_mask.append(data_sample.gt_depth.depth_valid_mask)
        gt_depth = torch.stack(batch_gt_depth_dist)
        gt_depth_mask = torch.stack(batch_gt_depth_valid_mask)
        assert gt_depth.shape == gt_depth_mask.shape
        # Get ground truth distribution
        B, N, H, W, _ = gt_depth.shape
        gt_depth = gt_depth.reshape(B * N, H ,W, self.D)
        gt_depth_mask = gt_depth_mask.reshape(B * N, H ,W, self.D).float()

        # Compute predict distribution
        depth_dist = depth_feat.softmax(dim=1).permute(0, 2, 3, 1)

        loss_depth = self.loss_depth(
            depth_dist,
            gt_depth,
            weight=gt_depth_mask,
            avg_factor=max(1.0, gt_depth_mask.sum()))

        loss_dict = dict(loss_depth=loss_depth)
        return loss_dict
