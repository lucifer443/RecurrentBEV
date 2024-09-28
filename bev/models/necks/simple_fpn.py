import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

from bev.registry import MODELS


@MODELS.register_module()
class Concat2FPN(BaseModule):
    """A simple FPN that use fuse TWO features from different levels by
        `Concat`.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=2,
                 extra_upsample=False,
                 extra_channels=None,
                 norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(Concat2FPN, self).__init__(init_cfg)
        self.extra_upsample = extra_upsample
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        self.conv = nn.Sequential(
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg),
            ConvModule(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                norm_cfg=norm_cfg))
        if self.extra_upsample:
            assert extra_channels is not None
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=True),
                ConvModule(
                    out_channels,
                    extra_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm_cfg=norm_cfg),
                nn.Conv2d(
                    extra_channels,
                    extra_channels,
                    kernel_size=1,
                    padding=0))

    def forward(self, feats):
        x1, x2 = feats
        x2 = self.up(x2)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        if self.extra_upsample:
            out = self.up2(out)
        return (out, )
