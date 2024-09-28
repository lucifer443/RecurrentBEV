import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet.models.backbones import ResNet as MMDET_ResNet

from bev.registry import MODELS


@MODELS.register_module()
class CustomResNet(MMDET_ResNet):
    """ResNet with some additional features.

    New features:
    - Add an option of whether to use stem.
    - Add an option of whether to use maxpool in stem.

    Args:
        with_stem (bool): Whether to use stem. Default: True.
        with_maxpool_in_stem (bool):  Whether to use maxpool in stem.
            Default: True.
        out_input (bool): Whether to put the input of this backbone in the outs.
    """
    def __init__(self,
                 with_stem=True,
                 with_maxpool_in_stem=True,
                 out_input=False,
                 **kwargs):
        self.with_stem = with_stem
        self.with_maxpool_in_stem = with_maxpool_in_stem
        self.out_input = out_input
        super(CustomResNet, self).__init__(**kwargs)
    
    def _make_stem_layer(self, in_channels, stem_channels):
        # Add a branch to avoid init stem layer
        if not self.with_stem:
            return
        elif self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        if self.with_maxpool_in_stem:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Forward function."""
        outs = []
        if self.out_input:
            outs.append(x)
        if self.with_stem:
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            if self.with_maxpool_in_stem:
                x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
