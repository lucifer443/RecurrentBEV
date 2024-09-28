import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmcv.cnn import build_conv_layer
from mmdet.models.backbones.resnet import BasicBlock

from bev.registry import MODELS


def get_cam_embedding(cam_channels, transforms):
    """Embed camera parameters from four transformation matrixs.
    """
    assert cam_channels in [12, 34, 10, 24, 27, 33]
    lidar_aug2img_aug = transforms['lidar_aug2img_aug'].float()
    img2img_aug = transforms['img2img_aug'].float()
    cam2img = transforms['cam2img'].float()
    lidar2cam = transforms['lidar2cam'].float()
    B, N, _, _ = img2img_aug.shape
    if 'lidar2lidar_aug' in transforms:
        lidar2lidar_aug = transforms['lidar2lidar_aug'].float()
    else:
        lidar2lidar_aug = torch.eye(4).cuda().expand(B, 4, 4)

    if cam_channels == 12:
        # Num of parameters: C_cam = 12
        cam_emb = lidar_aug2img_aug[:, :, :3, :]
    elif cam_channels == 34:
        # Num of parameters: C_cam = 6 + 4 + 12 + 12 = 34
        cam_emb = torch.cat([
            img2img_aug[:, :, :2, :3].reshape(B, N, -1),
            cam2img[:, :, (0, 1, 0, 1), (0, 1, 2, 2)],
            lidar2cam[:, :, :3, :].reshape(B, N, -1),
            lidar2lidar_aug[:, :3, :].reshape(B, -1).unsqueeze(1).expand(B, N, 12)],
            dim=-1)
    elif cam_channels == 10:
        # Num of parameters: C_cam = 6 + 4 = 10
        cam_emb = torch.cat([
            img2img_aug[:, :, :2, :3].reshape(B, N, -1),
            cam2img[:, :, (0, 1, 0, 1), (0, 1, 2, 2)]],
            dim=-1)
    elif cam_channels == 24:
        # Num of parameters: C_cam = 6 + 4 + 12 + 12 = 34
        cam_emb = torch.cat([
            lidar2cam[:, :, :3, :].reshape(B, N, -1),
            lidar2lidar_aug[:, :3, :].reshape(B, -1).unsqueeze(1).expand(B, N, 12)],
            dim=-1)
    elif cam_channels == 27:
        # Num of parameters: C_cam = 6 + 4 + 12 + 5 = 27
        l2la = lidar2lidar_aug[:, (0, 0, 1, 1, 2), (0, 1, 0, 1, 2)]
        l2la = l2la.unsqueeze(1).expand(B, N, 5)
        cam_emb = torch.cat([
            img2img_aug[:, :, :2, :3].reshape(B, N, -1),
            cam2img[:, :, (0, 1, 0, 1), (0, 1, 2, 2)],
            lidar2cam[:, :, :3, :].reshape(B, N, -1),
            l2la],
            dim=-1)
    elif cam_channels == 33:
        # Num of parameters: C_cam = 12 + 9 + 12 = 33
        l2la = lidar2lidar_aug.unsqueeze(1)
        cam2lidar_aug = l2la @ lidar2cam.inverse()
        cam_emb = torch.cat([
            img2img_aug[:, :, :3, :].reshape(B, N, -1),
            cam2img[:, :, :3, :3].reshape(B, N, -1),
            cam2lidar_aug[:, :, :3, :].reshape(B, N, -1)],
            dim=-1)
    else:
        raise NotImplementedError

    cam_emb = cam_emb.reshape(B * N, -1)
    return cam_emb


class MLP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None):
        super(MLP, self).__init__()
        hidden_channles = hidden_channels or out_channels
        self.fc1 = nn.Linear(in_channels, hidden_channles)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_channles, out_channels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CamEncoder(nn.Module):
    """Encode camera patameters to channel-wise scale factors.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(CamEncoder, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.mlp1 = MLP(in_channels, out_channels)
        self.mlp2 = MLP(out_channels, out_channels)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.bn(x)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.gate(x)
        return x


@MODELS.register_module()
class ConvEncoder(BaseModule):
    """Image encoder that further encodes img features using camera parameters.
    This ImageEcoder is used in the original BEVDepth.

    Args:
        cam_aware (bool): Whether to use camera parameter aware. Defaule: False.
        cam_channesl (int): Channels of camera parameter embedding. Meaning of
            different channel configurations is described in the function
            `get_cam_embedding()` above.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 cam_aware=False,
                 cam_channels=None,
                 init_cfg=None):
        super(ConvEncoder, self).__init__(init_cfg)
        self.cam_aware = cam_aware
        self.cam_channels = cam_channels
        self.deploy = False

        if self.cam_aware:
            self.cam_encoder = CamEncoder(cam_channels, in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, feats, transforms):
        outs = []
        for x in feats:
            if self.cam_aware:
                if self.deploy:
                    scale = self.scale
                else:
                    cam_embedding = get_cam_embedding(self.cam_channels, transforms)
                    scale = self.cam_encoder(cam_embedding)
                x = x * scale[..., None, None]
            x = self.conv(x)
            outs.append(x)

        return outs

    def switch_to_deploy(self, transforms, **kwargs):
        if self.deploy:
            return
        cam_embedding = get_cam_embedding(self.cam_channels, transforms)
        self.register_buffer('scale', self.cam_encoder(cam_embedding))
        self.deploy = True


@MODELS.register_module()
class DepthNet(BaseModule):
    """Depth encoder that extract depth distribution form img features using
    camera parameters.
    This DepthNet is used in the BEVDepth reproduced by HuangJunJie.

    Args:
        with_dcn (bool) : Whether to use deformable conv. Default: True.
        cam_aware (bool): Whether to use camera parameter aware. Defaule: False.
        cam_channesl (int): Channels of camera parameter embedding. Meaning of
            different channel configurations is described in the function
            `get_cam_embedding()` above.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 with_dcn=True,
                 cam_aware=False,
                 cam_channels=None,
                 init_cfg=None):
        super(DepthNet, self).__init__(init_cfg)
        self.cam_aware = cam_aware
        self.cam_channels = cam_channels
        self.with_dcn = with_dcn
        self.deploy = False

        self.input_conv = nn.Conv2d(in_channels, mid_channels, 1, 1, 0)
        if self.cam_aware:
            self.cam_encoder = nn.Sequential(
                nn.BatchNorm1d(cam_channels),
                nn.Linear(cam_channels, mid_channels),
                nn.Sigmoid())
        self.res_blocks = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        if self.with_dcn:
            self.dcn = build_conv_layer(
                cfg=dict(
                    type='DCNv2',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    deform_groups=1,
                    bias=True))
        self.conv = nn.Conv2d(mid_channels, out_channels, 1, 1, 0)

    def forward(self, feats, transforms):
        outs = []
        for x in feats:
            x = self.input_conv(x)
            if self.cam_aware:
                if self.deploy:
                    scale = self.scale
                else:
                    cam_embedding = get_cam_embedding(self.cam_channels, transforms)
                    scale = self.cam_encoder(cam_embedding)
                x = x * scale[..., None, None]
            x = self.res_blocks(x)
            if self.with_dcn:
                x = self.dcn(x)
            x = self.conv(x)
            outs.append(x)

        return outs

    def switch_to_deploy(self, transforms, **kwargs):
        if self.deploy:
            return
        cam_embedding = get_cam_embedding(self.cam_channels, transforms)
        self.register_buffer('scale', self.cam_encoder(cam_embedding))
        self.deploy = True


@MODELS.register_module()
class DepthNetOfficial(BaseModule):
    """Depth encoder that extract depth distribution form img features using
    camera parameters.
    This DepthNet is used in the offical BEVDepth.

    Args:
        with_dcn (bool) : Whether to use deformable conv. Default: True.
        cam_aware (bool): Whether to use camera parameter aware. Defaule: False.
        cam_channesl (int): Channels of camera parameter embedding. Meaning of
            different channel configurations is described in the function
            `get_cam_embedding()` above.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 with_dcn=False,
                 cam_aware=False,
                 cam_channels=None,
                 assp_mid_channels=-1,
                 bias=False,
                 init_cfg=None):
        super(DepthNetOfficial, self).__init__(init_cfg)
        self.cam_aware = cam_aware
        self.cam_channels = cam_channels
        self.with_dcn = with_dcn
        self.deploy = False

        if self.cam_aware:
            self.cam_encoder = CamEncoder(cam_channels, in_channels)

        downsample = nn.Conv2d(in_channels, mid_channels, 1, 1, 0) if in_channels != mid_channels else None

        conv_list = [
            BasicBlock(in_channels, mid_channels, downsample=downsample),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if assp_mid_channels > 0:
            conv_list.append(ASPP(mid_channels, assp_mid_channels))
        if self.with_dcn:
            dcn = build_conv_layer(
                cfg=dict(
                    type='DCNv2',
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    dilation=1,
                    deform_groups=1,
                    bias=True))
            conv_list.append(dcn)
        conv_list.append( nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=bias))
        self.depth_encoder = nn.Sequential(*conv_list)

    def forward(self, feats, transforms):
        outs = []
        for x in feats:
            if self.cam_aware:
                if self.deploy:
                    scale = self.scale
                else:
                    cam_embedding = get_cam_embedding(self.cam_channels, transforms)
                    scale = self.cam_encoder(cam_embedding)
                x = x * scale[..., None, None]
            x = self.depth_encoder(x)
            outs.append(x)

        return outs

    def switch_to_deploy(self, transforms, **kwargs):
        if self.deploy:
            return
        cam_embedding = get_cam_embedding(self.cam_channels, transforms)
        self.register_buffer('scale', self.cam_encoder(cam_embedding))
        self.deploy = True


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               inplanes,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
