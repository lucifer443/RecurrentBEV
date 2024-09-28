from typing import List, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModule

from bev.registry import MODELS
from bev.ops import voxel_pool_bevfusion, bev_pool_v2
from ..utils import gen_dx_bx, DiscreteDepth


@MODELS.register_module()
class LiftSplatVT(BaseModule):
    """View transformer which transforms img features to bev features
    Learn more in 'Lift, Splat, Shoot: Encoding Images from
    Arbitrary Camera Rigs by Implicitly Unprojecting to 3D'
    """
    def __init__(self,
                 grid_config: dict,
                 input_size: Sequence = (256, 704),
                 downsample: int = 16,
                 pool_method: str = 'bevfusion',
                 with_3d_pos: bool = True,
                 **kwargs):
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
            pool_method (str): Voxel pooling method to be used.
                Default: 'bevfusion'.
        """
        super(LiftSplatVT, self).__init__()
        self.grid_config = grid_config
        dx, bx, nx = gen_dx_bx(self.grid_config['xbound'],
                               self.grid_config['ybound'],
                               self.grid_config['zbound'],
                               )
        self.register_buffer('dx', dx)
        self.register_buffer('bx', bx)
        self.register_buffer('nx', nx)

        self.dbound = self.grid_config['dbound']
        self.D = self.grid_config['dbins']
        self.dmode = self.grid_config['dmode']
        self.discrete_depth = DiscreteDepth(self.dbound, self.D, self.dmode)

        self.input_size = input_size
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.pool_method = pool_method
        self.deploy = False
        self.with_3d_pos = with_3d_pos
        if self.with_3d_pos:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(
                    self.D * 3,
                    80 * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(
                    80 * 4,
                    80,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            )

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def create_frustum(self) -> Tensor:
        # make grid in image plane
        # origin H, origin W
        ogfH, ogfW = self.input_size
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = self.discrete_depth.bin_depths
        ds = ds.reshape(-1, 1, 1).expand(-1, fH, fW)
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(self.D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(self.D, fH, fW)

        # D x H x W x 3
        # the last dim is the coor after data aug and the real depth [u_a, v_a, d]
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, Trans: Tensor) -> Tensor:
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        device = Trans.device
        B, N, _, _ = Trans.shape
        D, H, W, _ = self.frustum.shape

        # Firstly, get coor befor normalization [u_a * d, v_a * d, d]
        points = torch.cat(
            [self.frustum[:, :, :, :2] * self.frustum[:, :, :, 2:3],
             self.frustum[:, :, :, 2:3]],
            dim=3)
        # Secondly, get 4-dim coor [u_a * d, v_a * d, d, 1]
        points = points.reshape(1, 1, D, H, W, 3, 1)
        ones = torch.ones(1, 1, D, H, W, 1, 1, device=device)
        points = torch.cat((points, ones), dim=5)
        # Thirdly, get lidar_aug coor [x, y, z, 1]
        Trans = Trans.reshape(B, N, 1, 1, 1, 4, 4)
        points = torch.linalg.solve(Trans, points)
        # Finally, get 3-dim coor
        points = points[..., :3, 0]

        return points

    def voxel_pooling(self, geom_feats: Tensor, x: Tensor):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        final = voxel_pool_bevfusion(x, geom_feats, B, self.nx[2], self.nx[1], self.nx[0])

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def voxel_pooling_prepare_v2(self, coor):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # record the index of selected points for acceleration purpose
        ranks_depth = torch.range(
            0, num_points - 1, dtype=torch.int, device=coor.device)
        ranks_feat = torch.range(
            0, num_points // D - 1, dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
        # convert coordinate into the voxel space
        coor = ((coor - (self.bx - self.dx / 2.)) / self.dx)
        coor = coor.long().view(num_points, 3)
        batch_idx = torch.range(0, B - 1).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.nx[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < self.nx[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < self.nx[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
                self.nx[2] * self.nx[1] * self.nx[0])
        ranks_bev += coor[:, 2] * (self.nx[1] * self.nx[0])
        ranks_bev += coor[:, 1] * self.nx[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, geom_feats: Tensor, img_feat: Tensor, depth_dist: Tensor):
        if self.deploy:
            ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = \
                self.ranks_bev, self.ranks_depth, self.ranks_feat, self.interval_starts, self.interval_lengths
        else:
            ranks_bev, ranks_depth, ranks_feat, \
                interval_starts, interval_lengths = \
                self.voxel_pooling_prepare_v2(geom_feats)

        feat = img_feat.permute(0, 1, 3, 4, 2)
        bev_feat_shape = (depth_dist.shape[0], int(self.nx[2]),
                          int(self.nx[1]), int(self.nx[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        final = bev_pool_v2(depth_dist, feat, ranks_depth, ranks_feat, ranks_bev,
                               bev_feat_shape, interval_starts,
                               interval_lengths)

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self,
                img_feats: Sequence[Tensor],
                depth_feats: Sequence[Tensor],
                transforms: dict,
                batch_input_metas: List[dict]):
        """
        Args:
            img_feats (Sequence[Tensor]): Multi-level img features with shape
                (BN, C, H ,W).
            depth_feats (Sequence[Tensor]): Multi-level depth features with
                shape (BN, D, H ,W).
            transforms (dict): Transformation matrixes with shape (B, N, 4, 4).
            batch_input_metas (List[dict])
        """
        assert len(img_feats) == 1
        assert len(depth_feats) == 1
        img_feat = img_feats[0]
        depth_feat = depth_feats[0]

        # Get depth distribution
        depth_dist = self.get_depth_dist(depth_feat)

        # Lift
        B = len(batch_input_metas)
        N = batch_input_metas[0]['num_views']
        _, C, H, W = img_feat.shape
        D = depth_dist.shape[1]
        volume = depth_dist.unsqueeze(1) * img_feat.unsqueeze(2)
        volume = volume.reshape(B, N, C, D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)  # (B, N, D, H, W, C)

        # Splat
        Trans = transforms['lidar_aug2img_aug']
        geom = self.get_geometry(Trans)

        if self.with_3d_pos:
            pos_3d = geom.clone().detach()
            pos_3d[..., 0] = (pos_3d[..., 0] - self.grid_config['xbound'][0]) / \
                             (self.grid_config['xbound'][1] - self.grid_config['xbound'][0])
            pos_3d[..., 1] = (pos_3d[..., 1] - self.grid_config['ybound'][0]) / \
                             (self.grid_config['ybound'][1] - self.grid_config['ybound'][0])
            pos_3d[..., 2] = (pos_3d[..., 2] - self.grid_config['zbound'][0]) / \
                             (self.grid_config['zbound'][1] - self.grid_config['zbound'][0])
            pos_3d = pos_3d.permute(0, 1, 2, 5, 3, 4).reshape(B*N, D*3, H, W)
            pos_3d_embed = self.position_encoder(pos_3d)  # (B*N, C, H, W)
            pos_3d_embed = pos_3d_embed.permute(0, 2, 3, 1).reshape(B, N, 1, H, W, C)
            volume += pos_3d_embed
            
        if self.pool_method == 'bev_pool_v2':
            bev_feat = self.voxel_pooling_v2(
                geom,
                img_feat.view(B, N, C, H, W),
                depth_dist.view(B, N, D, H, W))
        else:
            bev_feat = self.voxel_pooling(geom, volume)
        return (bev_feat, )

    def switch_to_deploy(self, transforms, **kwargs):
        if self.deploy:
            return
        Trans = transforms['lidar_aug2img_aug']
        geom = self.get_geometry(Trans)
        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(geom)
        self.register_buffer('ranks_bev', ranks_bev)
        self.register_buffer('ranks_depth', ranks_depth)
        self.register_buffer('ranks_feat', ranks_feat)
        self.register_buffer('interval_starts', interval_starts)
        self.register_buffer('interval_lengths', interval_lengths)
        self.deploy = True
