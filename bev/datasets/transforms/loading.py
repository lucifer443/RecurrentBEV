import copy
from typing import Optional, Dict

import torch
import numpy as np
import mmengine
import mmcv
from mmcv.transforms.base import BaseTransform
from mmdet3d.datasets.transforms.formating import to_tensor
from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles \
    as MMDET3D_LoadMultiViewImageFromFiles

from bev.registry import TRANSFORMS
from bev.models.utils import DiscreteDepth


@TRANSFORMS.register_module()
class LoadAdjacentFrames(BaseTransform):
    """Load adjacent frames for temporal usage.
    """
    def __init__(self,
                 num_prev_frames: int = 0,
                 num_next_frames: int = 0):
        self.num_prev_frames = num_prev_frames
        self.num_next_frames = num_next_frames

    def transform(self, results: Dict) -> dict:
        total_prev_frames = len(results['prev_idxs'])
        if total_prev_frames == 0:
            choices = [-1] * self.num_prev_frames
        else:
            choices = list(range(min(total_prev_frames, self.num_prev_frames)))
            while len(choices) < self.num_prev_frames:
                choices.append(choices[-1])

        # Get a deepcopy of required fields of current frame
        curr_info = dict()
        curr_info['lidar2global'] = results['lidar2global']
        curr_info['images'] = dict()
        curr_info['timestamp'] = results['timestamp']
        cam_keys = ['img_path', 'cam2img', 'lidar2cam']
        for cam, cam_info in results['images'].items():
            curr_info['images'][cam] = dict()
            for key in cam_keys:
                curr_info['images'][cam][key] = cam_info[key]

        # Construct a list including current frame for convenient
        info_list: list = results['prev_infos']
        info_list.insert(0, curr_info)
        choices = np.array(choices) + 1

        # Convert required fields to list
        results['lidar2global'] = [results['lidar2global']]
        results['time_interval'] = [0]
        for cam, cam_info in results['images'].items():
            for key in cam_keys:
                cam_info[key] = [cam_info[key]]

        # Add prev frame infos
        # The order of list is from the earliest time to the latest time
        # so current frame is the last element
        for chosen_idx in choices:
            results['lidar2global'].insert(0,
                info_list[chosen_idx]['lidar2global'])
            try:
                results['time_interval'].insert(0,
                    info_list[chosen_idx]['timestamp'] - results['timestamp'])
            except KeyError:
                breakpoint()
            for cam, cam_info in results['images'].items():
                for key in cam_keys:
                    cam_info[key].insert(0,
                        info_list[chosen_idx]['images'][cam][key])

        results['time_interval'] = np.array(results['time_interval'], dtype=np.float)
        # Delete adjacent frame infos
        del results['prev_infos']
        del results['next_infos']

        results['num_prev_frames'] = self.num_prev_frames
        results['num_next_frames'] = self.num_next_frames
        results['num_total_frames'] = 1 + self.num_prev_frames + self.num_next_frames
        return results


@TRANSFORMS.register_module()
class LoadMultiViewImageFromFiles(MMDET3D_LoadMultiViewImageFromFiles):
    """Differences from MMDET3D_LoadMultiViewImageFromFiles:

    - Delete old codes for num_ref_frames.
    - Support loading multi-frame images.
    """
    def __init__(self,
                 imread_backend = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.imread_backend = imread_backend

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam = [], [], []
        # >>>>>>>>>>
        num_total_frames = results.get('num_total_frames', 1)

        # Deal with single frame case
        for _, cam_item in results['images'].items():
            if isinstance(cam_item['img_path'], str):
                cam_item['img_path'] = [cam_item['img_path']]
            if len(cam_item['cam2img']) != num_total_frames:
                cam_item['cam2img'] = [cam_item['cam2img']]
            if len(cam_item['lidar2cam']) != num_total_frames:
                cam_item['lidar2cam'] = [cam_item['lidar2cam']]

        # num_total_frames is outside dimention
        # num_views is inside dimention
        for t in range(num_total_frames):
            for _, cam_item in results['images'].items():
                filename.append(cam_item['img_path'][t])
                cam2img.append(cam_item['cam2img'][t])
                lidar2cam.append(cam_item['lidar2cam'][t])
        # <<<<<<<<<<
        results['filename'] = filename
        results['cam2img'] = cam2img
        results['lidar2cam'] = lidar2cam

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # or (h, w, c, num_total_frames * num_view) for multi-frame
        # h and w can be different for different views
        img_bytes = [
            mmengine.fileio.get(name, backend_args=self.backend_args) \
            for name in filename
        ]
        if self.imread_backend is not None:
            mmcv.use_backend(self.imread_backend)
        imgs = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0)
                for img in imgs
            ]
        if self.to_float32:
            imgs = [img.astype(np.float32) for img in imgs]

        results['filename'] = filename
        results['img'] = imgs
        results['img_shape'] = imgs[0].shape
        results['ori_shape'] = imgs[0].shape
        # Set initial values for default meta_keys
        results['pad_shape'] = imgs[0].shape
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        # >>>>>>>>>>
        results['num_total_frames'] = num_total_frames
        # <<<<<<<<<<
        return results


@TRANSFORMS.register_module()
class ComputeDepthFromPoints(BaseTransform):
    """Compute multi-view depth map form points.

    Args:
        downsample (int): Depth map downsample ratio from image input.
            Default: 16.
        d_min (float): Minimum depth. Default: 0.5.
        d_max (float): Maximum depth. Default: 59.5.
        dbins (int): Num of depth bins. Used by 'cls' ground truth.
        dmode (str): Depth discretization mode. Used by 'cls' ground truth.
        gt_type (str): The output ground truth type. Choose from follows:
            - 'reg': One float value for a pixel position.
            - 'cls': dbins values for a pixel position.
            Default: 'reg'.
        cls_mode (str): Type of 'cls' ground truth. Choose from follows:
            - 'one_hot': One-hot distribution.
            - 'num_based': Distribution based on numbers of different depths in
                one pixel position.
            - 'appear_based': Distribution based on whether a depth is appeared.
    """
    def __init__(self,
                 downsample=16,
                 d_min=0.5,
                 d_max=59.5,
                 dbins=None,
                 dmode=None,
                 gt_type='reg',
                 cls_mode=None):
        self.downsample = downsample
        self.d_min = d_min
        self.d_max = d_max
        self.gt_type = gt_type
        if gt_type == 'cls':
            assert dbins is not None
            assert dmode is not None
            assert cls_mode is not None
            self.dbins = dbins
            self.discrete_depth = DiscreteDepth([d_min, d_max], dbins, dmode)
            self.cls_mode = cls_mode

    def _reserve_min_depth(self, coor, depth, width):
        """
        Reserve the minimum depth for one pixel position.
        """
        ranks = coor[:, 0] * width + coor[:, 1]
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        return coor, depth


    def get_depth_maps(self, points, height, width):
        # points is of shape (N, P, 3)
        N = points.shape[0]
        height, width = height // self.downsample, width // self.downsample
        coors = torch.round(points[:, :, [1, 0]] / self.downsample).long()
        depths = points[..., 2]
        coors_list = [coors[i] for i in range(N)]
        depths_list = [depths[i] for i in range(N)]

        valid_masks = []
        depth_maps = []
        for coor, depth in zip(coors_list, depths_list):
            # remove points out of range
            kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < height) \
                  & (coor[:, 1] >= 0) & (coor[:, 1] < width) \
                  & (depth >= self.d_min) & (depth < self.d_max)
            coor, depth = coor[kept1], depth[kept1]

            # 1.0 in valid_mask means the depth value of this pixel is valid
            # 0.0 means this pixel has no gt or depth value is out of range
            valid_mask = torch.zeros((height, width))
            valid_mask[coor[:, 0], coor[:, 1]] = 1.0

            if self.gt_type == 'reg':
                coor, depth = self._reserve_min_depth(coor, depth, width)
                depth_map = torch.zeros((height, width))
                depth_map[coor[:, 0], coor[:, 1]] = depth
            elif self.gt_type == 'cls':
                # P1是去除超出范围的点后剩余点的数量
                P1 = coor.shape[0]
                depth_indices = self.discrete_depth.get_indices(depth)
                # coor_3d contains indices of (H, W, D)
                coor_3d = torch.cat([coor,
                                     depth_indices.reshape(-1, 1)],
                                     dim=1)
                rank = coor_3d[:, 0] * width * self.dbins \
                     + coor_3d[:, 1] * self.dbins + coor_3d[:, 2]
                sort = rank.argsort()
                coor_3d, rank = coor_3d[sort], rank[sort]
                if self.cls_mode == 'one_hot':
                    rank_hw = coor_3d[:, 0] * width + coor_3d[:, 1]
                    kept_hw = torch.ones(P1).bool()
                    kept_hw[1:] = (rank_hw[1:] != rank_hw[:-1])
                    coor_3d = coor_3d[kept_hw]

                    depth_map = torch.zeros((height, width, self.dbins))
                    coor_indices = coor_3d[:, 0], coor_3d[:, 1], coor_3d[:, 2]
                    depth_map[coor_indices] = 1.0
                elif self.cls_mode == 'num_based':
                    # 保留不同的坐标索引，并统计同一个坐标索引出现的次数
                    kept2 = torch.ones(P1 + 1, dtype=torch.bool)
                    kept2[1:-1] = (rank[1:] != rank[:-1])
                    coor_3d = coor_3d[kept2[:-1]]
                    depth_num_cumsum = torch.arange(P1 + 1)[kept2]
                    depth_num = depth_num_cumsum[1:] - depth_num_cumsum[:-1]

                    depth_map = torch.zeros(height, width, self.dbins)
                    coor_indices = coor_3d[:, 0], coor_3d[:, 1], coor_3d[:, 2]
                    depth_map[coor_indices] = depth_num.float()
                    depth_map = depth_map / (depth_map.sum(dim=2).reshape(
                        height, width, 1) + 1e-6)
                elif self.cls_mode == 'appear_based':
                    depth_map = torch.zeros(height, width, self.dbins)
                    coor_indices = coor_3d[:, 0], coor_3d[:, 1], coor_3d[:, 2]
                    depth_map[coor_indices] = 1.0
                    depth_map = depth_map / (depth_map.sum(dim=2).reshape(
                        height, width, 1) + 1e-6)
                else:
                    raise NotImplementedError

                # valid_mask should be the same shape as depth_map
                valid_mask = valid_mask.reshape(height, width, 1).expand(
                height, width, self.dbins)

            else:
                raise NotImplementedError

            depth_maps.append(depth_map)
            valid_masks.append(valid_mask)

        return torch.stack(depth_maps, dim=0), torch.stack(valid_masks, dim=0)

    def transform(self, results):
        """Call function to generate multi-view depth maps.

        Note:
        - This pipeline should be called after `GetLidarAug2ImgAug`.
        - 'points' and 'img_shape' should be in the result dict.

        Args:
            results (dict): Result dict containing point clouds data and
                lidar_aug2img_aug.

        Returns:
            dict: The result dict containing the multi-view depth maps
                'gt_depth'of shape (N, H, W)
        """
        assert 'points' in results, "ComputeDepthFromPoints needs 'points' to get groud truth depth."
        points = results['points'].tensor[:, :3]
        # lidar_aug coor [x, y, z, 1]
        P = points.shape[0]
        points = torch.cat((points, torch.ones(P, 1)), dim=1)
        points = points.reshape(1, P, 4, 1)
        # allpy lidar_aug2img_aug
        assert 'lidar_aug2img_aug' in results, "ComputeDepthFromPoints needs 'lidar_aug2img_aug' to transform lidar coor to img coor."
        num_prev_frames = results.get('num_prev_frames', 0)
        num_next_frames = results.get('num_next_frames', 0)
        num_total_frames = results.get('num_total_frames', 1)
        num_views = results['num_views']
        assert num_prev_frames + num_next_frames + 1 == num_total_frames
        # use lidar_aug2img_aug of current frame
        lidar_aug2img_aug = results['lidar_aug2img_aug'][
            num_prev_frames * num_views : (num_prev_frames + 1) * num_views]
        lidar_aug2img_aug = to_tensor(lidar_aug2img_aug).unsqueeze(dim=1)
        points = lidar_aug2img_aug @ points
        # img_aug coor [u_a * d, v_a * d, d]
        points = points[..., :3, 0]
        # [u_a, v_a, d]
        points = torch.cat(
            [points[..., :2] / points[..., 2:3], points[..., 2:3]],
            dim=-1)

        assert 'img_shape' in results, "ComputeDepthFromPoints needs 'img_shape' to generate ground truth depth map."
        img_shape = results['img_shape'][0]
        depth_maps, valid_masks = self.get_depth_maps(points, *img_shape[:2])

        results['gt_depth_dist'] = depth_maps
        results['gt_depth_valid_mask'] = valid_masks
        return results
