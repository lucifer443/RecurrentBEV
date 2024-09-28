import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.datasets.transforms import RandomFlip3D as MMDET3D_RandomFlip3D
from mmdet3d.datasets.transforms import GlobalRotScaleTrans as MMDET3D_GlobalRotScaleTrans

from bev.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomFlip3D(MMDET3D_RandomFlip3D):
    """Differences from MMDET3D_RandomFlip3D:

    - Add `_record_homography_matrix_3d`
    - Remove flip of images
    """

    def _record_homography_matrix_3d(self, input_dict: dict) -> np.ndarray:
        """Record the 3d homography matrix for the RandomFlip3D."""
        # Compute transformation matrix
        matrix = np.eye(4, dtype=np.float32)
        if input_dict['pcd_horizontal_flip']:
            matrix[1, 1] = -1.0
        if input_dict['pcd_vertical_flip']:
            matrix[0, 0] = -1.0

        # Update homography_matrix_3d
        if input_dict.get('homography_matrix_3d', None) is None:
            input_dict['homography_matrix_3d'] = matrix
        else:
            input_dict['homography_matrix_3d'] = \
                matrix @ input_dict['homography_matrix_3d']

        return matrix

    def transform(self, input_dict: dict) -> dict:
        if self.sync_2d and 'img' in input_dict:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio_bev_horizontal else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])
        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        # Record 3d homography matrix for flip
        self._record_homography_matrix_3d(input_dict)
        return input_dict


@TRANSFORMS.register_module()
class GlobalRotScaleTrans(MMDET3D_GlobalRotScaleTrans):
    """Differences from MMDET3D_GlobalRotScaleTrans:

    - Add `_record_homography_matrix_3d`
    """
    def _record_homography_matrix_3d(self, input_dict: dict) -> np.ndarray:
        """Record the 3d homography matrix for the RandomFlip3D."""
        # Compute transformation matrix
        matrix = np.eye(4, dtype=np.float32)
        rot_and_scale = input_dict['pcd_rotation'].T * \
            input_dict['pcd_scale_factor']
        trans = input_dict['pcd_trans']
        matrix[:3, :3] = rot_and_scale
        matrix[:3, 3] = trans

        # Update homography_matrix_3d
        if input_dict.get('homography_matrix_3d', None) is None:
            input_dict['homography_matrix_3d'] = matrix
        else:
            input_dict['homography_matrix_3d'] = \
                matrix @ input_dict['homography_matrix_3d']

        return matrix

    def transform(self, input_dict: dict) -> dict:
        input_dict = super().transform(input_dict)
        self._record_homography_matrix_3d(input_dict)
        return input_dict


@TRANSFORMS.register_module()
class GetTransformationMatrices(BaseTransform):
    """Get the following transformation matrices:

    - img2img_aug
    - cam2img
    - lidar2cam
    - lidar2lidar_aug
    - lidar_aug2img_aug

    Required Keys:

    - num_total_frames (int)
    - num_views (int)
    - cam2img (list TN x 3 x 3)
    - lidar2cam (list TN x 4 x 4)
    - lidar2global (list T of np.float32 4 x 4)
    - homography_matrix (optional, list TN of np.float32 3 x 3)
    - homography_matrix_3d (optional, np.float32 4 x 4)

    Modified Keys:

    - cam2img (np.float32 TN x 3 x 3)
    - lidar2cam (np.float32 TN x 4 x 4)
    - lidar2global (np.float32 T x 4 x 4)

    Added Keys:

    - lidar2img (np.float32 TN x 4 x 4)
    - img2img_aug (np.float32 TN x 3 x 3)
    - lidar2lidar_aug (np.float32 4 x 4)
    - lidar_aug2img_aug (np.float32 TN x 4 x 4)
    """
    def __init__(self):
        pass

    def transform(self, results: dict) -> dict:
        assert ('num_total_frames' in results and 'num_views' in results
               ), 'Please use bev.LoadMultiViewImageFromFiles'
        num_imgs = results['num_total_frames'] * results['num_views']

        # cam2img
        results['cam2img'] = np.array(results['cam2img'], dtype=np.float32)
        # lidar2cam
        results['lidar2cam'] = np.array(results['lidar2cam'], dtype=np.float32)
        # lidar2global
        results['lidar2global'] = np.array(results['lidar2global'],
                                           dtype=np.float32).reshape(-1, 4, 4)
        # img2img_aug
        if 'homography_matrix' in results:
            results['img2img_aug'] = np.stack(results['homography_matrix'],
                                              axis=0)
        else:
            results['img2img_aug'] = np.tile(np.eye(3, dtype=np.float32),
                                             reps=(num_imgs, 1, 1))
        # lidar2lidar_aug
        if 'homography_matrix_3d' in results:
            results['lidar2lidar_aug'] = results['homography_matrix_3d']
        else:
            results['lidar2lidar_aug'] = np.eye(4, dtype=np.float32)
        # lidar_aug2img_aug
        img2img_aug = np.tile(np.eye(4, dtype=np.float32),
                              reps=(num_imgs, 1, 1))
        img2img_aug[:, :3, :3] = results['img2img_aug']
        cam2img = np.tile(np.eye(4, dtype=np.float32),
                          reps=(num_imgs, 1, 1))
        cam2img[:, :3, :3] = results['cam2img']
        results['lidar_aug2img_aug'] = (
            img2img_aug @
            cam2img @
            results['lidar2cam'] @
            np.linalg.inv(results['lidar2lidar_aug']))
        # lidar2img
        results['lidar2img'] = cam2img @ results['lidar2cam']
        return results
