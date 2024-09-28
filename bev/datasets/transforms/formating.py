import numpy as np
from mmengine.structures import InstanceData, PixelData, BaseDataElement
from mmdet.structures.bbox import BaseBoxes
from mmdet3d.structures import BaseInstance3DBoxes, PointData
from mmdet3d.structures.points import BasePoints
from mmdet3d.datasets.transforms import Pack3DDetInputs as MMDET3D_Pack3DDetInputs
from mmdet3d.datasets.transforms.formating import to_tensor

from bev.registry import TRANSFORMS
from bev.structures import BEVDataSample


@TRANSFORMS.register_module()
class Pack3DDetInputs(MMDET3D_Pack3DDetInputs):
    """Add processes to TRANSFORMS_KEYS, DEPTH_KEYS, MAP_KEYS,
    MONO_2D_KEYS, MONO_TRAFF_2D_KEYS, OCC_KEYS
    """
    # >>>>>>>>>>
    TRANSFORMS_KEYS = [
        'cam2img', 'lidar2cam', 'lidar2img', 'lidar2global',
        'img2img_aug', 'lidar2lidar_aug', 'lidar_aug2img_aug', 'time_interval'
    ]
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d'
    ]
    DEPTH_KEYS = [
        'gt_depth_dist', 'gt_depth_valid_mask',
    ]
    # <<<<<<<<<<

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img
                - transforms

            - 'data_samples' (:obj:`BEVDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = to_tensor(imgs)
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results['img'] = to_tensor(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))

        # 将下列字段使用通用规则转成tensor
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers_2d', 'depths', 'gt_labels_3d',
                # >>>>>>>>>>
                'gt_depth_dist', 'gt_depth_valid_mask',
                'cam2img', 'lidar2cam', 'lidar2img', 'lidar2global',
                'img2img_aug', 'lidar2lidar_aug', 'lidar_aug2img_aug',
                # <<<<<<<<<<
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])

        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = BEVDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()
        gt_depth = BaseDataElement()

        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                if 'num_total_frames' in results and isinstance(results[key], (tuple, list)) and \
                        len(results[key]) == results['num_total_frames']*results['num_views']:
                    img_metas[key] = results[key][-results['num_views']:]
                else:
                    img_metas[key] = results[key]
        data_sample.set_metainfo(img_metas)

        inputs = {}
        transforms = {}

        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.TRANSFORMS_KEYS:
                    transforms[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                elif key in self.DEPTH_KEYS:
                    gt_depth.set_data({self._remove_prefix(key): results[key]})
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')
        inputs['transforms'] = transforms

        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        data_sample.gt_depth = gt_depth
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = dict()

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        return packed_results
