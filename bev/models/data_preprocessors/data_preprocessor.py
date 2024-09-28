from typing import List

import torch
from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor

from bev.registry import MODELS
from bev.structures import BEVDataSample


@MODELS.register_module()
class BEVDataPreprocessor(Det3DDataPreprocessor):

    def simple_process(self, data: dict, training: bool = False) -> dict:
        base_inputs = super(BEVDataPreprocessor, self).simple_process(data=data, training=training)
        batch_inputs, data_samples = base_inputs['inputs'], base_inputs['data_samples']

        if 'transforms' in data['inputs']:
            _batch_transforms = self.cast_data(data['inputs']['transforms'])
            batch_transforms = {}
            for name, transform in _batch_transforms.items():
                batch_transforms[name] = torch.stack(transform)
            batch_inputs['transforms'] = batch_transforms

        batch_inputs = self.split_frames(batch_inputs, data_samples)

        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def split_frames(self,
                     batch_inputs: dict,
                     data_samples: List[BEVDataSample]):
        """Split multi-frame inputs.

        Note: All adjacent inputs have an extra time_squence dimention T.

        Returns:
            batch_inputs which contains:
                imgs (Tensor): Tensor with shape (B, N, C, H, W).
                prev_imgs (List[Tensor]): List of tensors with shape
                    (B, N, C, H, W). Length is num_prev_frames.
                next_imgs (List[Tensor]): List of tensors with shape
                    (B, N, C, H, W). Length is num_next_frames.
                transforms (Dict[str, Tensor]): Transformation matrics dict
                    with shape (B, N, d, d) or (B, d, d).
                prev_transforms (List[Dict[str, Tensor]]): List of
                    transformation matrics dict with shape (B, N, d, d)
                    or (B, d, d).
                next_transforms: (List[Dict[str, Tensor]]): List of
                    transformation matrics dict with shape (B, N, d, d)
                    or (B, d, d).
        """
        num_prev_frames = data_samples[0].get('num_prev_frames', 0)
        num_next_frames = data_samples[0].get('num_next_frames', 0)
        num_total_frames = data_samples[0].get('num_total_frames', 1)
        assert num_total_frames == 1 + num_prev_frames + num_next_frames

        imgs = batch_inputs['imgs']
        transforms = batch_inputs['transforms']

        T = num_total_frames
        N = data_samples[0].metainfo['num_views']

        # Split imgs
        imgs_list = imgs.split(N, dim=1)
        curr_imgs = imgs_list[num_prev_frames]
        prev_imgs = imgs_list[:num_prev_frames]
        next_imgs = imgs_list[num_prev_frames:]

        # Split transoforms
        transforms_list = [dict() for _ in range(T)]
        for key in transforms:
            trans = transforms[key]

            # transforms with shape (B, TN, d, d)
            if 'img' in key or 'cam' in key:
                trans_list = trans.split(N, dim=1)
            # transforms with shape (B, T, d, d)
            elif key == 'lidar2global':
                trans_list = trans.split(1, dim=1)
                trans_list = [trans.squeeze(dim=1) for trans in trans_list]
            elif key == 'time_interval':
                trans_list = trans.split(1, dim=1)
            # transforms with shape (B, d, d)
            else:
                trans_list = [trans for _ in range(T)]

            for t in range(T):
                transforms_list[t][key] = trans_list[t]

        curr_transforms = transforms_list[num_prev_frames]
        prev_transforms = transforms_list[:num_prev_frames]
        next_transforms = transforms_list[num_prev_frames:]

        # Save to input dict
        batch_inputs['imgs'] = curr_imgs
        batch_inputs['prev_imgs_list'] = prev_imgs
        batch_inputs['next_imgs_list'] = next_imgs

        batch_inputs['transforms'] = curr_transforms
        batch_inputs['prev_transforms_list'] = prev_transforms
        batch_inputs['next_transforms_list'] = next_transforms

        return batch_inputs
