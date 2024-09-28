from functools import partial
from typing import List, Callable

from mmengine.structures import InstanceData
from mmdet.structures.bbox import BaseBoxes

from bev.structures.bev_data_sample import BEVDataSample, SampleList
from bev.utils.typing_utils import OptConfigType

import torch


def get_head_cfg(head: str, cfg: OptConfigType):
    if cfg is None:
        return None
    else:
        return cfg.get(head, None)


def add_loss_prefix(loss_dict: dict, prefix: str) -> dict:
    new_dict = dict()
    for name, value in loss_dict.items():
        new_dict[prefix + '.' + name] = value
    return new_dict


def multi_apply(func: Callable, *args, **kwargs) -> tuple:
    """Apply function to a list of arguments.

    Differences from mmdet.models.utils.multi_apply:
    - This version handles the situation that func returns only one value.

    Caution:
    - The returned value may be wrong when func return a tuple that do not
        want to be separated.

    Note:
    - This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments
        args: Some lists of arguments to be processed.
        kwargs: Key word arguments applied to every func.

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = tuple(map(pfunc, *args))
    if isinstance(map_results[0], tuple):
        return tuple(map(list, zip(*map_results)))
    else:
        return map_results


def project_data_samples_to_mono(
        batch_data_samples: List[BEVDataSample],
        transforms,
        ):
    """Convert batched data samples to mono type

    The result data samples list have a length of batch_size * num_views.
    Only metainfo and gt/pred related to mono will be preserved in results.

    """
    num_views = batch_data_samples[0].num_views
    batch_mono_data_samples = []

    for sample_idx, ds in enumerate(batch_data_samples):
        metainfo = ds.metainfo
        bboxes_3d = ds.gt_instances_3d.get('bboxes_3d')
        labels_3d = ds.gt_instances_3d.get('labels_3d')

        h, w = metainfo['batch_input_shape']
        corners_3d = bboxes_3d.corners
        corners_3d = torch.cat([corners_3d, torch.ones_like(corners_3d)[..., :1]], dim=-1).unsqueeze(0)
        corners_2d = torch.matmul(corners_3d.cuda(), transforms['lidar_aug2img_aug'][sample_idx].permute(0, 2, 1).unsqueeze(1).float())
        front_corners = corners_2d[..., 2] > 0 # filter depth < 0
        corners_2d = corners_2d[..., :2] / (corners_2d[..., 2:3] + 1e-7)  # [num_cams, num_bboxes, 8, 2]
        valid_corners_2d = ((corners_2d[..., 0] > 0) & (corners_2d[..., 0] < (w - 1))) & (
                    (corners_2d[..., 1] > 0) & (corners_2d[..., 1] < (h - 1)))
        valid_corners_2d = (torch.sum(valid_corners_2d, dim=-1) > 0) & (torch.sum(front_corners, dim=-1) > 0)
        corners_2d[..., 0].clip_(0, w-1)
        corners_2d[..., 1].clip_(0, h-1)
        bboxes_2d_top_left = corners_2d.min(dim=2).values
        bboxes_2d_bottom_right = corners_2d.max(dim=2).values
        bboxes_2d = torch.cat([bboxes_2d_top_left, bboxes_2d_bottom_right], dim=-1)

        for cam_idx in range(num_views):
            mono_ds = BEVDataSample()

            # get mono metainfo
            mono_metainfo = dict()
            for key, value in metainfo.items():
                if isinstance(value, list) and len(value) == num_views:
                    mono_metainfo[key] = value[cam_idx]
                else:
                    mono_metainfo[key] = value

            mono_ds.set_metainfo(mono_metainfo)

            valid_bboxes = valid_corners_2d[cam_idx].nonzero().squeeze(-1)

            gt_instances = InstanceData()
            gt_instances['bboxes'] = bboxes_2d[cam_idx, valid_bboxes, :]
            gt_instances['labels'] = labels_3d[valid_bboxes]
            mono_ds.gt_instances = gt_instances
            batch_mono_data_samples.append(mono_ds)

    return batch_mono_data_samples
