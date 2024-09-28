import numpy as np
from mmdet3d.datasets import NuScenesDataset as MMDET3D_NuScenesDataset

from bev.registry import DATASETS
from .transforms import LoadAdjacentFrames


@DATASETS.register_module()
class NuScenesDataset(MMDET3D_NuScenesDataset):
    """Differences from MMDET3D_NuScenesDataset:

    - parse_data_info():
        - Add more transformation matrices.
    - get_data_info():
        - Support load previous and next frame infos.
    """

    def parse_data_info(self, info: dict) -> dict:
        # process some transformation matrices
        ego2global = np.array(info['ego2global'], dtype=np.float32)
        lidar2ego = np.array(
            info['lidar_points']['lidar2ego'], dtype=np.float32)
        info['ego2global'] = ego2global
        info['lidar2ego'] = lidar2ego
        info['lidar2global'] = ego2global @ lidar2ego

        # add first_in_scene flag
        total_prev_frames = len(info['prev_idxs'])
        info['first_in_scene'] = True if total_prev_frames == 0 else False

        data_info = super().parse_data_info(info)
        return data_info

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)

        for transform in self.pipeline.transforms:
            # add prev_infos and next_infos
            if isinstance(transform, LoadAdjacentFrames):
                num_prev_frames = transform.num_prev_frames
                num_next_frames = transform.num_next_frames

                prev_infos = []
                for i in data_info['prev_idxs'][:num_prev_frames]:
                    prev_infos.append(super().get_data_info(i))
                data_info['prev_infos'] = prev_infos

                next_infos = []
                for i in data_info['next_idxs'][:num_next_frames]:
                    prev_infos.append(super().get_data_info(i))
                data_info['next_infos'] = next_infos
                break

        return data_info
