import math
from typing import Iterator, List

import torch
from mmengine.dataset import DefaultSampler
from mmengine.fileio import load

from bev.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SceneContinuousSampler(DefaultSampler):
    """This sampler samples continuous data in one rank.
    Each scene is complete, which makes the accuracy of multi-gpu testing correct.

    Note:

    - This sampler should only be used in video test for temporal models.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.round_up == True
        assert self.shuffle == False

        scenes_len = self.get_scenes_len()

        rank_scenes = [math.ceil((len(scenes_len) - rank_idx) / self.world_size) \
            for rank_idx in range(self.world_size)]
        self.ori_rank_samples = [sum(scenes_len[sum(rank_scenes[:rank_idx]):sum(rank_scenes[:rank_idx + 1])]) \
            for rank_idx in range(self.world_size)]
        self.num_samples = max(self.ori_rank_samples)

    def get_scenes_len(self) -> List[int]:
        """Get the length of each scene in the dataset"""

        data_list = load(self.dataset.ann_file)['data_list']
        scene_start_idx = []
        for i, info in enumerate(data_list):
            if len(info['prev_idxs']) == 0:
                scene_start_idx.append(i)
        scene_start_idx.append(len(data_list))

        scenes_len = []
        for i, j in enumerate(scene_start_idx):
            if i != 0:
                scenes_len.append(j - scene_start_idx[i - 1])

        assert sum(scenes_len) == len(data_list)

        return scenes_len

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        indices = torch.arange(len(self.dataset)).tolist()

        # Subsample and add extra samples
        indices = indices[sum(self.ori_rank_samples[:self.rank]):
                          sum(self.ori_rank_samples[:self.rank + 1])]
        # Results list will be truncated to length of dataset in BaseMetric.evaluate()
        # So we need add extra samples to the tail of each rank
        while len(indices) < self.num_samples:
            indices.append(0)

        return iter(indices)
