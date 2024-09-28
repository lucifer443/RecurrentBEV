from typing import Tuple

import numpy as np
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import RandomCrop as MMDET_RandomCrop

from bev.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RangeLimitedRandomCrop(MMDET_RandomCrop):
    """Differences from MMDET_RandomCrop:

    - This class limit the crop range

    Args:
        relative_x_offset_range (tuple[float]): Relative range of random crop
            in x direction. (x_min, x_max) in [0, 1.0]. Default to (0.0, 1.0).
        relative_y_offset_range (tuple[float]): Relative range of random crop
            in y direction. (y_min, y_max) in [0, 1.0]. Default to (0.0, 1.0).
    """

    def __init__(self,
                 relative_x_offset_range=(0.0, 1.0),
                 relative_y_offset_range=(0.0, 1.0),
                 **kwargs):
        super(RangeLimitedRandomCrop, self).__init__(**kwargs)
        for range in [relative_x_offset_range, relative_y_offset_range]:
            assert 0 <= range[0] <= range[1] <= 1
        self.relative_x_offset_range = relative_x_offset_range
        self.relative_y_offset_range = relative_y_offset_range

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_range_h = (margin_h * self.relative_y_offset_range[0],
                          margin_h * self.relative_y_offset_range[1] + 1)
        offset_h = np.random.randint(*offset_range_h)
        offset_range_w = (margin_w * self.relative_x_offset_range[0],
                          margin_w * self.relative_x_offset_range[1] + 1)
        offset_w = np.random.randint(*offset_range_w)

        # used by _crop_data()
        self.offset_h = offset_h
        self.offset_w = offset_w

        return offset_h, offset_w
