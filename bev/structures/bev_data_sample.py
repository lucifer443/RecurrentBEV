from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.structures import BaseDataElement
from mmdet3d.structures import Det3DDataSample


class BEVDataSample(Det3DDataSample):

    @property
    def gt_depth(self) -> BaseDataElement:
        return self._gt_depth

    @gt_depth.setter
    def gt_depth(self, value: BaseDataElement):
        self.set_field(value, '_gt_depth', dtype=BaseDataElement)

    @gt_depth.deleter
    def gt_depth(self):
        del self._gt_depth

    @property
    def eval_ann_info(self) -> dict:
        return self._eval_ann_info

    @eval_ann_info.setter
    def eval_ann_info(self, value: dict):
        self.set_field(value, '_eval_ann_info', dtype=dict)

    @eval_ann_info.deleter
    def eval_ann_info(self):
        del self._eval_ann_info


SampleList = List[BEVDataSample]
OptSampleList = Optional[SampleList]
ForwardResults = Union[Dict[str, torch.Tensor], List[BEVDataSample],
                       Tuple[torch.Tensor], torch.Tensor]
