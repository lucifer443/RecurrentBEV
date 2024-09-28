from typing import List, Dict, Optional, Union, Tuple, Sequence
import torch
from torch import Tensor
from mmengine.structures import InstanceData
from mmdet3d.models.detectors import Base3DDetector

from bev.registry import MODELS
from bev.structures.bev_data_sample import BEVDataSample
from bev.utils.typing_utils import ConfigType, OptConfigType
from bev.utils.misc import cast_data
from ..utils import (get_head_cfg, add_loss_prefix, project_data_samples_to_mono, )

from mmengine.runner import autocast


@MODELS.register_module()
class RecurrentBEV(Base3DDetector):
    """Lift 2D featrues to 3D, use 3D features to detect 3D objects."""

    def __init__(self,
                 img_backbone: ConfigType = None,
                 img_neck: ConfigType = None,
                 view_transformer: ConfigType = None,
                 bev_backbone: ConfigType = None,
                 bev_neck: ConfigType = None,
                 bev_bbox3d_head: ConfigType = None,
                 temporal_fusion: ConfigType = None,
                 aux_2d_branch: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 data_preprocessor: ConfigType = None,
                 deploy: bool = False,
                 **kwargs):
        super(RecurrentBEV, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor, **kwargs)

        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck)
        self.view_transformer = MODELS.build(view_transformer)
        self.bev_backbone = MODELS.build(bev_backbone)
        self.bev_neck = MODELS.build(bev_neck)

        bev_bbox3d_train_cfg = get_head_cfg('bev_bbox3d', train_cfg)
        bev_bbox3d_test_cfg = get_head_cfg('bev_bbox3d', test_cfg)
        bev_bbox3d_head.update(train_cfg=bev_bbox3d_train_cfg)
        bev_bbox3d_head.update(test_cfg=bev_bbox3d_test_cfg)
        self.bev_bbox3d_head = MODELS.build(bev_bbox3d_head)

        self.temporal_fusion = MODELS.build(temporal_fusion)
        self.prev_bevs_list = []
        self.prev_transforms_list = []
        self.max_cached_frames = 1

        if aux_2d_branch:
            self.img_bbox_neck = MODELS.build(aux_2d_branch.get("neck"))
            self.img_bbox_head = MODELS.build(aux_2d_branch.get("head"))

        self.deploy = deploy

    @property
    def aux_2d_branch(self):
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    def extract_img_backbone_feat(
            self,
            batch_inputs_dict: dict,
            batch_input_metas: List[dict]
    ) -> Sequence[Tensor]:
        imgs = batch_inputs_dict['imgs']
        # Update real input shape of each single img
        input_shape = imgs.shape[-2:]
        for img_meta in batch_input_metas:
            img_meta.update(input_shape=input_shape)

        B, N, Ci, Hi, Wi = imgs.shape
        imgs = imgs.reshape(B * N, Ci, Hi, Wi)
        img_feats = self.img_backbone(imgs)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())

        return img_feats

    def extract_feat(self,
                     batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> dict:
        transforms = batch_inputs_dict['transforms']
        prev_bevs_list = batch_inputs_dict['prev_bevs_list']
        prev_transforms_list = batch_inputs_dict['prev_transforms_list']

        img_backbone_feats = self.extract_img_backbone_feat(batch_inputs_dict, batch_input_metas)
        img_feats = self.img_neck(img_backbone_feats)
        coarse_bev_feats, depth_feats = self.view_transformer(
            img_feats, transforms, batch_input_metas)

        coarse_bev_feats = self.temporal_fusion(
             coarse_bev_feats,
             transforms,
             prev_bevs_list,
             prev_transforms_list)

        bev_feats = self.bev_backbone(coarse_bev_feats[0])
        bev_feats = self.bev_neck(bev_feats)

        return {"img_backbone": img_backbone_feats,
                "coarse_bev": coarse_bev_feats,
                "bev": bev_feats,
                "depth": depth_feats}

    def obtain_prev_bevs(self,
                         batch_inputs_dict: dict,
                         batch_data_samples: List[BEVDataSample]
                         ) -> List[Sequence[Tensor]]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        transforms = batch_inputs_dict['transforms']
        prev_imgs_list = batch_inputs_dict['prev_imgs_list']
        prev_transforms_list = batch_inputs_dict['prev_transforms_list']

        if not len(prev_imgs_list):
            return []

        prev_bevs_list = []
        for t in range(len(prev_imgs_list)):
            # Construct input_dict of this frame
            batch_inputs = dict(
                imgs=prev_imgs_list[t],
                transforms=prev_transforms_list[t],
                prev_bevs_list=prev_bevs_list,
                prev_transforms_list=prev_transforms_list[:t])

            with torch.no_grad():
                img_backbone_feats = self.extract_img_backbone_feat(batch_inputs, batch_input_metas)
                img_feats = self.img_neck(img_backbone_feats)
                bev_feats, depth_feats = self.view_transformer(
                    img_feats, prev_transforms_list[t], batch_input_metas)
            prev_bevs_list.append(bev_feats)

        merged_prev_bevs_list = []
        for t, bevs in enumerate(prev_bevs_list):
            prev_bevs = self.temporal_fusion(bevs,
                                             prev_transforms_list[t],
                                             merged_prev_bevs_list[t-1:t],
                                             prev_transforms_list[t-1:t])
            merged_prev_bevs_list.append(prev_bevs)

        return merged_prev_bevs_list

    def loss(self,
             batch_inputs_dict: dict,
             batch_data_samples: List[BEVDataSample],
             **kwargs) -> Union[dict, list]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        # Obtain infos of previous frames
        is_training = self.training
        self.eval()
        prev_bevs_list = self.obtain_prev_bevs(batch_inputs_dict,
                                               batch_data_samples)
        self.train(mode=is_training)

        # Update batch_inputs_dict and extract features
        batch_inputs_dict['prev_bevs_list'] = prev_bevs_list
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        transforms = batch_inputs_dict['transforms']
        losses = self._loss(batch_data_samples, feats, transforms=transforms, **kwargs)
        return losses

    @autocast(enabled=False, cache_enabled=False)
    def _loss(self,
              batch_data_samples: List[BEVDataSample],
              feats: dict,
              transforms,
              **kwargs) -> dict:
        feats = cast_data(feats, torch.float32)

        losses = dict()

        loss_dict = self.bev_bbox3d_head.loss(
            feats['bev'], batch_data_samples, **kwargs)
        loss_dict = add_loss_prefix(loss_dict, 'bev_bbox3d')
        losses.update(loss_dict)

        if self.view_transformer.with_loss_depth:
            loss_dict = self.view_transformer.loss(
                feats['depth'], batch_data_samples)
            loss_dict = add_loss_prefix(loss_dict, 'view_trans')
            losses.update(loss_dict)

        if self.aux_2d_branch:
            # prepare data_samples for mono 2d detection
            aux_feats = self.img_bbox_neck(feats['img_backbone'])
            batch_mono_data_samples = project_data_samples_to_mono(batch_data_samples, transforms)
            loss_dict = self.img_bbox_head.loss(
                aux_feats, batch_mono_data_samples)
            loss_dict = add_loss_prefix(loss_dict, 'img_bbox')
            losses.update(loss_dict)

        return losses

    def predict(self,
                batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[BEVDataSample],
                **kwargs) -> List[BEVDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]

        # Obtain infos of previous frames
        if batch_input_metas[0]['first_in_scene']:
            self.prev_bevs_list = []
            self.prev_transforms_list = []
        prev_bevs_list = self.prev_bevs_list
        prev_transforms_list = self.prev_transforms_list

        # Update batch_inputs_dict and extract features
        batch_inputs_dict['prev_bevs_list'] = prev_bevs_list
        batch_inputs_dict['prev_transforms_list'] = prev_transforms_list
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        # Save infos of previous frames for video test
        self.prev_bevs_list.append(feats['coarse_bev'])
        self.prev_transforms_list.append(batch_inputs_dict['transforms'])
        while len(self.prev_bevs_list) > self.max_cached_frames:
            self.prev_bevs_list.pop(0)
            self.prev_transforms_list.pop(0)

        preds = self.bev_bbox3d_head.predict(
            feats['bev'], batch_data_samples, **kwargs)
        for i, data_sample in enumerate(batch_data_samples):
            data_sample.pred_instances_3d = preds[i]
            data_sample.pred_instances = InstanceData()  # Needed by mmdet3d.NuscenesMetric
        return batch_data_samples

    def _forward(self,
                 batch_inputs_dict: Dict[str, Optional[Tensor]],
                 metainfos: List[dict]) -> dict:
        feats = self.extract_feat(batch_inputs_dict, metainfos)
        outs = dict()
        outs['bev_bbox3d'] = self.bev_bbox3d_head.forward(feats['bev'])
        return outs

    def switch_to_deploy(self, **kwargs):
        for name, module in self.named_modules():
            if name == '': continue
            if hasattr(module, 'switch_to_deploy'):
                if module.deploy == True: continue
                print(f'Switch {name} to deploy mode.')
                if 'backbone' in name:
                    module.switch_to_deploy()
                else:
                    module.switch_to_deploy(**kwargs)
        self.deploy = True
