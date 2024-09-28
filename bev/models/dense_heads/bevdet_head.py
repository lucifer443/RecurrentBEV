from typing import List, Tuple, Optional

import torch
from mmengine.structures import InstanceData
from mmdet3d.structures.bbox_3d import xywhr2xyxyr
from mmdet3d.models.layers import circle_nms, nms_bev
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.dense_heads import CenterHead

from bev.registry import MODELS


@MODELS.register_module()
class BEVDetHead(CenterHead):
    """Head used in BEVDet <https://arxiv.org/abs/2112.11790v3>.
    Suppport head-specific losses and scale-nms.

    All changs from CenterHead are wrapped in '# =========='

    Args:
        head_specific_loss (bool, optional): Whether to use head-specific losses.
            Default: True.
        kwargs: Please refer to CenterHead.
    """

    def __init__(self,
                 record_separated_head_loss: bool = True,
                 scale_nms: bool = True,
                 per_class_reg: Optional[list] = None,
                 **kwargs):
        super(BEVDetHead, self).__init__(**kwargs)
        self.record_separated_head_loss = record_separated_head_loss
        self.scale_nms=scale_nms
        self.per_class_reg = per_class_reg

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        """Loss function for CenterHead.

        Differences from CenterHead:

        - Add a flag that whether record separate head loss

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d`` and\
                ``labels_3d`` attributes.

        Returns:
            dict[str,torch.Tensor]: Loss of heatmap and bbox of each task.
        """

        heatmaps, anno_boxes, inds, masks = self.get_targets(
            batch_gt_instances_3d)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)
            # >>>>>>>>>>
            if self.record_separated_head_loss:
                loss_bbox_reg = self.loss_bbox(
                    pred[..., 0:2], target_box[..., 0:2], bbox_weights[..., 0:2], avg_factor=(num + 1e-4))
                loss_bbox_height = self.loss_bbox(
                    pred[..., 2:3], target_box[..., 2:3], bbox_weights[..., 2:3], avg_factor=(num + 1e-4))
                loss_bbox_dim = self.loss_bbox(
                    pred[..., 3:6], target_box[..., 3:6], bbox_weights[..., 3:6], avg_factor=(num + 1e-4))
                loss_bbox_rot = self.loss_bbox(
                    pred[..., 6:8], target_box[..., 6:8], bbox_weights[..., 6:8], avg_factor=(num + 1e-4))
                loss_bbox_vel = self.loss_bbox(
                    pred[..., 8:10], target_box[..., 8:10], bbox_weights[..., 8:10], avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_bbox_reg'] = loss_bbox_reg
                loss_dict[f'task{task_id}.loss_bbox_height'] = loss_bbox_height
                loss_dict[f'task{task_id}.loss_bbox_dim'] = loss_bbox_dim
                loss_dict[f'task{task_id}.loss_bbox_rot'] = loss_bbox_rot
                loss_dict[f'task{task_id}.loss_bbox_vel'] = loss_bbox_vel
            else:
                loss_bbox = self.loss_bbox(
                    pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
                loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox

            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            # <<<<<<<<<<
        return loss_dict

    def predict_by_feat(self, preds_dicts: Tuple[List[dict]],
                        batch_input_metas: List[dict], *args,
                        **kwargs) -> List[InstanceData]:
        """Generate bboxes from bbox head predictions.

        Differences from CenterHead:

        - Support different nms_type for different tasks

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results of
                multiple tasks. The outer tuple indicate  different
                tasks head, and the internal list indicate different
                FPN level.
            batch_input_metas (list[dict]): Meta info of multiple
                inputs.

        Returns:
            list[:obj:`InstanceData`]: Instance prediction
            results of each sample after the post process.
            Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (:obj:`LiDARInstance3DBoxes`): Prediction
                  of bboxes, contains a tensor with shape
                  (num_instances, 7) or (num_instances, 9), and
                  the last 2 dimensions of 9 is
                  velocity.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            # ========== for scale nms
            nms_type_cfg = self.test_cfg.get('nms_type')
            if self.scale_nms:
                nms_type = nms_type_cfg[task_id]
            else:
                nms_type = nms_type_cfg
            assert nms_type in ['circle', 'rotate']
            # ==========
            if nms_type == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels,
                                             batch_input_metas,
                                             task_id))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            temp_instances = InstanceData()
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = batch_input_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            temp_instances.bboxes_3d = bboxes
            temp_instances.scores_3d = scores
            temp_instances.labels_3d = labels
            ret_list.append(temp_instances)
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas,
                            # ==========
                            task_id
                            # ==========
                            ):
        """Rotate nms for each task.

        Differences from CenterHead:

        - Support scale nms in BEVDet

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.
            task_id (int): Index of current task.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # ========== scale box size before nms
            if self.scale_nms:
                nms_rescale_factor = self.test_cfg.get(
                    'nms_rescale_factor',
                    [1.0 for _ in range(len(self.task_heads))])[task_id]
                if isinstance(nms_rescale_factor, list):
                    for cid in range(len(nms_rescale_factor)):
                        box_preds[cls_labels==cid, 3:6] = box_preds[cls_labels==cid, 3:6] * nms_rescale_factor[cid]
                else:
                    box_preds[:,3:6] = box_preds[:,3:6] * nms_rescale_factor
            # ==========

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                # ========== for scale nms
                if self.scale_nms:
                    nms_thresh = self.test_cfg['nms_thr'][task_id]
                else:
                    nms_thresh = self.test_cfg['nms_thr']
                # ==========

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=nms_thresh,
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # ========== recover box size after nms
            if self.scale_nms:
                if isinstance(nms_rescale_factor, list):
                    for cid in range(len(nms_rescale_factor)):
                        box_preds[top_labels==cid, 3:6] = box_preds[top_labels==cid, 3:6] / nms_rescale_factor[cid]
                else:
                    box_preds[:, 3:6] = box_preds[:, 3:6] / nms_rescale_factor
            # ==========

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
