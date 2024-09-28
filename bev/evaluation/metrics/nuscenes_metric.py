import os.path as osp
from tempfile import TemporaryDirectory
from typing import List, Dict, Optional, Sequence, Union
from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor
from pyquaternion import Quaternion
from prettytable import PrettyTable
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
import mmengine
from mmengine.logging import print_log
from mmengine.evaluator import BaseMetric
from mmdet3d.structures import LiDARInstance3DBoxes

from bev.registry import METRICS


@METRICS.register_module()
class NuScenesMetric(BaseMetric):
    """
    """
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }

    def __init__(self,
                 data_root: str,
                 tasks: Union[List[str], str] = 'detection',
                 modality: dict = dict(use_camera=True, use_lidar=False),
                 prefix: Optional[str] = None,
                 save_dir: Optional[str] = None,
                 format_only: bool = False,
                 verbose: bool = False,
                 render_curves: bool = False,
                 collect_device: str = 'cpu',
                 **kwargs) -> None:
        self.default_prefix = 'nuscenes'
        super().__init__(collect_device=collect_device, prefix=prefix)

        self.data_root = data_root
        if isinstance(tasks, str):
             tasks = [tasks]
        self.with_detection = True if 'detection' in tasks else False
        self.with_tracking = True if 'tracking' in tasks else False
        self.with_prediction = True if 'prediction' in tasks else False
        if modality is None:
            modality = dict(
                use_camera=True,
                use_lidar=False,
            )
        self.modality = modality

        self.save_dir = save_dir
        self.format_only = format_only
        if self.format_only:
            assert save_dir is not None, 'save_dir must not be '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.verbose = verbose
        self.render_curves = render_curves
        self.detection_eval_config = config_factory('detection_cvpr_2019')
        self.tracking_eval_config = config_factory('tracking_nips_2019')

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            result['token'] = data_sample['token']
            result['lidar2global'] = data_sample['lidar2global']
            self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.
        """
        self.classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']

        if self.save_dir is None:
            use_tmp_dir = True
            tmp_dir = TemporaryDirectory()
            save_dir = tmp_dir.name
        else:
            use_tmp_dir = False
            save_dir = self.save_dir

        self.format_results(results, save_dir)

        if self.format_only:
            assert self.save_dir is not None
            print_log(f'Results are saved in {osp.basename(save_dir)}',
                      logger='current')
            return {}

        metric_dict = self.nusc_evaluate(save_dir)

        if use_tmp_dir:
            tmp_dir.cleanup()

        return metric_dict

    def format_results(
        self,
        results: List[dict],
        save_dir: Optional[str] = None
    ) -> str:
        """Format the mmdet3d results to standard NuScenes json file.

        Args:
            results (List[dict]): Testing results of the dataset.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            save_dir (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
            dict containing the json filepaths, ``tmp_dir`` is the temporal
            directory created for saving json files when ``jsonfile_prefix`` is
            not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert isinstance(results[0]['pred_instances_3d']['bboxes_3d'],
                          LiDARInstance3DBoxes)

        print(f'\nFormating results to nuscenes standard format...')

        if self.with_detection:
             det_results = NuscDetResults(meta=self.modality, classes=self.classes)
        if self.with_tracking:
             track_results = NuscTrackResults(meta=self.modality, classes=self.classes)

        for res in mmengine.track_iter_progress(results):
            token = res['token']
            lidar2global = res['lidar2global']
            bboxes: LiDARInstance3DBoxes = res['pred_instances_3d']['bboxes_3d']
            scores = res['pred_instances_3d']['scores_3d']
            labels = res['pred_instances_3d']['labels_3d']
            if self.with_tracking:
                 instances_id = res['pred_instances_3d']['instances_id']

            # convert bboxes to global coordinate
            bboxes.rotate(lidar2global[0, :3, :3].mT)
            bboxes.translate(lidar2global[0, :3, 3])

            if self.with_detection:
                det_results.add_sample_results(token, bboxes, scores, labels)
            if self.with_tracking:
                track_results.add_sample_results(
                    token, bboxes, scores, labels, instances_id)

        if self.with_detection:
            det_results.save_submission(osp.join(save_dir, 'detection'))
        if self.with_tracking:
            track_results.save_submission(osp.join(save_dir, 'tracking'))

    def nusc_evaluate(self, save_dir: dict) -> Dict[str, float]:
        """Evaluation in Nuscenes protocol.

        Args:
            result_dict (dict): Formatted results of the dataset.
            metric (str): Metrics to be evaluated. Defaults to 'bbox'.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            logger (MMLogger, optional): Logger used for printing related
                information during evaluation. Defaults to None.

        Returns:
            Dict[str, float]: Results of each evaluation metric.
        """
        metric_dict = dict()

        if self.with_detection:
            det_metrics = self._nusc_detection_eval(osp.join(save_dir, 'detection'))
            metric_dict.update(det_metrics)

        if self.with_tracking:
             track_metrics = self._nusc_tracking_eval(osp.join(save_dir, 'tracking'))
             metric_dict.update(track_metrics)

        return metric_dict

    def _nusc_detection_eval(self, save_dir):
        from nuscenes.eval.detection.evaluate import DetectionEval

        print(f'Evaluating 3d detection results...')
        result_path = osp.join(save_dir, 'submission.json')

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=self.verbose)
        nusc_eval = DetectionEval(
            nusc,
            config=self.detection_eval_config,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=save_dir,
            verbose=self.verbose)
        nusc_eval.main(render_curves=self.render_curves)

        # record metrics
        metric_dict = {}
        metrics = mmengine.load(osp.join(save_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = 'detection'
        for name in self.classes:
            for k, v in metrics['label_aps'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_AP_dist_{k}'] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{name}_{k}'] = val
            for k, v in metrics['tp_errors'].items():
                val = float(f'{v:.4f}')
                detail[f'{metric_prefix}/{self.ErrNameMapping[k]}'] = val

        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        metric_dict.update(detail)

        # print metrics
        pre = 'detection/'
        total_table = PrettyTable(header=False)
        total_table.add_row(['mAP', round(metric_dict[f'{pre}mAP'], 4)])
        total_table.add_row(['mATE', metric_dict[f'{pre}mATE']])
        total_table.add_row(['mASE', metric_dict[f'{pre}mASE']])
        total_table.add_row(['mAOE', metric_dict[f'{pre}mAOE']])
        total_table.add_row(['mAVE', metric_dict[f'{pre}mAVE']])
        total_table.add_row(['mAAE', metric_dict[f'{pre}mAAE']])
        total_table.add_row(['NDS', round(metric_dict[f'{pre}NDS'], 4)])

        print_log('Total detection results:', logger='current')
        print_log('\n' + total_table.get_string(), logger='current')

        cls_table = PrettyTable()
        metrics = ['AP_0.5', 'AP_1.0', 'AP_2.0', 'AP_4.0', 'mAP',
                   'ATE', 'ASE', 'AOE', 'AVE', 'AAE']
        cls_table.field_names = ['Object Class'] + metrics
        for cls in self.classes:
            ap05 = metric_dict[f'{pre}{cls}_AP_dist_0.5']
            ap10 = metric_dict[f'{pre}{cls}_AP_dist_1.0']
            ap20 = metric_dict[f'{pre}{cls}_AP_dist_2.0']
            ap40 = metric_dict[f'{pre}{cls}_AP_dist_4.0']
            ap = round((ap05 + ap10 + ap20 + ap40) / 4, 4)
            ate = metric_dict[f'{pre}{cls}_trans_err']
            ase = metric_dict[f'{pre}{cls}_scale_err']
            aoe = metric_dict[f'{pre}{cls}_orient_err']
            ave = metric_dict[f'{pre}{cls}_vel_err']
            aae = metric_dict[f'{pre}{cls}_attr_err']
            if cls == 'construction_vehicle': cls = 'construction'
            cls_table.add_row(
                [cls, ap05, ap10, ap20, ap40, ap, ate, ase, aoe, ave, aae])

        print_log('Per-class detection results:', logger='current')
        print_log('\n' + cls_table.get_string(), logger='current')


        return metric_dict

    def _nusc_tracking_eval(self, save_dir):
        from nuscenes.eval.tracking.evaluate import TrackingEval

        print(f'Evaluating tracking results...')
        result_path = osp.join(save_dir, 'submission.json')

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = TrackingEval(
            config=self.tracking_eval_config,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=save_dir,
            nusc_version=self.version,
            nusc_dataroot=self.data_root,
            verbose=self.verbose)
        nusc_eval.main(render_curves=self.render_curves)

        # record and print metrics
        metrics = mmengine.load(osp.join(save_dir, 'metrics_summary.json'))
        metric_dict = {}
        total_table = PrettyTable(header=False)

        metric_prefix = 'tracking'
        metric_names = [
            'amota', 'amotp', 'motar', 'mota', 'motp',
            'recall', 'gt', 'mt', 'ml', 'faf', 'tp', 'fp', 'fn',
            'ids', 'frag', 'tid', 'lgd',
        ]

        for name in metric_names:
            if metrics[name].is_integer():
                value = int(metrics[name])
                value_str = str(value)
            else:
                value_str = f'{metrics[name]:.3f}'
                value = float(value_str)
            metric_dict[f'{metric_prefix}/{name}'] = value
            total_table.add_row([f'{name.upper()}', value_str])

        print_log('Total tracking results:', logger='current')
        print_log('\n' + total_table.get_string(), logger='current')

        return metric_dict


class NuscResults(metaclass=ABCMeta):
    def __init__(self,
                 meta: dict = dict(use_camera=True, use_lidar=False),
                 classes: list = None):
        self.meta = meta
        self.classes = classes
        self.results = dict()

    def save_submission(self, save_dir):
        mmengine.mkdir_or_exist(save_dir)
        submission = dict(
            meta=self.meta,
            results=self.results)
        mmengine.dump(
            submission,
            osp.join(save_dir, 'submission.json'))

    @abstractmethod
    def add_sample_results(self, sample_token):
        """Add one result in a sample
        """

class NuscDetResults(NuscResults):
    def add_sample_results(self,
                          sample_token: str,
                          bboxes: LiDARInstance3DBoxes,
                          scores: Tensor,
                          labels: str) -> None:
        if sample_token not in self.results:
            self.results[sample_token] = []

        for i in range(len(bboxes)):
            quaternion = Quaternion(axis=[0, 0, 1], radians=bboxes.yaw[i])
            class_name = self.classes[labels[i]]
            velocity = bboxes.tensor[i, 7:9]
            attr_name = self._infer_attr(class_name, velocity)

            instance_result = dict(
                sample_token=sample_token,
                translation=bboxes.gravity_center[i].tolist(),
                size=bboxes.dims[i, [1, 0, 2]].tolist(),
                rotation=quaternion.elements.tolist(),
                velocity=velocity.tolist(),
                detection_name=class_name,
                detection_score=scores[i].item(),
                attribute_name=attr_name)

            self.results[sample_token].append(instance_result)

    def _infer_attr(self, class_name: str, velocity: Tensor):
        DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
        }

        speed = torch.linalg.vector_norm(velocity)
        if speed > 0.2:
            if class_name in [
                'car',
                'construction_vehicle',
                'bus',
                'truck',
                'trailer',
            ]:
                attr = 'vehicle.moving'
            elif class_name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = DefaultAttribute[class_name]
        else:
            if class_name in ['pedestrian']:
                        attr = 'pedestrian.standing'
            elif class_name in ['bus']:
                        attr = 'vehicle.stopped'
            else:
                attr = DefaultAttribute[class_name]
        return attr

class NuscTrackResults(NuscResults):
    def add_sample_results(self,
                          sample_token: str,
                          bboxes: LiDARInstance3DBoxes,
                          scores: Tensor,
                          labels: str,
                          instances_id: Tensor) -> None:
        if sample_token not in self.results:
            self.results[sample_token] = []

        tracking_names = [
            'bicycle', 'bus', 'car', 'motorcycle', 'pedestrian',
            'trailer', 'truck',
        ]
        for i in range(len(bboxes)):
            class_name = self.classes[labels[i]]
            if class_name not in tracking_names:
                continue

            quaternion = Quaternion(axis=[0, 0, 1], radians=bboxes.yaw[i])

            instance_result = dict(
                sample_token=sample_token,
                translation=bboxes.gravity_center[i].tolist(),
                size=bboxes.dims[i, [1, 0, 2]].tolist(),
                rotation=quaternion.elements.tolist(),
                velocity=bboxes.tensor[i, 7:9].tolist(),
                tracking_id=str(instances_id[i].item()),
                tracking_name=class_name,
                tracking_score=scores[i].item())

            self.results[sample_token].append(instance_result)
