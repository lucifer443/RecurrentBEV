import os.path as osp
from typing import Optional, Sequence, List

import mmcv
from mmengine.fileio import get
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.utils import mkdir_or_exist
from mmengine.visualization import Visualizer

from bev.registry import HOOKS
from bev.structures import BEVDataSample


@HOOKS.register_module()
class BEVVisualizationHook(Hook):
    """BEV Detection Visualization Hook. Used to visualize validation and
    testing process prediction results.

    Different from mmdet3d.Det3DVisualizationHook:
    - Remove all codes for show.
    - Remove vis_task.
    - Move `self._test_index += 1` to the bottom.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.
    2. If ``test_out_dir`` is specified, it means that the prediction results
        need to be saved to ``test_out_dir``. In order to avoid vis_backends
        also storing data, so ``vis_backends`` needs to be excluded.
    3. ``vis_backends`` takes effect if the user does not specify ``show``
        and `test_out_dir``. You can set ``vis_backends`` to WandbVisBackend or
        TensorboardVisBackend to store the prediction result in Wandb or
        Tensorboard.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        save_to_backend (bool)
        save_to_file (bool)
        save_dir (str, optional): directory where painted images
            will be saved in testing process.
        interval (int): The interval of visualization. Defaults to 50.
        score_thr (float): The threshold to visualize the bboxes
            and masks. Defaults to 0.3.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 draw: bool = False,
                 save_dest: str = 'file',
                 interval: int = 50,
                 score_thr: float = 0.3,
                 backend_args: Optional[dict] = None):
        assert save_dest in ['file', 'vis_backend']
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.draw = draw
        self.save_dest = save_dest

        self.interval = interval
        self.score_thr = score_thr
        self.backend_args = backend_args
        self._test_index = 0

    def after_val_iter(self,
                       runner: Runner,
                       batch_idx: int,
                       data_batch: dict,
                       outputs: Sequence[BEVDataSample]) -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        # There is no guarantee that the same batch of images
        # is visualized for each evaluation.
        total_curr_iter = runner.iter + batch_idx

        # Visualize only the first data
        data_input = dict()

        # Add imgs
        if 'filename' in outputs[0]:
            filenames = outputs[0].filename
            assert isinstance(filenames, Sequence)
            data_input['imgs'] = []
            for filename in filenames:
                img_bytes = get(filename, backend_args=self.backend_args)
                img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                data_input['imgs'].append(img)

        # Add transforms
        data_input['transforms'] = dict()
        for key, trans in data_batch['inputs']['transforms'].items():
            data_input['transforms'][key] = trans[0]

        if total_curr_iter % self.interval == 0:
            self._visualizer.add_datasample(
                'val sample',
                data_input,
                data_sample=outputs[0],
                pred_score_thr=self.score_thr,
                step=total_curr_iter)

    def after_test_iter(self,
                        runner: Runner,
                        batch_idx: int,
                        data_batch: dict,
                        outputs: Sequence[BEVDataSample]) -> None:
        """Run after every testing iterations.

        Args:
            runner (:obj:`Runner`): The runner of the testing process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`DetDataSample`]): A batch of data samples
                that contain annotations and predictions.
        """
        if self.draw is False:
            return

        if self.save_dest == 'file':
            save_dir = osp.join(runner.work_dir, runner.timestamp,
                                'vis_data/vis_image')
            mkdir_or_exist(save_dir)

        for i, data_sample in enumerate(outputs):
            data_input = dict()
            # Add imgs
            if 'filename' in data_sample:
                filenames = data_sample.filename
                assert isinstance(filenames, Sequence)
                data_input['imgs'] = []
                for filename in filenames:
                    img_bytes = get(filename, backend_args=self.backend_args)
                    img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    data_input['imgs'].append(img)

            # Add transforms
            data_input['transforms'] = dict()
            for key, trans in data_batch['inputs']['transforms'].items():
                data_input['transforms'][key] = trans[i]

            if self.save_dest == 'file':
                sample_idx = data_sample.sample_idx
                out_file = osp.join(save_dir, f'sample-{sample_idx}.jpg')
            else:
                out_file = None

            self._visualizer.add_datasample(
                'test sample',
                data_input,
                data_sample=data_sample,
                out_file=out_file,
                bboxes_3d_score_thr=self.score_thr,
                step=self._test_index)

            self._test_index += 1
