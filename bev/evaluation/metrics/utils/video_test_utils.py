import logging
import os.path as osp
import pickle
import shutil
import tempfile
from typing import Any, List, Optional

import mmengine
from mmengine.dist import barrier, get_dist_info, broadcast, broadcast_object_list, is_main_process, all_gather_object
from mmengine.evaluator.metric import _to_cpu
from mmengine.logging import print_log
import torch


def video_test_evaluate(self, size: int, ori_rank_samples) -> dict:
    """Evaluate the model performance of the whole dataset after processing
    all batches.

    Args:
        size (int): Length of the entire validation dataset. When batch
            size > 1, the dataloader may pad some data samples to make
            sure all ranks have the same length of dataset slice. The
            ``collect_results`` function will drop the padded data based on
            this size.

    Returns:
        dict: Evaluation metrics dict on the val dataset. The keys are the
        names of the metrics, and the values are corresponding results.
    """
    if len(self.results) == 0:
        print_log(
            f'{self.__class__.__name__} got empty `self.results`. Please '
            'ensure that the processed results are properly added into '
            '`self.results` in `process` method.',
            logger='current',
            level=logging.WARNING)

    # >>>>>>>>>>>>>>>>>>>>>
    if self.collect_device == 'cpu':
        results = video_test_collect_results(
            self.results,
            size,
            self.collect_device,
            tmpdir=self.collect_dir,
            ori_rank_samples=ori_rank_samples)
    else:
        video_test_collect_results(self.results, size, self.collect_device, ori_rank_samples=ori_rank_samples)
    # <<<<<<<<<<<<<<<<<<<<<

    if is_main_process():
        # cast all tensors in results list to cpu
        results = _to_cpu(results)
        _metrics = self.compute_metrics(results)  # type: ignore
        # Add prefix to metric names
        if self.prefix:
            _metrics = {
                '/'.join((self.prefix, k)): v
                for k, v in _metrics.items()
            }
        metrics = [_metrics]
    else:
        metrics = [None]  # type: ignore

    broadcast_object_list(metrics)

    # reset the results list
    self.results.clear()
    return metrics[0]


def video_test_collect_results(results: list,
                    size: int,
                    device: str = 'cpu',
                    tmpdir: Optional[str] = None,
                    ori_rank_samples=None) -> Optional[list]:
    """Collected results in distributed environments.

    Args:
        results (list[object]): Result list containing result parts to be
            collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        device (str): Device name. Optional values are 'cpu' and 'gpu'.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a temporal directory for it.
            ``tmpdir`` should be None when device is 'gpu'. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results(data, size, device='cpu')
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    if device not in ['gpu', 'cpu']:
        raise NotImplementedError(
            f"device must be 'cpu' or 'gpu', but got {device}")

    if device == 'gpu':
        assert tmpdir is None, 'tmpdir should be None when device is "gpu"'
        return collect_results_gpu(results, size, ori_rank_samples)
    else:
        return collect_results_cpu(results, size, ori_rank_samples, tmpdir)


def collect_results_cpu(result_part: List[Any],
                        size: int,
                        ori_rank_samples: List[int],
                        tmpdir: Optional[str] = None) -> Optional[List]:
    """Collect results under cpu mode.

    On cpu mode, this function will save the results on different gpus to
    ``tmpdir`` and collect them by the rank 0 worker.

    Args:
        result_part (list): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.
        tmpdir (str | None): Temporal directory for collected results to
            store. If set to None, it will create a random temporal directory
            for it. Defaults to None.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_cpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8)
        if rank == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:  # type: ignore
        pickle.dump(result_part, f, protocol=2)

    barrier()

    # collect all parts
    if rank != 0:
        return None
    else:
        try:
            # load results of all parts from tmp dir
            part_list = []
            for i in range(world_size):
                path = osp.join(tmpdir, f'part_{i}.pkl')  # type: ignore
                if not osp.exists(path):
                    raise FileNotFoundError(
                        f'{tmpdir} is not an shared directory for '
                        f'rank {i}, please make sure {tmpdir} is a shared '
                        'directory for all ranks!')
                with open(path, 'rb') as f:
                    part_list.append(pickle.load(f))
            # >>>>>>>>>>>>>>>>>>>>>
            # sort the results
            ordered_results = []
            for i in range(world_size):
                # the dataloader may pad some samples
                ordered_results.extend(list(part_list[i])[:ori_rank_samples[i]])
            # <<<<<<<<<<<<<<<<<<<<<
            return ordered_results
        finally:
            # remove tmp dir
            shutil.rmtree(tmpdir)  # type: ignore


def collect_results_gpu(result_part: List[Any], size: int, ori_rank_samples: List[int]) -> Optional[List]:
    """Collect results under gpu mode.

    On gpu mode, this function will encode results to gpu tensors and use gpu
    communication for results collection.

    Args:
        result_part (list[object]): Result list containing result parts
            to be collected. Each item of ``result_part`` should be a picklable
            object.
        size (int): Size of the results, commonly equal to length of
            the results.

    Returns:
        list or None: The collected results.

    Examples:
        >>> # distributed environment
        >>> # We have 2 process groups, 2 ranks.
        >>> import mmengine.dist as dist
        >>> if dist.get_rank() == 0:
                data = ['foo', {1: 2}]
            else:
                data = [24, {'a': 'b'}]
        >>> size = 4
        >>> output = dist.collect_results_gpu(data, size)
        >>> output
        ['foo', 24, {1: 2}, {'a': 'b'}]  # rank 0
        None  # rank 1
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]

    # gather all result part. Note that NCCL does not support gather so use
    # all_gather_object instead.
    part_list = all_gather_object(result_part)

    if rank == 0:
        # >>>>>>>>>>>>>>>>>>>>>
        # sort the results
        ordered_results = []
        for i in range(world_size):
            # the dataloader may pad some samples
            ordered_results.extend(list(part_list[i])[:ori_rank_samples[i]])
        # <<<<<<<<<<<<<<<<<<<<<
        return ordered_results
    else:
        return None
