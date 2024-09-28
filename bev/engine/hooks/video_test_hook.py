import functools
import types

from mmengine.hooks import Hook
from mmengine.logging import MMLogger, print_log

from bev.datasets import SceneContinuousSampler
from bev.evaluation.metrics.utils import video_test_evaluate
from bev.registry import HOOKS


@HOOKS.register_module()
class VideoTestHook(Hook):
    def before_val(self, runner) -> None:
        logger: MMLogger = MMLogger.get_current_instance()
        if runner.val_loop is not None:
            sampler = runner.val_loop.dataloader.sampler
            assert isinstance(sampler, SceneContinuousSampler)

            evaluate = functools.partial(video_test_evaluate, ori_rank_samples=sampler.ori_rank_samples)
            for metric in runner.val_loop.evaluator.metrics:
                metric.evaluate = types.MethodType(evaluate, metric)

                print_log(f'Set rank samples to val metrics {metric.__class__.__name__}, '
                          f'ori_rank_samples: {sampler.ori_rank_samples}.', logger=logger)

    def before_test(self, runner) -> None:
        logger: MMLogger = MMLogger.get_current_instance()
        if runner.test_loop is not None:
            sampler = runner.test_loop.dataloader.sampler
            assert isinstance(sampler, SceneContinuousSampler)

            evaluate = functools.partial(video_test_evaluate, ori_rank_samples=sampler.ori_rank_samples)
            for metric in runner.test_loop.evaluator.metrics:
                metric.evaluate = types.MethodType(evaluate, metric)

                print_log(f'Set rank samples to test metrics {metric.__class__.__name__}, '
                          f'ori_rank_samples: {sampler.ori_rank_samples}.', logger=logger)
