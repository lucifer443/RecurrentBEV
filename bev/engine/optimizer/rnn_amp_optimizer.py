from mmengine.registry import OPTIM_WRAPPERS
from mmengine.optim.optimizer import AmpOptimWrapper

from contextlib import contextmanager
import torch.nn as nn


@OPTIM_WRAPPERS.register_module()
class RNNAmpOptimWrapper(AmpOptimWrapper):

    @contextmanager
    def optim_context(self, model: nn.Module):
        """Enables the context for mixed precision training, and enables the
        context for disabling gradient synchronization during gradient
        accumulation context.

        Args:
            model (nn.Module): The training model.
        """
        from mmengine.runner.amp import autocast
        with super().optim_context(model), autocast(dtype=self.cast_dtype, cache_enabled=False):
            yield