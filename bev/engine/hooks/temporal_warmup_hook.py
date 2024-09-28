from mmengine.hooks import Hook
from bev.registry import HOOKS
from mmengine.model import is_model_wrapper


@HOOKS.register_module()
class TemporalWarmupHook(Hook):

    def __init__(self, warmup_epochs=2) -> None:
        self.warmup_epochs = warmup_epochs
        self._in_warmup = False
        self._restart_dataloader = False
        self.sequence_length = 0

    def before_train_epoch(self, runner) -> None:
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        epoch_to_be_switched = ((epoch + 1) <= self.warmup_epochs)

        model = runner.model
        # TODO: refactor after mmengine using model wrapper
        if is_model_wrapper(model):
            model = model.module

        dataset = train_loader.dataset.dataset if hasattr(train_loader.dataset, 'dataset') else train_loader.dataset

        if epoch_to_be_switched:
            if not self._in_warmup:
                runner.logger.info('Not use prev frames now!')

                model.bev_bbox3d_head.train_cfg.code_weights[-2:] = [0.2, 0.2]

                self.sequence_length = dataset.pipeline.transforms[0].num_prev_frames
                dataset.pipeline.transforms[0].num_prev_frames = 0

                # The dataset pipeline cannot be updated when persistent_workers
                # is True, so we need to force the dataloader's multi-process
                # restart. This is a very hacky approach.
                if hasattr(train_loader, 'persistent_workers'
                           ) and train_loader.persistent_workers is True:
                    train_loader._DataLoader__initialized = False
                    train_loader._iterator = None
                    self._restart_dataloader = True
                    self._in_warmup = True
            else:
                # Once the restart is complete, we need to restore
                # the initialization flag.
                if self._restart_dataloader:
                    train_loader._DataLoader__initialized = True

        elif self.warmup_epochs>0:
            if self._in_warmup:
                runner.logger.info('Temporal warmup finished!')

                model.bev_bbox3d_head.train_cfg.code_weights[-2:] = [1.0, 1.0]

                dataset.pipeline.transforms[0].num_prev_frames = self.sequence_length
                if hasattr(train_loader, 'persistent_workers'
                           ) and train_loader.persistent_workers is True:
                    train_loader._DataLoader__initialized = False
                    train_loader._iterator = None
                    self._restart_dataloader = True
                self._in_warmup = False
            else:
                if self._restart_dataloader:
                    train_loader._DataLoader__initialized = True




