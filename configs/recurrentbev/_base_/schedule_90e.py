lr = 2e-4
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 1000,
        end_factor=1.0,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='ConstantLR',
        factor=1.0,
        by_epoch=True,
        begin=0,
        end=90),
]

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=90,
    val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# base_batch_size should be checked carefully!
auto_scale_lr = dict(enable=True, base_batch_size=64)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='mmdet.ExpMomentumEMA',
        momentum=0.0001,
        gamma=2000,
        update_buffers=True),
    dict(type='bev.VideoTestHook'),
    dict(type='bev.TemporalWarmupHook', warmup_epochs=6)
]

randomness = dict(seed=0)
