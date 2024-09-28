_base_ = ['./_base_/runtime.py',
          './_base_/recurrent_bev.py',
          './_base_/schedule_90e.py']

aux_2d_loss_scale = 10

model = dict(
    aux_2d_branch=dict(
        neck=dict(
            type='mmdet.FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',  # use P5
            num_outs=5,
            relu_before_extra_convs=True),
        head=dict(
            type='mmdet.FCOSHead',
            num_classes=10,
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            strides=[8, 16, 32, 64, 128],
            bbox_coder=dict(type='mmdet.DistancePointBBoxCoder'),
            loss_cls=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * aux_2d_loss_scale),
            loss_bbox=dict(type='mmdet.IoULoss', loss_weight=1.0 * aux_2d_loss_scale),
            loss_centerness=dict(
                type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 * aux_2d_loss_scale))))

_base_.train_pipeline[0]['num_prev_frames'] = 8

train_dataloader = dict(
    dataset=dict(
        pipeline=_base_.train_pipeline))
