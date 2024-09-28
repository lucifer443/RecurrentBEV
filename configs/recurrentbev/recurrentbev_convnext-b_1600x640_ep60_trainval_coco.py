_base_ = ['./recurrentbev_v2-99_1600x640_ep60_test.py']

checkpoint_file = 'checkpoints/convnext-b_coco.pth'

model = dict(
    data_preprocessor=dict(
        type='BEVDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='base',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.7,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        with_cp=True,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    img_neck=dict(in_channels=[128, 256, 512, 1024]),
    img_bbox_neck=dict(in_channels=[128, 256, 512, 1024],),)
