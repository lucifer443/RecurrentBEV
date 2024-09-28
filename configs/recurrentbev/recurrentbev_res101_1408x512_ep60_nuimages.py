_base_ = ['./recurrentbev_res101_1408x512_ep90.py']

model = dict(
    img_backbone=dict(
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/cascade_mask_rcnn_r101_fpn_1x_nuim_20201024_134804-45215b1e.pth',
            prefix='backbone.')))

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=60,
    val_interval=10)
