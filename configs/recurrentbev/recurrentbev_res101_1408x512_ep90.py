_base_ = ['./recurrentbev_res50_704x256_ep90.py']

bev_xbound = [-51.2, 51.2, 0.4]
bev_ybound = [-51.2, 51.2, 0.4]
bev_zbound = [-5.0, 3.0, 8.0]
depth_bound = [1.0, 60.0]
depth_bins = 118
depth_discretize_mode='LID'

bev_grid_config=dict(
    xbound=bev_xbound,
    ybound=bev_ybound,
    zbound=bev_zbound,
    dbound=depth_bound,
    dbins=depth_bins,
    dmode=depth_discretize_mode)

# configs for CenterHead
voxel_size = [0.1, 0.1, 8.0]
grid_size = [
    int((bev_xbound[1] - bev_xbound[0]) / voxel_size[0]),
    int((bev_ybound[1] - bev_ybound[0]) / voxel_size[1]),
    1]
out_size_factor = int(bev_xbound[2] / voxel_size[1])

model = dict(
    img_backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint='torchvision://resnet101')
    ),
    temporal_fusion=dict(
        grid_config=bev_grid_config),
    view_transformer=dict(
        view_trans=dict(
            grid_config=bev_grid_config,
            input_size=(512, 1408),),
        depth_encoder=dict(
            _delete_=True,
            type='DepthNetOfficial',
            in_channels=512,
            mid_channels=512,
            out_channels=depth_bins,
            with_dcn=True,
            assp_mid_channels=96,
            cam_aware=True,
            cam_channels=27),
    ),
    bev_bbox3d_head=dict(
        bbox_coder=dict(
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2])),
    train_cfg=dict(
        bev_bbox3d=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor)),
    test_cfg=dict(
        bev_bbox3d=dict(
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2]))
)

_base_.train_multiview_transforms[0].update(
    dict(ratio_range=(0.82, 0.99))
)
_base_.train_multiview_transforms[1].update(
    dict(crop_size=(1408, 512))
)
_base_.train_multiview_transforms[2].update(
    dict(size=(1408, 512))
)

_base_.test_multiview_transforms[0].update(
    dict(ratio_range=(0.92, 0.92))
)
_base_.test_multiview_transforms[1].update(
    dict(crop_size=(1408, 512))
)

_base_.train_pipeline[4]['transforms'] = _base_.train_multiview_transforms
_base_.train_pipeline[8].update(
    dict(type='ComputeDepthFromPoints',
         d_min=depth_bound[0],
         d_max=depth_bound[1],
         dbins=depth_bins,
         dmode=depth_discretize_mode))
_base_.test_pipeline[3]['transforms'] = _base_.test_multiview_transforms

sync_bn='torch'

train_dataloader = dict(
    batch_size=4,
    dataset=dict(pipeline=_base_.train_pipeline)
)

test_dataloader = dict(
    dataset=dict(pipeline=_base_.test_pipeline)
)

val_dataloader = test_dataloader
