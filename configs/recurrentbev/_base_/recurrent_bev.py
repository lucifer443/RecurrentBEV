# Global
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# Following order is related to BEVDetHead (CenterHead)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
metainfo = dict(classes=class_names)

# Model
bev_xbound = [-51.2, 51.2, 0.8]
bev_ybound = [-51.2, 51.2, 0.8]
bev_zbound = [-5.0, 3.0, 8.0]
depth_bound = [0.5, 59.5]
depth_bins = 59
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
C_img = 80

model = dict(
    type='RecurrentBEV',
    data_preprocessor=dict(
        type='BEVDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet3d.SECONDFPN',
        norm_cfg=dict(type='SyncBN', momentum=0.01),
        in_channels=[256, 512, 1024, 2048],
        out_channels=[64, 64, 128, 256],
        upsample_strides=[0.25, 0.5, 1, 2]),
    view_transformer=dict(
        type='ViewTransformer',
        shared_encoder=dict(
            type='ConvModule',
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU')),
        img_encoder=dict(
            type='ConvEncoder',
            in_channels=512,
            out_channels=C_img,
            kernel_size=1,
            padding=0,
            cam_aware=True,
            cam_channels=27),
        depth_encoder=dict(
            type='DepthNet',
            in_channels=512,
            mid_channels=256,
            out_channels=depth_bins,
            with_dcn=True,
            cam_aware=True,
            cam_channels=27),
        view_trans=dict(
            type='LiftSplatVT',
            grid_config=bev_grid_config,
            input_size=(256, 704),
            downsample=16,
            pool_method='bevfusion'),
        post_process=dict(
            inplanes=C_img,
            planes=C_img,
            num_blocks=2),
        loss_depth=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,
            elementwise=True,
            reduction='mean',
            loss_weight=1440.0)),
    bev_backbone=dict(
        type='CustomResNet',
        depth=18,
        with_stem=True,
        with_maxpool_in_stem=False,
        in_channels=C_img * 2,
        stem_channels=C_img * 2,
        base_channels=C_img * 2,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_input=False,
        out_indices=(0, 2),
        norm_eval=False),
    bev_neck=dict(
        type='Concat2FPN',
        in_channels=C_img * 8 + C_img * 2,
        out_channels=512,
        scale_factor=4,
        extra_upsample=True,
        extra_channels=256),
    bev_bbox3d_head=dict(
        type='SingleTaskBEVDetHead',
        in_channels=256,
        tasks=[
            dict(num_class=10, class_names=class_names),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=256,
        bbox_coder=dict(
            type='mmdet3d.CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=1000,
            score_threshold=0.05,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='mmdet3d.SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=6.),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=1.5),
        norm_bbox=True,
        record_separated_head_loss=True,
        scale_nms=True),
    temporal_fusion=dict(
        type='TransformAwareGRU',
        grid_config=bev_grid_config,
        transform_aware_config=dict(
            in_channels=C_img * 2,
            out_channels=C_img * 2,
            transform_channels=12,
            kernel_size=1,
            padding=0,),
        encoder_config=dict(curr_channels=C_img, hidden_channels=C_img*2, kernel_size=5),
        used_prev_frames=1,
        empty_padding='zero',
        prev_bev_channels=C_img*2),
    train_cfg=dict(
        bev_bbox3d=dict(
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        bev_bbox3d=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=None,
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.0,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            pre_max_size=None,
            post_max_size=500,
            # Scale-NMS
            nms_type=['rotate'],
            nms_thr=[0.2],
            nms_rescale_factor=[[1.0, 0.7, 0.7, 0.4, 0.55,
                                 1.1, 1.0, 1.0, 1.5, 3.5]])))

train_multiview_transforms = [
    dict(
        type='mmengine.RandomResize',
        scale=(1600, 900),
        ratio_range=(0.38, 0.55),
        keep_ratio=True,
        resize_type='mmdet.Resize',
        backend='pillow',
        interpolation='bicubic'),
    dict(
        type='RangeLimitedRandomCrop',
        crop_size=(704, 256),
        crop_type="absolute",
        relative_x_offset_range=(0.0, 1.0),
        relative_y_offset_range=(1.0, 1.0)),
    dict(type='mmdet.Pad', size=(704, 256)),
    dict(
        type='mmdet.RandomFlip',
        prob=0.5,
        direction='horizontal'),
    dict(
        type='mmdet.Rotate',
        prob=1.0,
        reversal_prob=0.5,
        level=None,
        min_mag=0.0,
        max_mag=5.4,
        img_border_value=0),
]

train_pipeline = [
    dict(
        type='LoadAdjacentFrames',
        num_prev_frames=0),
    dict(
        type='LoadMultiViewImageFromFiles',
        num_views=6,
        set_default_scale=False,
        imread_backend='turbojpeg',
        color_type='color'),
    dict(
        type='mmdet3d.LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='mmdet3d.LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(
        type='mmdet3d.MultiViewWrapper',
        process_fields=['img'],
        collected_keys=[
            'img_shape', 'pad_shape', 'homography_matrix'],
        override_aug_config=False,
        transforms=train_multiview_transforms),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        flip_box3d=True),
    dict(type='GetTransformationMatrices'),
    dict(
        type='ComputeDepthFromPoints',
        downsample=16,
        d_min=depth_bound[0],
        d_max=depth_bound[1],
        dbins=depth_bins,
        dmode=depth_discretize_mode,
        gt_type='cls',
        cls_mode='one_hot'),
    dict(
        type='mmdet3d.ObjectRangeFilter',
        point_cloud_range=point_cloud_range
        ),
    dict(
        type='mmdet3d.ObjectNameFilter',
        classes=class_names
        ),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'gt_bboxes_3d', 'gt_labels_3d',
            'gt_depth_dist', 'gt_depth_valid_mask',
            'cam2img', 'lidar2cam', 'lidar2img', 'lidar2global',
            'img2img_aug', 'lidar2lidar_aug', 'lidar_aug2img_aug'],
        meta_keys=[
            'img_shape', 'sample_idx', 'filename', 'num_views',
            'num_prev_frames', 'num_total_frames', 'first_in_scene']),
]

test_multiview_transforms = [
    dict(
        type='mmengine.RandomResize',
        scale=(1600, 900),
        ratio_range=(0.48, 0.48),
        keep_ratio=True,
        resize_type='mmdet.Resize',
        backend='pillow',
        interpolation='bicubic'),
    dict(
        type='RangeLimitedRandomCrop',
        crop_size=(704, 256),
        crop_type="absolute",
        relative_x_offset_range=(0.5, 0.5),
        relative_y_offset_range=(1.0, 1.0)),
]

test_pipeline = [
    dict(
        type='LoadAdjacentFrames',
        num_prev_frames=0),
    dict(
        type='LoadMultiViewImageFromFiles',
        num_views=6,
        set_default_scale=False,
        imread_backend='turbojpeg',
        color_type='color'),
    dict(
        type='mmdet3d.LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5),
    dict(
        type='mmdet3d.MultiViewWrapper',
        process_fields=['img'],
        collected_keys=[
            'img_shape', 'pad_shape', 'homography_matrix'],
        override_aug_config=False,
        transforms=test_multiview_transforms),
    dict(type='GetTransformationMatrices'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'img', 'cam2img', 'lidar2cam', 'lidar2img', 'img2img_aug',
            'lidar2global', 'lidar2lidar_aug', 'lidar_aug2img_aug'],
        meta_keys=[
            'img_shape', 'box_type_3d', 'num_views',
            'sample_idx', 'token', 'lidar2global',  # used in NuScenesMetric
            'first_in_scene', 'timestamp',  # used in VelocityBasedTracker
            'filename',  # used in BEVVisualizationHook
            'num_prev_frames', 'num_total_frames',
        ]),
]

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    sweeps='sweeps/LIDAR_TOP',
    CAM_FRONT='samples/CAM_FRONT',
    CAM_FRONT_LEFT='samples/CAM_FRONT_LEFT',
    CAM_FRONT_RIGHT='samples/CAM_FRONT_RIGHT',
    CAM_BACK='samples/CAM_BACK',
    CAM_BACK_RIGHT='samples/CAM_BACK_RIGHT',
    CAM_BACK_LEFT='samples/CAM_BACK_LEFT',
    occupancy_gts='occ_gts')
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        serialize_data=True,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='annotations/nuscenes_temporal_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=False,
        box_type_3d='LiDAR'))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='SceneContinuousSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        serialize_data=True,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='annotations/nuscenes_temporal_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        load_eval_anns=True,
        box_type_3d='LiDAR'))

test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='bev.NuScenesMetric',
        data_root=data_root,
        ann_file=data_root + 'annotations/nuscenes_temporal_infos_val.pkl',
        metric='bbox'),
]
test_evaluator = val_evaluator