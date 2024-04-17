_base_ = [
    '../_base_/datasets/l-radset.py',
    '../_base_/schedules/cyclic-40e.py', 
    '../_base_/default_runtime.py'
]

voxel_size = [0.05, 0.05, 0.1]
point_cloud_range = [0, -40, -4, 70.4, 40, 2]

data_root = 'data/m2dset/'
class_names = [
	'car', 'bus', 'truck', 'motorbike', 'bicycle', 'person', 'child', 
	'barrier', 'trafficcone'
]
metainfo = dict(CLASSES=class_names)
backend_args = None

input_modality = dict(use_lidar=True, use_camera=False)
data_prefix=dict(
            pts='lidar_reduced',)
            # CAM_BACK_LEFT='images/image_0',
            # CAM_FRONT_30='images/image_1',
            # CAM_FRONT_60='images/image_2',
            # CAM_FRONT_120='images/image_3',
            # CAM_BACK_RIGHT='images/image_4',
            # CAM_FRONT_RIGHT='images/image_5',
            # CAM_BACK_120='images/image_6',
            # CAM_FRONT_LEFT='images/image_7')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'm2dset_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            bus=5,
            truck=5,
            motorbike=4,
            bicycle=4,
            person=3,
            child=3,
            barrier=4,
            trafficcone=2)),
    classes=class_names,
    sample_groups=dict(
	    car=2,
            bus=4,
            truck=3,
            motorbike=6,
            bicycle=6,
            person=2,
            child=2,
            barrier=2,
            trafficcone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler, use_ground_plane=False),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
        dict(
            type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=6)

train_cfg = dict(max_epochs=48, val_interval=8)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))

model = dict(
    type='PointVoxelRCNN',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(80000, 90000))),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[61, 1600, 1408],
        order=('conv', 'norm', 'act'),
        encoder_paddings=((0, 0, 0), ((1, 1, 1), 0, 0), ((1, 1, 1), 0, 0),
                          ((0, 1, 1), 0, 0)),
        return_middle_feats=True),
    points_encoder=dict(
        type='VoxelSetAbstraction',
        num_keypoints=2048,
        fused_out_channel=128,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        voxel_sa_cfgs_list=[
            dict(
                type='StackedSAModuleMSG',
                in_channels=16,
                scale_factor=1,
                radius=(0.4, 0.8),
                sample_nums=(16, 16),
                mlp_channels=((16, 16), (16, 16)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=32,
                scale_factor=2,
                radius=(0.8, 1.2),
                sample_nums=(16, 32),
                mlp_channels=((32, 32), (32, 32)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=4,
                radius=(1.2, 2.4),
                sample_nums=(16, 32),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True),
            dict(
                type='StackedSAModuleMSG',
                in_channels=64,
                scale_factor=8,
                radius=(2.4, 4.8),
                sample_nums=(16, 32),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True)
        ],
        rawpoints_sa_cfgs=dict(
            type='StackedSAModuleMSG',
            in_channels=1,
            radius=(0.4, 0.8),
            sample_nums=(16, 16),
            mlp_channels=((16, 16), (16, 16)),
            use_xyz=True),
        bev_feat_channel=256,
        bev_scale_factor=8),
    backbone=dict(
        type='SECOND',
        in_channels=384,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    rpn_head=dict(
        type='PartA2RPNHead',
        num_classes=9,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        dir_offset=0.78539,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
             ranges=[
		        [0, -40.2, -1.2082, 70.4, 40.2, -1.2082],
                [0, -40.2, -0.2006, 70.4, 40.2, -0.2006],
                [0, -40.2, -0.4842, 70.4, 40.2, -0.4842],
		        [0, -40.2, -1.2745, 70.4, 40.2, -1.2745],
		        [0, -40.2, -1.5867, 70.4, 40.2, -1.5867],
		        [0, -40.2, -1.2295, 70.4, 40.2, -1.2295],
		        [0, -40.2, -1.2082, 70.4, 40.2, -1.2082],
		        [0, -40.2, -1.4977, 70.4, 40.2, -1.4977],
		        [0, -40.2, -1.5552, 70.4, 40.2, -1.5552]],
            sizes=[
		        [4.41, 1.93, 1.63], # car
		        [9.82, 2.83, 3.14], # bus
		        [7.82, 2.68, 3.01], # truck
		        [1.79, 0.73, 1.43], # motorbike
		        [1.62, 0.57, 1.15], # bicycle
		        [0.59, 0.62, 1.61], # person
		        [0.49,  0.47, 1.13], # child
		        [1.47, 1.05, 0.86], # barrier
		        [0.40, 0.39, 0.72] # traffic cone
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        assigner_per_size=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    roi_head=dict(
        type='PVRCNNRoiHead',
        num_classes=9,
        semantic_head=dict(
            type='ForegroundSegmentationHead',
            in_channels=768,
            extra_width=0.1,
            loss_seg=dict(
                type='mmdet.FocalLoss',
                use_sigmoid=True,
                reduction='sum',
                gamma=2.0,
                alpha=0.25,
                activated=True,
                loss_weight=1.0)),
        bbox_roi_extractor=dict(
            type='Batch3DRoIGridExtractor',
            grid_size=6,
            roi_layer=dict(
                type='StackedSAModuleMSG',
                in_channels=128,
                radius=(0.8, 1.6),
                sample_nums=(16, 16),
                mlp_channels=((64, 64), (64, 64)),
                use_xyz=True,
                pool_mod='max'),
        ),
        bbox_head=dict(
            type='PVRCNNBBoxHead',
            in_channels=128,
            grid_size=6,
            num_classes=9,
            class_agnostic=True,
            shared_fc_channels=(256, 256),
            reg_channels=(256, 256),
            cls_channels=(256, 256),
            dropout_ratio=0.3,
            with_corner_loss=True,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0 / 9.0,
                reduction='sum',
                loss_weight=1.0),
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=True,
                reduction='sum',
                loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=[
                dict(  # car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
		        dict(  # bus
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
		        dict(  # truck
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
		        dict(  # motorbike
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
		        dict(  # bicycle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # person
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
		        dict(  # child
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
		        dict(  # barrier
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # traffic cone
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.25,
                    neg_iou_thr=0.1,
                    min_pos_iou=0.1,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=9000,
            nms_post=512,
            max_num=512,
            nms_thr=0.8,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            assigner=[
                dict(  # car
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.55,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
		        dict(  # bus
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
		        dict(  # truck
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.4,
                    min_pos_iou=0.4,
                    ignore_iof_thr=-1),
		        dict(  # motorbike
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
		        dict(  # bicycle
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # person
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
		        dict(  # child
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
		        dict(  # barrier
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # traffic cone
                    type='Max3DIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.25,
                    neg_iou_thr=0.1,
                    min_pos_iou=0.1,
                    ignore_iof_thr=-1),
            ],
            sampler=dict(
                type='IoUNegPiecewiseSampler',
                num=128,
                pos_fraction=0.5,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True),
            cls_pos_thr=0.75,
            cls_neg_thr=0.25)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1024,
            nms_post=100,
            max_num=100,
            nms_thr=0.7,
            score_thr=0,
            use_rotate_nms=True),
        rcnn=dict(
            use_rotate_nms=True,
            use_raw_score=True,
            nms_thr=0.1,
            score_thr=0.1)))

lr = 0.001
optim_wrapper = dict(optimizer=dict(lr=lr))
param_scheduler = [
    # learning rate scheduler
    # During the first 16 epochs, learning rate increases from 0 to lr * 10
    # during the next 24 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=lr * 10,
        begin=0,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=25,
        eta_min=lr * 1e-4,
        begin=15,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 16 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 24 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=15,
        eta_min=0.85 / 0.95,
        begin=0,
        end=15,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=25,
        eta_min=1,
        begin=15,
        end=40,
        by_epoch=True,
        convert_to_iter_based=True)
]

resume = True
resume_from = 'work_dirs/pv_rcnn_8xb2-80e_m2dset-3d/epoch_16.pth'