voxel_size = [0.2, 0.2, 6]  # 根据您的数据集做调整
# point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]  # 根据您的数据集做调整
model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=32,
            point_cloud_range=[0, -40.2, -4, 70.2, 40.2, 2],
            voxel_size=voxel_size,
            max_voxels=(80000, 90000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=[-40.2, -40.2, -4, 70.2, 40.2, 2]),
    # `output_shape` 需要根据 `point_cloud_range` 和 `voxel_size` 做相应调整
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[402, 552]),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=3,
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        # 根据您的数据集调整 `ranges` 和 `sizes`
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
		        [-40.2, -40.2, -1.2082, 70.2, 40.2, -1.2082],
                [-40.2, -40.2, -0.2006, 70.2, 40.2, -0.2006],
                [-40.2, -40.2, -0.4842, 70.2, 40.2, -0.4842],
		        [-40.2, -40.2, -1.2745, 70.2, 40.2, -1.2745],
		        [-40.2, -40.2, -1.5867, 70.2, 40.2, -1.5867],
		        [-40.2, -40.2, -1.2295, 70.2, 40.2, -1.2295],
		        [-40.2, -40.2, -1.2082, 70.2, 40.2, -1.2082],
		        [-40.2, -40.2, -1.4977, 70.2, 40.2, -1.4977],
		        [-40.2, -40.2, -1.5552, 70.2, 40.2, -1.5552]
            ],
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
    # 模型训练和测试设置
    train_cfg=dict(
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
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))