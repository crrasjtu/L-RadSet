_base_ = ['./detr3d_r101_gridmask_m2dset.py']

custom_imports = dict(imports=['projects.DETR3D.detr3d'])
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40.2, -40.2, -4.0, 70.2, 40.2, 2.0]
voxel_size = [0.2, 0.2, 6]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], bgr_to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
	'car', 'bus', 'truck', 'motorbike', 'bicycle', 'person', 'child', 
	'barrier', 'trafficcone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True)

dataset_type = 'M2DSetDataset'
data_root = 'data/m2dset/'

test_transforms = [
    dict(
        type='RandomResize3D',
        scale=(1920, 1080),
        ratio_range=(0.5, 0.5),
        keep_ratio=True)
]
train_transforms = [dict(type='PhotoMetricDistortion3D')] + test_transforms

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, num_views=6),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='MultiViewWrapper', transforms=train_transforms),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

metainfo = dict(classes=class_names)
data_prefix=dict(
            pts='',
            CAM_BACK_LEFT='images/image_0',
            CAM_FRONT_30='images/image_1',
            CAM_FRONT_60='images/image_2',
            CAM_FRONT_120='images/image_3',
            CAM_BACK_RIGHT='images/image_4',
            CAM_FRONT_RIGHT='images/image_5',
            CAM_BACK_120='images/image_6',
            CAM_FRONT_LEFT='images/image_7')

train_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='m2dset_infos_train.pkl',
            pipeline=train_pipeline,
            load_type='frame_based',
            metainfo=metainfo,
            modality=input_modality,
            test_mode=False,
            data_prefix=data_prefix,
            # we use box_type_3d='LiDAR' in m2dset dataset
            box_type_3d='LiDAR')))
