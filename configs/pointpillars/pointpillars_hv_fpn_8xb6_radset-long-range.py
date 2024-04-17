_base_ = [
    '../_base_/models/pointpillars_hv_fpn_radset-long-range.py',
    '../_base_/datasets/radset.py', 
    '../_base_/schedules/schedule-2x.py',
    '../_base_/default_runtime.py'
]

# For nuScenes dataset, we usually evaluate the model at the end of training.
# Since the models are trained by 24 epochs by default, we set evaluation
# interval to be 24. Please change the interval accordingly if you do not
# use a default schedule.
# train_dataloader = dict(
#     sampler=dict(type='DefaultSampler', shuffle=False))
point_cloud_range = [0, -40.2, -4, 150.4, 40.2, 2]
lr = 0.001
epoch_num = 48
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    # max_norm=10 is better for SECOND
    clip_grad=dict(max_norm=35, norm_type=2))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=48, val_interval=2)
train_dataloader = dict(
    batch_size=4,
    num_workers=6,)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))

# resume = True
# resume_from = 'work_dirs/pointpillars_hv_fpn_8xb6_mrdset/epoch_12.pth'