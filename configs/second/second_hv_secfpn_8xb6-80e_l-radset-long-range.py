_base_ = [
    '../_base_/models/second_hv_secfpn_radset-long-range.py',
    '../_base_/datasets/radset.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -40.2, -4, 150.4, 40.2, 2]

train_dataloader = dict(
    batch_size=4,
    num_workers=6)

train_cfg = dict(max_epochs=48, val_interval=2)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))