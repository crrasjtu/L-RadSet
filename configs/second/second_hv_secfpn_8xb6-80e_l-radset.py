_base_ = [
    '../_base_/models/second_hv_secfpn_l-radset.py',
    '../_base_/datasets/l-radset.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -40.2, -4, 70.2, 40.2, 2]

train_dataloader = dict(
    batch_size=8,
    num_workers=16)

train_cfg = dict(max_epochs=48, val_interval=4)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))