_base_ = [
    '../_base_/models/second_hv_secfpn_l-radset-long-range.py',
    '../_base_/datasets/l-radset-long.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

point_cloud_range = [0, -40.2, -4, 150.4, 40.2, 2]

train_dataloader = dict(
    batch_size=4,
    num_workers=6)

train_cfg = dict(max_epochs=40, val_interval=2)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=2))

resume = True
resume_from = 'work_dirs/second_hv_secfpn_8xb6-80e_radset-long-range/epoch_18.pth'