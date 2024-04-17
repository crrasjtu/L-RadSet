_base_ = [
    '../_base_/models/pointpillars_hv_fpn_l-radset.py',
    '../_base_/datasets/l-radset.py',
    '../_base_/schedules/cyclic-40e.py', 
    '../_base_/default_runtime.py'
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

train_cfg = dict(by_epoch=True, max_epochs=48, val_interval=2)