_base_ = [
    '../_base_/datasets/border.py', 
    '../_base_/models/detr_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]

model = dict(
    bbox_head = dict(
        num_classes = 1
    )
)
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(project='neurocle', tags=['det', 'detr', 'border']),)
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')