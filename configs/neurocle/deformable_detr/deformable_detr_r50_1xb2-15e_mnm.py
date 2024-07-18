_base_ = [
    '../_base_/datasets/mnm.py', 
    '../_base_/models/deformable_detr_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]

model = dict(
    bbox_head = dict(
        num_classes = 6
    )
)

# optimizer
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            # 'sampling_offsets': dict(lr_mult=0.1),
            # 'reference_points': dict(lr_mult=0.1)
        }))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(project='neurocle', tags=['det', 'deformable_detr', 'mnm']),)
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

train_dataloader = dict(
    batch_size=8,
)
val_dataloader = dict(
    batch_size=8,
)