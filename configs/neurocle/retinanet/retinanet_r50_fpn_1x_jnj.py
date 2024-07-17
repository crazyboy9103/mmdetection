_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/jnj.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    # './retinanet_tta.py'
]

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(project='neurocle', tags=['det', 'retinanet', 'jnj']),)
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    bbox_head=dict(
        num_classes=23
    )
)