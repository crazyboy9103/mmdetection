_base_ = [
    '../_base_/datasets/border.py',
    '../_base_/models/nas-fcos_r50-caffe_fpn_fcoshead-gn-head.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head = dict(
        num_classes = 1
    )
)

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(project='neurocle', tags=['det', 'nas_fcos', 'border']),)
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

