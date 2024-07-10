_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/border.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    # './retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs=dict(project='neurocle', tags=['det', 'retinanet', 'border']),)
]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

model = dict(
    bbox_head=dict(
        num_classes=1
    )
)

load_from="https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
