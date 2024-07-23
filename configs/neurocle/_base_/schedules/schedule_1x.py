max_epochs=30
# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# learning rate
param_scheduler = [
    # dict(
    #     type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 16, 24],
        gamma=0.8),
    # dict(
    #     type='ReduceOnPlateauLR',
    #     monitor='coco/bbox_mAP',
    #     rule='greater',
    #     factor=0.8,
    #     patience=0,
    #     min_value=1e-5
    # ),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
