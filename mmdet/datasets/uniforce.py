# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class UniforceDataset(CocoDataset):
    """"""

    METAINFO = {
        'classes':
        ('TAG'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }