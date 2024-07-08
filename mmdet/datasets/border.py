# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class BorderDataset(CocoDataset):
    """"""

    METAINFO = {
        'classes':
        ('class1'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60)]
    }