# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class MnMDataset(CocoDataset):
    """"""

    METAINFO = {
        'classes':
        ('red', 'orange', 'yellow', 'green', 'blue', 'black'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(255, 0, 0), (255, 165, 0), (255, 255, 0), (0, 128, 0), (0, 0, 255), (0, 0, 0)]
    }