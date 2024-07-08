# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import DATASETS
from .coco import CocoDataset

@DATASETS.register_module()
class JnjDataset(CocoDataset):
    """"""

    METAINFO = {
        'classes':
        ('23&24', '22', '21', '20', '19', '16', '15', '14', '13', '12', 
         '11', '10', '9', '8', '7', '6', '5', '4', '3', '2', '1', '17', '18'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0)]
    }