
import os
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import pylab
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import cv2
data_root = 'D:/DataBackup/COCO'
data_type = 'val2014'
data_path = data_root + '/' + data_type
ann_path = data_root + '/annotations'
ann_name = 'instances_' + data_type
ann_file = ann_path + '/' + ann_name + '.json'

index_file = data_root + '/' + ann_name + '_index.json'
data_brief_file = data_root + '/' + ann_name + '_brief.json'

coco = COCO(ann_file)
cats = coco.loadCats(coco.getCatIds())

result_file = './instances_val2014_fakebbox100_results.json'
cocoDt = coco.loadRes(result_file)
cocoeval = COCOeval(coco, cocoDt, iouType='bbox')
cocoeval.evaluate()
cocoeval.accumulate()
cocoeval.summarize()
# print(cocoeval)

