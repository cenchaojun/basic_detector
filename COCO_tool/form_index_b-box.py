
import os
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
import pylab
from pycocotools.coco import COCO
import cv2
data_root = 'D:/DataBackup/COCO'
data_type = 'train2014'
data_path = data_root + '/' + data_type
ann_path = data_root + '/annotations'
ann_name = 'instances_' + data_type
ann_file = ann_path + '/' + ann_name + '.json'

index_file = data_root + '/' + ann_name + '_index.json'
data_brief_file = data_root + '/' + ann_name + '_brief.json'

coco = COCO(ann_file)
cats = coco.loadCats(coco.getCatIds())
def Cat_to_map(cats):
    name_id_map = {}
    for c in cats:
        name_id_map[c['name']] = c['id']
    return name_id_map
name_id_map = Cat_to_map(cats)
id_name_map = {value: key for key, value in name_id_map.items()}

import json

with open(ann_file) as f:
    anns = json.load(f)

with open(index_file, 'wt+') as f:
    count = 0
    index_content = {}
    for img_info in anns['images']:
        if img_info['id'] in index_content.keys():
            raise Exception('Repeat id in img_info')
        file_name = img_info['file_name']
        file_path = data_path + '/' + file_name
        index_content[img_info['id']] = [file_path]
    for ann in anns['annotations']:
        bbox = ann['bbox']
        label = ann['category_id']
        if ann['image_id'] not in index_content.keys():
            raise Exception('id %d not in img_info' % ann['id'])
        index_content[ann['image_id']].append([bbox, label])
        count += 1
        print('%d Done!' % count)
    index_content = list(index_content.values())
    json.dump(index_content, f)
print('%s Saved!' % index_file)

with open(index_file, 'r') as f:
    index = json.load(f)
    a = 0

with open(data_brief_file, 'wt+') as f:
    brief = {}
    brief['data_path'] = data_path
    brief['ann_file'] = ann_file
    brief['index_file'] = index_file
    brief['total number'] = count
    brief['format of index file'] = '[[filename(str), [b-box(x, y, w, h, 4 float)], label(integral)], ...], ...]'
    brief['id2name'] = id_name_map
    json.dump(brief, f)

print('%s Saved!' % data_brief_file)
