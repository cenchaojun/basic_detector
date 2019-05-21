import ssd_tool.prior_box as prior_box_tool
import ssd_tool.match as match_tool
import torch
import numpy as np
import os

import json
"""index_file中的格式:
    '[{\'file_path\' : ..., \', img_id(str)\': ..., ' \
    '\'dets\': [[b-box(x1, x2, y1, y2 4 int)], label(integral)], ...], ...]'
    { 'n_true': 个数， 'img_id': [[x1, y1, x2, y2, det], ...]], ...}"""
def generate_target_box(config_file, index_file, iou_threshold):
    """生成目标框，并存储到targets文件夹中，
        2012_004310.json：目标的label：0表示背景，非0表示原始的标签+1
        priors.json：先验窗信息
    input :
        index_file : 'D:/DataBackup/VOC2012/VOC2012_index.json'
        iou_threshold : matched iou threshold
    output:
        2012_004310.json：目标的label：0表示背景，非0表示原始的标签+1
        priors.json：先验窗信息"""

    root = '../targets'
    if not os.path.exists(root):
        os.mkdir(root)

    # 存储先验窗
    prior_boxes = prior_box_tool.generate_prior_box(config_file)  # return array
    with open(root + '/' + 'priors.json', 'wt+') as f:
        json.dump(prior_boxes.tolist(), f)

    # 加载index文件
    prior_boxes = torch.Tensor(prior_boxes)
    with open(index_file, 'r') as f:
        content = json.load(f)

    # 对于每张图片的index文件
    for ann_info in content:
        img_id = ann_info['img_id']
        # 存储label文件
        with open(root + '/' + img_id + '.json', 'wt+') as f:
            temp = {img_id: []}
            truths = []
            labels = []
            for det in ann_info['dets']:
                [bbox, label] = det
                truths.append(bbox)
                labels.append(label)
            matched_idx = match_tool.match(truths, labels,
                                           prior_boxes, iou_threshold)
            temp[img_id] += matched_idx.tolist()
            print('img: %s Done！' % img_id)
            json.dump(matched_idx.tolist(), f)


def load_target(target_folder):
    prior_file_path = target_folder + '/priors.json'
    output = {}
    idx_list = os.listdir(target_folder)
    count = 0
    for idx_file_name in idx_list:
        idx_file_path = target_folder + '/' + idx_file_name
        [img_id, ext_json_] = os.path.splitext(idx_file_name)
        with open(idx_file_path, 'r') as f:
            output[img_id] = json.load(f)
        count += 1
        print(count)
    for o in output.items():
        print(o)

if __name__ == '__main__':
    import ssd_tool.config as config
    cfg = config.test
    index_file_path = 'D:/DataBackup/VOC2012/VOC2012_index.json'
    target_folder = '../targets'
    GENERATE = True
    if GENERATE:
        generate_target_box(cfg, index_file_path, 0.5)
        a = 0
    else:
        load_target(target_folder)



