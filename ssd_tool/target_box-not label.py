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
    """
    input :
        index_file : 'D:/DataBackup/VOC2012/VOC2012_index.json'
        iou_threshold : matched iou threshold
    output: matched anchors for every image, formed as a dict, key is img_id
    {'2007_000212': [[x1, y1, x2, y2, label], ...],
     '2007_000214': [[x1, y1, x2, y2, label], ...],
     ... }"""
    root = '../targets'
    if not os.path.exists(root):
        os.mkdir(root)

    prior_boxes = prior_box_tool.generate_prior_box(config_file)  # return array
    prior_boxes = torch.Tensor(prior_boxes)
    with open(index_file, 'r') as f:
        content = json.load(f)
    output = {}
    for ann_info in content:
        img_id = ann_info['img_id']
        # output[img_id] = []
        with open(root + img_id + '.json', 'wt+') as f:
            temp = {img_id:[]}
            for cat in range(config_file['num_classes'] - 1):
                truths = []
                for det in ann_info['dets']:
                    [bbox, label] = det
                    if label == cat:
                        truths.append(bbox)

                if len(truths) == 0:
                    continue
                matched_idx = match_tool.match(truths, prior_boxes,iou_threshold).numpy()  # return tensor
                matched_prior = prior_boxes[matched_idx]
                a = matched_prior.shape[0]
                labels = torch.ones((matched_prior.shape[0], 1)) * cat
                matched_result = torch.cat([matched_prior, labels], dim=1)
                # output[img_id] += matched_result.tolist()
                temp[img_id] += matched_result.tolist()
            print('img: %s Done' % img_id)
            json.dump(temp, f)
    return output


if __name__ == '__main__':
    import ssd_tool.config as config
    cfg = config.test
    index_file_path = 'D:/DataBackup/VOC2012/VOC2012_index.json'
    output = generate_target_box(cfg, index_file_path, 0.5)
    a = 0