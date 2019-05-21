from math import sqrt as sqrt
from itertools import product as product
import numpy as np

"""config示例
voc = {
    'num_classes': 21,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    's_min': 0.2,
    's_max': 0.95,
    'aspect_ratios': [1, 2, 3, 0.5, 0.3333333],
    'variance': [0.1, 0.2],
}"""


def generate_prior_box(config_file):
    """按照论文中的方法得到anchor，
    ssd.pytorch中的获得方式为38*38*4 + 19*19*6+ 10*10*6+ 5*5*6+ 3*3*4+ 1*1*4，有所不同"""
    fea_maps = config_file['feature_maps']
    m = len(fea_maps)
    s_min = config_file['s_min']
    s_max = config_file['s_max']
    s = []
    for k, fea_size in enumerate(fea_maps):
        s.append(s_min + (s_max - s_min) * (k-1) / (m-1))
    s.append(1)
    aspect_ratios = config_file['aspect_ratios']

    anchors = []
    for (k, fea_size) in enumerate(fea_maps):
        for i, j in product(range(fea_size), repeat=2):
            # 按着图像的坐标系来，j变化->x变化, i变化->y变化
            # 最终，anchor是横向扫描
            cx = (j + 0.5) / fea_size
            cy = (i + 0.5) / fea_size
            s_k_prime = sqrt(s[k] * s[k+1])

            # 对于38*38的anchor，特殊考虑
            if k == 0:
                # 只取3个ar，1个scale
                ar = [1, 2, 0.5]
                scale = 0.1
                for a in ar:
                    w = s[k] * sqrt(a)
                    h = s[k] / sqrt(a)
                    anchors.append([cx, cy, w, h])
                continue
            # add one anchor for ratio 1
            anchors.append([cx, cy, s_k_prime, s_k_prime])

            # add other anchors in different ratios
            for a in aspect_ratios:
                w = s[k] * sqrt(a)
                h = s[k] / sqrt(a)
                anchors.append([cx, cy, w, h])
    anchors = np.array(anchors)
    anchors = np.concatenate([(anchors[:, :2] - anchors[:, 2:] / 2),
                             (anchors[:, :2] + anchors[:, 2:] / 2)], axis=1)

    [W, H] = config_file['img_size']
    anchors = anchors * [W, H, W, H]
    return anchors

if __name__ == '__main__':
    import ssd_tool.config as config
    cfg = config.test
    prior = generate_prior_box(cfg)
    W, H = cfg['img_size']
    DRAW = False
    import test_tool
    if DRAW:
        for (i, a) in enumerate(prior):
            test_tool.draw_b_box(prior[i:i+1, :], W, H, wait_time=1)
            # test_tool.draw_b_box(prior[i:-1:1000, :], W, H, wait_time=1) 矩形雨
            print(a)

    print(len(prior))
    a = 0

