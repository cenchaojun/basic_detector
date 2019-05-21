import json
import pickle
import numpy as np
import cv2
# 这里的bounding box统一为：(x1,y1, x2,y2)，即左上角，右下角的形式

def generate_normal(means, covs, N):
    """means : 均值列表
    # covs : 协方差列表
    # N : 生成个数
    # return (x, y), label"""
    data = []
    M = len(means)
    Mode_n = np.ones(M, dtype=np.int) * int(N / M)
    if len(Mode_n) != 1:
        Mode_n[-1] = N - Mode_n[-1]
    # m为类别标签，n为每个类别的数量
    for [m, n] in enumerate(Mode_n):
        if len(means[m]) == 1:
            sample = np.random.normal(means[m][0], covs[m][0], n).reshape([n, 1])
        else:
            sample = np.random.multivariate_normal(means[m], covs[m], n)
        data += np.hstack((sample, np.zeros([n, 1]) + m)).tolist()
    return data


def generate_b_box(means, covs, size_means, size_covs, N, W, H):
    """W : 窗口width   H : 窗口height   means: 中心点期望，  size_means：bbox尺寸期望
    return 左上角，右下角, label（所属中心）"""
    # 生成中心点
    x_y_label = np.array(generate_normal(means, covs, N), dtype=np.int)
    label = x_y_label[:, 2:3]
    x_y = x_y_label[:, 0:2]
    x_y = np.maximum(np.minimum(x_y, [W, H]), 0)
    # 生成窗子的大小
    w_h = np.array(generate_normal(size_means, size_covs, N), dtype=np.int)[:, 0:2]
    w_h = np.maximum(np.minimum(w_h, np.array([W, H]) - x_y), 0)
    return np.hstack((x_y, x_y + w_h, label))

def generate_dets_rand_score(bboxes):
    """使用均匀分布生成score
    return 左上角，右下角，score"""
    score = np.random.rand(bboxes.shape[0], 1)
    return np.hstack((bboxes, score))


def generate_dets_L2_score(means, covs, size_means, size_covs, N, W, H):
    """使用bbox距离类别中心的距离生成score
    return 左上角，右下角，score"""
    # 获得bbox和标签
    bbox_label = generate_b_box(means, covs, size_means, size_covs, N, W, H)
    labels = bbox_label[:, 4]
    bboxes = bbox_label[:, 0:4]
    max_label = np.max(labels)
    # 获得bbox中心
    centers = (bboxes[:, 2: 4] + bboxes[:, 0: 2]) / 2
    scores = []
    # 按照每一类别计算分数
    for label in range(max_label + 1):
        center = centers[labels == label]
        # 使用L2 范数
        score = np.linalg.norm(center - means[label], axis=1).tolist()
        scores.extend(score)
    # 归一化到[0, 1]
    scores = np.array(scores).reshape(bboxes.shape[0], 1)
    scores = scores / np.max(scores)
    return np.hstack((bboxes, scores))


def draw_b_box(bboxes, W, H, wait_time=1000, color=(255, 0, 0)):
    """绘制bbox"""
    bboxes = bboxes.astype(np.int)
    img = np.zeros([H, W, 3], dtype=np.uint8)
    img[:, :, 0] = np.ones([H, W]) * 0
    img[:, :, 1] = np.ones([H, W]) * 100
    img[:, :, 2] = np.ones([H, W]) * 200
    for b in bboxes:
        [x1, y1, x2, y2] = b
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    cv2.imshow('b boxes', img)
    cv2.waitKey(wait_time)

def NMS(dets, IOUthresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 获得每个区域的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # score降序排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 获得相交区域，order[1:]为剩余的区域
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交区域的w和h
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        # 计算IOU
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        # inds为第一个没有被剔除的
        inds = np.where(ovr <= IOUthresh)[0]
        order = order[inds + 1]

    return keep

if __name__ == '__main__':
    means = [[100, 100], [200, 300]]
    covs = [[[10, 0], [0, 10]], [[10, 0], [0, 10]]]
    size_means = [[50, 100], [100, 100]]
    size_covs = [[[200, 0], [0, 200]], [[100, 0], [0, 100]]]
    W, H = 500, 500
    thresh = 0.1
    # A = generate_normal(means, covs, 100)
    # 生成bbox
    GenerateData = True
    b_box_test_file = './b_box_test_file.pkl'
    if GenerateData:
        D = generate_dets_L2_score(means, covs, size_means, size_covs, 100, W, H)
        with open(b_box_test_file, 'wb+') as f:
            pickle.dump(D, f)
    else:
        with open(b_box_test_file, 'rb') as f:
            D = pickle.load(f)
    # 绘制bbox
    B = D[:, 0:4]
    draw_b_box(B, W, H)
    keep_index = NMS(D, IOUthresh=thresh)
    D = B[keep_index, :]
    draw_b_box(B[keep_index, :], W, H, color=(255, 100, 100))
