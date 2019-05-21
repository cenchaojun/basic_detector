# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import numpy as np
import json
import pickle
# 在此文件中，bbox的格式均为x1, x2, y1, y2
def make_GT_dict(BBGT, cat):
    """根据BBGT生成评估用的GT
    BBGT的格式:
        '[{\'file_path\' : ..., \', img_id(str)\': ..., ' \
        '\'dets\': [[b-box(x1, x2, y1, y2 4 int)], label(integral)], ...], ...]'
        输出 : GT
        GT 的格式为:
        { 'n_true': 个数， 'img_id': [[x1, y1, x2, y2, det], ...]], ...}"""
    GT = {}
    # GT :
    # { 'n_true': 个数， 'img_id': [[x1, y1, x2, y2, det], ...]], ...}
    ntrue = 0
    for gt in BBGT:
        GT[gt['img_id']] = []
        dets = gt['dets']
        for det in dets:
            if det[1] == cat:
                ntrue += 1
                GT[gt['img_id']].append(det[0] + [0])
    GT['n_true'] = ntrue
    return GT

def make_DT_dict(BBDT, img_index, cat):
    """根据BBDT创建预测用的DT
    index列表，指示每个bbox的img_id
    # [2001, 2001, 2002, 2002, 2002, ...]
    # BBDT的格式
    # [[x1, y1, x2, y2, confidence, category], ...]
    # 输出：
    # [[x1, y1, x2, y2, confidence], ...]，筛选掉不属于cat的bbox"""
    BBDT = np.array(BBDT)
    img_index = np.array(img_index)
    ids = BBDT[:, -1].astype(int) == cat

    return [BBDT[ids][:, 0:5], img_index[ids]]

def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def eval_ap(BBGT, BBDT, bbdt_index, IoUthresh, cat, use_07_metric=False):
    """计算类别为cat的ap，
    输入：BBGT和BBDT
    BBGT可以直接通过加载VOC2012_index.json来获得
    BBDT的格式为：
    [[x1, y1, x2, y2, confidence, category], ...]
    bbdt_index 表示BBDT里元素对应的img_index
    cat 为类别
    输出：recall数组，是递增数列,
          precision数组,表示精度
          ap，表示平均精度"""
    ntrue = 0
    GT = make_GT_dict(BBGT, cat)
    [DT, img_ids] = make_DT_dict(BBDT, bbdt_index, cat)

    # confidence 降序排列
    confidence = DT[:, -1]
    sorted_idx = np.argsort(-confidence)

    # 重新排列DT
    img_ids = img_ids[sorted_idx]
    DT = DT[sorted_idx]
    nDT = len(img_ids)   # 所有的预测检测框数量
    tp = np.zeros(nDT)   # true positive, 1 : tp, 0 : fp
    fp = np.zeros(nDT)

    for d in range(nDT):
        dt = DT[d]
        bb = dt[0:4]
        gtbb = np.array(GT[img_ids[d]])
        if gtbb.size != 0:
            ixmin = np.maximum(gtbb[:, 0], bb[0])
            iymin = np.maximum(gtbb[:, 1], bb[1])
            ixmax = np.minimum(gtbb[:, 2], bb[2])
            iymax = np.minimum(gtbb[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (gtbb[:, 2] - gtbb[:, 0] + 1.) *
                   (gtbb[:, 3] - gtbb[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            max_idx = np.argmax(overlaps)
            if ovmax > IoUthresh:
                # if not R['difficult'][jmax]:
                if not gtbb[max_idx, 4]:
                    tp[d] = 1.
                    gtbb[max_idx, 4] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.
    # cumsum：累加，[1,2,3] -> cumsum -> [1, 3, 6]
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec_array = tp / float(GT['n_true'])
    prec_array = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec_array, prec_array, use_07_metric)
    return rec_array, prec_array, ap

def eval_mAp(BBGT, BBDT, bbdt_index, IoUthresh, use_07_metric=False):
    """eval 20 category mAp
    输入：BBGT和BBDT
    BBGT可以直接通过加载VOC2012_index.json来获得
    BBDT的格式为：
    [[x1, y1, x2, y2, confidence, category], ...]
    bbdt_index 表示BBDT里每一个元素对应的img_index是哪个
    cat 为类别
    输出：recall数组，是递增数列,
          precision数组,表示精度
          ap，表示平均精度"""
    mAp = 0
    for c in range(20):
        [rec, pre, ap] = eval_ap(BBGT, BBDT, bbdt_index, IoUthresh, c, use_07_metric=use_07_metric)
        mAp += ap
    return mAp / 20

def make_fake_BBDT(index_file, cat, drop_number=10):
    """通过VOC2012_index.json创建类别为cat的bbox，用于测试
    index的格式
    # [2007_0002001, 2007_0002008,  ...]
    # BBDT的格式
    # [[x1, y1, x2, y2, confidence, category], ...]"""
    with open(index_file, 'r') as f:
        rawBBDT = json.load(f)
        print('load index %s' % index_file)
    index = []
    BBDT = []
    confidence = 0.9
    for info in rawBBDT:
        for det in info['dets']:
            if det[-1] == cat:
                BBDT.append(det[0] + [confidence] + [cat])
                index.append(info['img_id'])

    # random drop
    drop_rate = min(1.0, float(drop_number) / len(BBDT))
    remain_idx = np.random.rand(len(BBDT)) > drop_rate
    BBDT = np.array(BBDT)[remain_idx]
    index = np.array(index)[remain_idx]

    return BBDT, index

if __name__ == '__main__':
    index_file_path = 'D:/DataBackup/VOC2012/VOC2012_index.json'
    # 创建fake example
    with open(index_file_path,'r') as f:
        BBGT = json.load(f)
        # GT = make_GT_dict(BBGT, 0)
        [BBDT, index] = make_fake_BBDT(index_file_path, 0, drop_number=100)
    # 评估
    [rec_array, prec_array, Ap] = eval_ap(BBGT, BBDT, index, 0.9, 0)
    print('sum pre: %d, len pre: %d' % (sum(prec_array.tolist()), len(prec_array)))
    print('rec array', rec_array.tolist())
    print('Ap: %lf' % Ap)
    mAp = eval_mAp(BBGT, BBDT, index, 0.8)
    print('mAp: %lf' % mAp)
    a = 0




