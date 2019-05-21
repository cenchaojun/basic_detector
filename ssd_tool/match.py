import torch


def compute_IoU(box_a, box_b):
    """摘抄于ssd.pytorch的box_utils
    input: box_a : A * 4        box_b : B * 4   (Tensor)
    4 : (xmin, ymin, xmax, ymax)
    output: A * B                               (Tensor)"""
    A = box_a.size(0)
    B = box_b.size(0)
    expand_a = box_a.unsqueeze(1).expand(A, B, 4)
    expand_b = box_b.unsqueeze(0).expand(A, B, 4)
    # 交集的(x1, y1, x2, y2)
    inter_min = torch.max(expand_a[:, :, 0:2], expand_b[:, :, 0:2])
    inter_max = torch.min(expand_a[:, :, 2:], expand_b[:, :, 2:])

    # clamp（加紧）使得所有元素大于0
    # inter_wh : A * B * 2
    inter_wh = torch.clamp((inter_max - inter_min), min=0)

    # inter_area : A * B
    inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]

    box_a_wh = box_a[:, 2:] - box_a[:, 0:2]
    box_b_wh = box_b[:, 2:] - box_b[:, 0:2]
    box_a_area = box_a_wh[:, 0] * box_a_wh[:, 1]  # A
    box_b_area = box_b_wh[:, 0] * box_b_wh[:, 1]  # B

    # union area A*B
    union_area = (box_a_area.unsqueeze(1).expand(A, B) +
                 box_b_area.unsqueeze(0).expand(A, B)) - inter_area

    # IoU A*B
    IoU = inter_area / union_area

    return IoU


def match(truths, labels, priors, IoUthreshold):
    """ match GT and priors, return matched index of priors
    input:
        Ground Truths : n_truth * 4   (Transform to Tensor(in code))
        labels: category of each GT bbox
        prior: n_prior * 4            (Transform to Tensor(in code))
    output:
        matched_idx : n_prior * 1     (Tensor)
            >0 : category, i.e. positive
            0 : background, i.e. negative"""
    truths = torch.Tensor(truths)
    priors = torch.Tensor(priors)
    iou: torch.Tensor = compute_IoU(truths, priors)  # iou : n_truth * n_prior
    # best_priors：每个GT匹配到的最优prior
    # best_truth：每个prior匹配到的最优ground truth
    best_priors_idx = iou.max(dim=1)[1]
    [best_truth_iou, best_truth_idx] = iou.max(dim=0)
    best_truth_idx = best_truth_idx.byte()

    # 置为2，保证不会被筛选掉
    matched_idx = torch.zeros_like(best_truth_idx).byte()
    for (i, label) in enumerate(labels):
        matched_idx[best_priors_idx[i]] = label + 1

    # 删除掉那些已经匹配的prior
    best_truth_iou[best_priors_idx] = 0

    # 在剩下的prior中选取IOU足够大的prior
    matched_truth_idx = best_truth_iou > IoUthreshold

    # 赋予匹配的gt标签
    # print(best_truth_idx[matched_truth_idx])
    labels = torch.Tensor(labels).byte()
    matched_idx[matched_truth_idx] = labels[best_truth_idx[matched_truth_idx].tolist()] + 1

    return matched_idx


if __name__ == '__main__':
    box_a = torch.Tensor([[0, 0, 10, 10],
                          [50, 50, 100, 100]])
    labels_a = torch.Tensor([0, 1]).byte()
    labels_b = torch.Tensor([0, 1, 2, 3]).byte()

    box_b = torch.Tensor([[0, 0, 5, 5],
                          [0, 0, 10, 10],
                          [60, 60, 110, 110],
                          [50, 50, 100, 100]])
    IOU_Truth = torch.Tensor([[0.25, 1, 0, 0],
                          [0, 0, 1600/(2500+2500-1600), 1]])
    truths = box_a
    priors = box_b
    IoUthreshold = 0.5

    print(compute_IoU(box_a, box_a))
    print(compute_IoU(box_a, box_b))
    print(compute_IoU(box_b, box_a))
    print(compute_IoU(box_b, box_b))

    print(match(box_a, labels_a, box_a, 0.5))
    print(match(box_a, labels_a, box_b, 0.5))
    print(match(box_b, labels_b, box_a, 0.5))
    print(match(box_b, labels_b, box_b, 0.5))
    idx = match(box_a, labels_a, box_b, 0.5)
    print(idx)
    print(box_b[idx])










