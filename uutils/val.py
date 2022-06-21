"""useful utils for validating object detection"""
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchmetrics
import torchvision

from . import bbox as bb
from . import tools

def topn_acc(src, tgt, *, n=5):
    """src,tgt are only class predictions"""
    """cxn 1xn"""

    h, w = src.shape

    top = []
    for _ in range(n):
        t = src.argmax(-1).tolist()
        for j, xj in enumerate(t):
            src[j, xj] = 0
            top.append(t)

    top = torch.Tensor(top).mT

    acc = [int(y in x) for x, y in zip(top, tgt)]
    acc = sum(acc) / len(acc)
    return acc


def top1_acc(src, tgt):
    return topn_acc(src, tgt, n=1)


def top5_acc(src, tgt):
    return topn_acc(src, tgt, n=5)


def nms(*, preds=None, bboxes=None, logits=None, size=None, conf_thres=0.5, iou_thres=0.5):
    "(x1, y1, x2, y2, conf, cls)"

    if not (bboxes is None or logits is None):

        bboxes = bb.rescale(bboxes, size)
        prob = F.softmax(logits[..., :-1],dim=-1)
        c = prob.argmax(-1)[..., None]
        conf = prob.max(-1)[0][..., None]
        preds = torch.cat([bboxes, conf, c], dim=-1)
        preds = preds[conf.flatten() > conf_thres]

    result = []
    cls = set(preds[..., -1].flatten().tolist())
    for c in cls:

        isc = preds[..., -1] == c
        x = preds[isc]
        keep = torchvision.ops.nms(x[..., :4], x[..., 4], iou_thres)
        result += x[keep].reshape(-1, 6).tolist()
    result = torch.Tensor(result)
    return result
