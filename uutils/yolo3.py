import torch
import torchmetrics
import torchvision

import uutils as UU


def nms(preds, *, bboxes=None, logits=None, conf_thres=0.25, iou_thres=0.5):
   '(x1, y1, x2, y2, conf, cls)'

    if not (bboxes is None or logits is None):

        prob = logits[...,:-1]
        c = prob.argmax(-1)
        conf = logits.max(-1)[0]

        preds = torch.cat([bboxes,conf,c],dim=-1) 

    result = []
    cls = set(preds[...,-1].tolist())
    for c in cls:

        isc = preds[...,-1] == c
        x = preds[isc].tolist()
        keep = torchvision.ops.nms(preds[...,:4],preds[...,4],iou_thres)
        result += preds[keep].tolist()

    return result


