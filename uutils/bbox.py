"""useful utils for bboxes"""
"borrowed some functions from d2l.ai"

import os
import time
import tkinter

from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision
from torchvision import transforms
from torchvision import utils as U
from torchvision.transforms import functional as F
from transformers import YolosFeatureExtractor, YolosForObjectDetection


def xyxy2xywh(boxes):
    """convert from xyxy to x1y1 wh"""
    """its the coco standard"""

    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    return torch.stack((x1, y1, w, h), axis=-1)


def xyxy2ccwh(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""

    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack((cx, cy, w, h), axis=-1)


def ccwh2xyxy(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""

    cx, cy, w, h = boxes.unbind(-1)
    b = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(b, dim=-1)


def rescale(boxes, size):
    "rescale from yolo output to xyxy standard"

    h, w, *_ = size
    b = ccwh2xyxy(boxes)
    b = b * torch.Tensor([w, h, w, h])
    return b


def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""

    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = num_sizes + num_ratios - 1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = (
        torch.cat(
            (
                size_tensor * torch.sqrt(ratio_tensor[0]),
                sizes[0] * torch.sqrt(ratio_tensor[1:]),
            )
        )
        * in_height
        / in_width
    )  # Handle rectangular inputs
    h = torch.cat(
        (
            size_tensor / torch.sqrt(ratio_tensor[0]),
            sizes[0] / torch.sqrt(ratio_tensor[1:]),
        )
    )

    # Divide by 2 to get half height and half width
    anchor_manipulations = (
        torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    )

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack(
        [shift_x, shift_y, shift_x, shift_y], dim=1
    ).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)


def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    "xyxy -> xywh"

    return plt.Rectangle(
        xy=(bbox[0], bbox[1]),
        width=bbox[2] - bbox[0],
        height=bbox[3] - bbox[1],
        fill=False,
        edgecolor=color,
        linewidth=2,
    )


def show_bboxes(ax, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default=None):
        return (
            default
            if obj is None
            else [obj]
            if not isinstance(obj, (list, tuple))
            else obj
        )

    labels = make_list(labels)
    colors = make_list(colors, ["b", "g", "r", "m", "c"])

    for i, bbox in enumerate(bboxes):

        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        ax.add_patch(rect)

        if labels and len(labels) > i:

            text_color = "k" if color == "w" else "w"
            ax.text(
                rect.xy[0],
                rect.xy[1],
                labels[i],
                va="center",
                ha="center",
                fontsize=9,
                color=text_color,
                bbox=dict(facecolor=color, lw=0),
            )


def iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""

    box_area = lambda b: ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))

    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)

    x1y1 = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    x2y2 = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (x2y2 - x1y1).clamp(min=0)

    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""

    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = iou(anchors, ground_truth)

    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)

    for _ in range(num_gt_boxes):

        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard

    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""

    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)

    return offset


def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""

    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]

    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

    # Initialize class labels and assigned bounding box coordinates with zeros
    class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
    assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

    # Label classes of anchor boxes using their assigned ground-truth
    # bounding boxes. If an anchor box is not assigned any, we label its
    # class as background (the value remains zero)
    indices_true = torch.nonzero(anchors_bbox_map >= 0)
    bb_idx = anchors_bbox_map[indices_true]
    class_labels[indices_true] = label[bb_idx, 0].long() + 1
    assigned_bb[indices_true] = label[bb_idx, 1:]

    # Offset transformation
    offset = offset_boxes(anchors, assigned_bb) * bbox_mask
    batch_offset.append(offset.reshape(-1))
    batch_mask.append(bbox_mask.reshape(-1))
    batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)

    return (bbox_offset, bbox_mask, class_labels)


def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""

    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox


def mean_ap(clss):

    # for each class get AP ... avg for mAP
    for cls in clss:
        pass


def main():

    img = plt.imread("../catdog.jpg")
    h, w = img.shape[:2]

    print(h, w)
    X = torch.rand(size=(1, 3, h, w))  # Construct input data
    Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    print(Y.shape)

    boxes = Y.reshape(h, w, 5, 4)
    bbox_scale = torch.tensor((w, h, w, h))
    labels = [
        "s=0.75, r=1",
        "s=0.5, r=1",
        "s=0.25, r=1",
        "s=0.75, r=2",
        "s=0.75, r=0.5",
    ]
    fig = plt.imshow(img)
    show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale, labels)

    plt.show()

    labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))

    anchors = torch.tensor(
        [
            [0.1, 0.08, 0.52, 0.92],
            [0.08, 0.2, 0.56, 0.95],
            [0.15, 0.3, 0.62, 0.91],
            [0.55, 0.2, 0.9, 0.88],
        ]
    )
    offset_preds = torch.tensor([0] * anchors.numel())
    cls_probs = torch.tensor(
        [
            [0] * 4,  # Predicted background likelihood
            [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood
            [0.1, 0.2, 0.3, 0.9],
        ]  # Predicted cat likelihood
    )

    quit()

    X = torch.rand(size=(10, 4)) * 205  # Construct input data
    print(X)

    print(iou(X, X))


if __name__ == "__main__":
    main()
