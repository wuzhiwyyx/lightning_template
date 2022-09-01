'''
 # @ Author: Zhi Wu
 # @ Create Time: 1970-01-01 00:00:00
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-27 13:38:22
 # @ Description: Additional useful definition.
 '''

import torch
from torchvision.ops.boxes import box_area

def box_iou(boxes1, boxes2, eps=1e-7):
    """Calucate ious between two bunch of boxes.

    Args:
        boxes1 (tensor): First bunch of boxes.
        boxes2 (tensor): Second bunch of boxes.
        eps (float, optional): Avoid zero divide. Defaults to 1e-7.

    Returns:
        _type_: _description_
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + eps)
    return iou, union