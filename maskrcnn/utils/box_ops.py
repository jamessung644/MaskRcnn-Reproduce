"""박스 연산 유틸리티.

박스 표현은 전부 (x1, y1, x2, y2) 코너 좌표.
박스 회귀 파라미터화는 R-CNN 계열 논문 공통의 (tx, ty, tw, th):
    tx = (x - xa) / wa,  ty = (y - ya) / ha
    tw = log(w / wa),    th = log(h / ha)
"""

import math
from typing import Tuple

import torch
from torch import Tensor


def encode_boxes(reference: Tensor, proposals: Tensor, weights: Tuple[float, ...]) -> Tensor:
    """proposals -> reference(GT) 로 가는 회귀 타깃 (tx, ty, tw, th)을 계산한다."""
    wx, wy, ww, wh = weights

    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]

    gx = (reference[:, 0] + reference[:, 2]) * 0.5
    gy = (reference[:, 1] + reference[:, 3]) * 0.5
    gw = reference[:, 2] - reference[:, 0]
    gh = reference[:, 3] - reference[:, 1]

    tx = wx * (gx - px) / pw
    ty = wy * (gy - py) / ph
    tw = ww * torch.log(gw / pw)
    th = wh * torch.log(gh / ph)
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(deltas: Tensor, boxes: Tensor, weights: Tuple[float, ...]) -> Tensor:
    """회귀 출력 deltas를 anchor/proposal 박스에 적용해 예측 박스를 얻는다.

    deltas: (N, K*4), boxes: (N, 4) -> (N, K, 4)
    """
    wx, wy, ww, wh = weights

    px = (boxes[:, 0] + boxes[:, 2]) * 0.5
    py = (boxes[:, 1] + boxes[:, 3]) * 0.5
    pw = boxes[:, 2] - boxes[:, 0]
    ph = boxes[:, 3] - boxes[:, 1]

    deltas = deltas.reshape(deltas.shape[0], -1, 4)
    tx = deltas[:, :, 0] / wx
    ty = deltas[:, :, 1] / wy
    tw = deltas[:, :, 2] / ww
    th = deltas[:, :, 3] / wh

    # exp 폭주 방지 (Detectron과 동일한 클램프)
    scale_clamp = math.log(1000.0 / 16)
    tw = torch.clamp(tw, max=scale_clamp)
    th = torch.clamp(th, max=scale_clamp)

    cx = tx * pw[:, None] + px[:, None]
    cy = ty * ph[:, None] + py[:, None]
    w = torch.exp(tw) * pw[:, None]
    h = torch.exp(th) * ph[:, None]

    out = torch.stack(
        [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=2
    )
    return out


def clip_boxes_to_image(boxes: Tensor, image_size: Tuple[int, int]) -> Tensor:
    """박스를 이미지 경계 (H, W) 안으로 자른다."""
    h, w = image_size
    boxes = boxes.clone()
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min=0, max=w)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min=0, max=h)
    return boxes


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """한 변이 min_size 이상인 박스의 인덱스를 반환한다."""
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return torch.where(keep)[0]


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """pairwise IoU 행렬 (N, M)을 계산한다."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area1[:, None] + area2[None, :] - inter)
