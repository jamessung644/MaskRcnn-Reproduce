"""학습 타깃 매칭/샘플링 유틸리티 (Faster R-CNN / Mask R-CNN 학습 절차).

- Matcher: IoU 행렬을 받아 각 예측(anchor/proposal)을 GT에 할당한다.
  두 임계값 사이의 애매한 예측은 -1(무시)로 표시한다 (Faster R-CNN 3.1.2절).
- BalancedPositiveNegativeSampler: positive:negative 비율을 맞춰
  미니배치를 뽑는다 (RPN 256@1:1, RoI 512@25%).
- smooth_l1_loss: 박스 회귀 손실 (Fast R-CNN 논문 식 (3)의 smooth_L1).
"""

from typing import List, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


class Matcher:
    """IoU 기준으로 예측 박스를 GT에 할당한다.

    반환값 matches[i]는 예측 i에 매칭된 GT 인덱스이며,
        >= 0        : 해당 GT에 매칭된 positive
        BELOW_LOW   : background (negative)
        BETWEEN     : 무시 (loss에서 제외)
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_thresh: float, low_thresh: float,
                 allow_low_quality_matches: bool = True):
        assert low_thresh <= high_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou: Tensor) -> Tensor:
        """iou: (num_gt, num_pred) -> matches: (num_pred,) int64."""
        if iou.numel() == 0:
            # GT가 없는 이미지: 전부 background
            return torch.full((iou.shape[1],), self.BELOW_LOW_THRESHOLD,
                              dtype=torch.int64, device=iou.device)

        matched_vals, matches = iou.max(dim=0)  # 예측마다 가장 잘 맞는 GT

        below = matched_vals < self.low_thresh
        between = (matched_vals >= self.low_thresh) & (matched_vals < self.high_thresh)
        matches = matches.clone()
        matches[below] = self.BELOW_LOW_THRESHOLD
        matches[between] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            # Faster R-CNN 3.1.2절: 각 GT에 대해 IoU가 최대인 anchor도
            # positive로 강제한다 (positive가 하나도 없는 GT 방지).
            highest_per_gt, _ = iou.max(dim=1)
            # 각 GT의 최대 IoU와 같은 값을 갖는 (gt, pred) 위치를 모두 복원
            gt_idx, pred_idx = torch.where(
                iou == highest_per_gt[:, None]
            )
            matches[pred_idx] = gt_idx

        return matches


class BalancedPositiveNegativeSampler:
    """positive/negative 인덱스를 목표 비율로 뽑는다."""

    def __init__(self, batch_size_per_image: int, positive_fraction: float):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """labels: (N,) 1=positive, 0=negative, -1=무시.
        반환: (pos_mask, neg_mask) 불리언 텐서."""
        positive = torch.where(labels >= 1)[0]
        negative = torch.where(labels == 0)[0]

        num_pos = int(self.batch_size_per_image * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos)
        num_neg = self.batch_size_per_image - num_pos
        num_neg = min(negative.numel(), num_neg)

        perm_pos = torch.randperm(positive.numel(), device=labels.device)[:num_pos]
        perm_neg = torch.randperm(negative.numel(), device=labels.device)[:num_neg]

        pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        pos_mask[positive[perm_pos]] = True
        neg_mask[negative[perm_neg]] = True
        return pos_mask, neg_mask


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.0,
                   reduction: str = "sum") -> Tensor:
    """Fast R-CNN 식 (3)의 smooth_L1. beta 미만은 2차, 이상은 1차.

    (torch.nn.functional.smooth_l1_loss가 같은 정의를 제공하므로 그대로 사용.)
    """
    return F.smooth_l1_loss(pred, target, beta=beta, reduction=reduction)
