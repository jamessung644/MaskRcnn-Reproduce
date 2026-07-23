"""Region Proposal Network (Ren et al., NIPS 2015 + FPN 적용).

FPN 논문 4.1절: RPN 헤드(3x3 conv + 두 개의 1x1 conv)는 모든 피라미드
레벨 P2~P6에서 파라미터를 공유한다.

objectness는 앵커당 1개 로짓 + sigmoid로 구현했다 (원 논문은 2-way
softmax지만 수학적으로 동치이며, Detectron 계열 표준 구현 방식).

학습(He et al. / Ren et al. 3.1절):
    - 각 anchor를 GT에 매칭(IoU>=0.7 pos, <0.3 neg, 그 사이 무시).
    - 이미지당 256개를 pos:neg=1:1로 샘플링.
    - 손실 L = L_obj(BCE) + L_box(smooth_L1, positive anchor만).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from ..ops import nms

from ..utils import box_ops
from .matcher import BalancedPositiveNegativeSampler, Matcher, smooth_l1_loss


class RPNHead(nn.Module):
    """레벨 공유 RPN 헤드: 3x3 conv -> {objectness 1xA, box 4xA}."""

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_deltas = nn.Conv2d(in_channels, num_anchors * 4, 1)

        # Faster R-CNN 논문: 새 레이어는 N(0, 0.01) 초기화
        for m in (self.conv, self.objectness, self.bbox_deltas):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.zeros_(m.bias)

    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits, deltas = [], []
        for x in features:
            t = torch.relu(self.conv(x))
            logits.append(self.objectness(t))
            deltas.append(self.bbox_deltas(t))
        return logits, deltas


class RPN(nn.Module):
    def __init__(self, cfg, anchor_generator):
        super().__init__()
        self.cfg = cfg
        self.anchor_generator = anchor_generator
        self.head = RPNHead(cfg.fpn_out_channels,
                            anchor_generator.num_anchors_per_location)

        # 학습용 타깃 매칭/샘플링
        self.matcher = Matcher(cfg.rpn_fg_iou_thresh, cfg.rpn_bg_iou_thresh,
                               allow_low_quality_matches=True)
        self.sampler = BalancedPositiveNegativeSampler(
            cfg.rpn_batch_size_per_image, cfg.rpn_positive_fraction)

    @staticmethod
    def _flatten_per_level(logits: Tensor, deltas: Tensor) -> Tuple[Tensor, Tensor]:
        """(B, A, H, W) / (B, A*4, H, W) -> (B, H*W*A) / (B, H*W*A, 4)

        앵커 생성 순서(위치 우선, 위치 안에서 A개)와 일치하도록 변환한다.
        """
        b, a, h, w = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(b, -1)
        deltas = deltas.view(b, a, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
        return logits, deltas

    def forward(self, features: Dict[str, Tensor],
                image_sizes: List[Tuple[int, int]],
                targets: Optional[List[Dict[str, Tensor]]] = None
                ) -> Tuple[List[Tensor], Dict[str, Tensor]]:
        """이미지별 proposal 박스 리스트와 (학습 시) 손실 딕셔너리를 반환한다.

        반환: (proposals, losses)
            proposals: 이미지별 (N_i, 4)
            losses: 학습 모드에서 {"rpn_objectness", "rpn_box_reg"}, 아니면 {}
        """
        feature_list = [features[k] for k in ("p2", "p3", "p4", "p5", "p6")]
        logits, deltas = self.head(feature_list)

        device = feature_list[0].device
        anchors_per_level = self.anchor_generator(
            [f.shape[-2:] for f in feature_list], device
        )

        # 레벨별로 (B, HWA), (B, HWA, 4)로 평탄화
        flat_logits, flat_deltas = [], []
        for lvl_logits, lvl_deltas in zip(logits, deltas):
            fl, fd = self._flatten_per_level(lvl_logits, lvl_deltas)
            flat_logits.append(fl)
            flat_deltas.append(fd)

        proposals = self._generate_proposals(
            flat_logits, flat_deltas, anchors_per_level, image_sizes
        )

        losses: Dict[str, Tensor] = {}
        if self.training:
            assert targets is not None, "학습 모드에서는 targets가 필요하다"
            objectness = torch.cat(flat_logits, dim=1)      # (B, sumA)
            pred_deltas = torch.cat(flat_deltas, dim=1)     # (B, sumA, 4)
            anchors = torch.cat(anchors_per_level, dim=0)   # (sumA, 4)
            losses = self.compute_loss(objectness, pred_deltas, anchors, targets)

        return proposals, losses

    # ------------------------------------------------------------------
    def _generate_proposals(self, flat_logits: List[Tensor],
                            flat_deltas: List[Tensor],
                            anchors_per_level: List[Tensor],
                            image_sizes: List[Tuple[int, int]]) -> List[Tensor]:
        pre_nms_top_n = (self.cfg.rpn_pre_nms_top_n_train if self.training
                         else self.cfg.rpn_pre_nms_top_n_test)
        post_nms_top_n = (self.cfg.rpn_post_nms_top_n_train if self.training
                          else self.cfg.rpn_post_nms_top_n_test)

        batch_size = flat_logits[0].shape[0]
        proposals_per_image: List[Tensor] = []

        for img_idx in range(batch_size):
            level_boxes, level_scores = [], []

            for lvl in range(len(flat_logits)):
                scores = flat_logits[lvl][img_idx]          # (HWA,)
                box_deltas = flat_deltas[lvl][img_idx]      # (HWA, 4)
                anchors = anchors_per_level[lvl]

                # 레벨별 pre-NMS top-k (FPN에서는 레벨마다 적용)
                k = min(pre_nms_top_n, scores.shape[0])
                scores, topk_idx = scores.topk(k)
                boxes = box_ops.decode_boxes(
                    box_deltas[topk_idx].reshape(k, 4),
                    anchors[topk_idx],
                    self.cfg.rpn_box_reg_weights,
                ).reshape(k, 4)

                boxes = box_ops.clip_boxes_to_image(boxes, image_sizes[img_idx])
                keep = box_ops.remove_small_boxes(boxes, self.cfg.rpn_min_size)
                level_boxes.append(boxes[keep])
                level_scores.append(scores[keep])

            boxes = torch.cat(level_boxes)
            scores = torch.cat(level_scores).sigmoid()

            # 전체 레벨을 합쳐 NMS 후 post-NMS top-k 유지
            keep = nms(boxes, scores, self.cfg.rpn_nms_thresh)
            keep = keep[:post_nms_top_n]
            proposals_per_image.append(boxes[keep])

        return proposals_per_image

    # ------------------------------------------------------------------
    def compute_loss(self, objectness: Tensor, pred_deltas: Tensor,
                     anchors: Tensor,
                     targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """RPN 손실 (objectness BCE + positive anchor의 box smooth_L1).

        objectness: (B, A), pred_deltas: (B, A, 4), anchors: (A, 4)
        """
        labels_batch, reg_targets_batch = [], []
        for target in targets:
            gt_boxes = target["boxes"]
            if gt_boxes.numel() == 0:
                # GT가 없으면 전부 background, 회귀 타깃은 0
                labels = torch.zeros((anchors.shape[0],), device=anchors.device)
                reg_targets = torch.zeros_like(anchors)
            else:
                iou = box_ops.box_iou(gt_boxes, anchors)  # (num_gt, A)
                matched = self.matcher(iou)               # (A,)

                labels = torch.zeros((anchors.shape[0],), device=anchors.device)
                labels[matched >= 0] = 1.0
                labels[matched == Matcher.BETWEEN_THRESHOLDS] = -1.0  # 무시

                matched_gt = gt_boxes[matched.clamp(min=0)]
                reg_targets = box_ops.encode_boxes(
                    matched_gt, anchors, self.cfg.rpn_box_reg_weights)

            labels_batch.append(labels)
            reg_targets_batch.append(reg_targets)

        # 이미지별로 샘플링한 뒤 배치 전체로 이어붙여 정규화
        sampled_obj, sampled_lbl = [], []
        pos_pred_deltas, pos_reg_targets = [], []

        for i, labels in enumerate(labels_batch):
            pos_mask, neg_mask = self.sampler(labels)
            sampled = pos_mask | neg_mask

            sampled_obj.append(objectness[i][sampled])
            sampled_lbl.append(labels[sampled])
            if pos_mask.any():
                pos_pred_deltas.append(pred_deltas[i][pos_mask])
                pos_reg_targets.append(reg_targets_batch[i][pos_mask])

        sampled_obj = torch.cat(sampled_obj)
        sampled_lbl = torch.cat(sampled_lbl)
        num_sampled = max(sampled_obj.numel(), 1)

        objectness_loss = F.binary_cross_entropy_with_logits(
            sampled_obj, sampled_lbl, reduction="sum") / num_sampled

        if pos_pred_deltas:
            box_loss = smooth_l1_loss(
                torch.cat(pos_pred_deltas), torch.cat(pos_reg_targets),
                beta=1.0 / 9, reduction="sum") / num_sampled
        else:
            box_loss = objectness_loss.new_zeros(())

        return {"rpn_objectness": objectness_loss, "rpn_box_reg": box_loss}
