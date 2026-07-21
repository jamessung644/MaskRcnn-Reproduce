"""Mask R-CNN 전체 조립 (He et al., ICCV 2017).

파이프라인 (논문 3절):
    이미지 -> ResNet-50 (C2~C5) -> FPN (P2~P6)
          -> RPN (P2~P6에서 proposal 생성)
          -> RoIAlign 7x7  -> Box head  -> 분류 + 박스 회귀
          -> RoIAlign 14x14 -> Mask head -> 클래스별 28x28 마스크

추론 절차 (논문 3.1절 'Inference'):
    1. RPN이 이미지당 1000개(FPN 기준) proposal을 낸다.
    2. Box branch를 돌리고 NMS 후 상위 100개 detection만 남긴다.
    3. Mask branch는 이 100개 박스에 대해서만 실행한다 (속도 개선).
    4. 각 RoI에서 분류된 클래스 k의 마스크 채널만 취해 sigmoid.

학습 절차 (논문 3.1절 'Training'):
    L = L_cls + L_box + L_mask
    - proposal을 GT에 매칭(IoU>=0.5 foreground), 512개를 25% positive로 샘플링.
    - L_cls: (K+1)-way cross-entropy, L_box: positive RoI의 클래스별 smooth_L1.
    - L_mask: positive RoI에서 GT 클래스 채널만 골라 per-pixel BCE (마스크 간
      경쟁 없음). GT 마스크는 proposal 박스로 잘라 28x28로 리샘플한 것.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import nms, roi_align

from ..config import Config
from ..utils import box_ops
from .anchors import AnchorGenerator
from .fpn import FPN
from .heads import BoxHead, BoxPredictor, MaskHead
from .matcher import BalancedPositiveNegativeSampler, Matcher, smooth_l1_loss
from .pooler import MultiScaleRoIAlign
from .resnet import resnet18, resnet34, resnet50, resnet101
from .rpn import RPN

# backbone_depth -> 생성 함수. 18/34는 BasicBlock 기반 경량 백본으로, VRAM이
# 작거나(예: 16GB급 GPU) 처리량을 우선할 때 쓴다(연산량이 50/101보다 훨씬
# 작은 대신, torchvision COCO 사전학습 백본과는 shape가 달라 그 이득은
# 못 받는다 — resnet.py 모듈 docstring 참고).
_BACKBONES = {18: resnet18, 34: resnet34, 50: resnet50, 101: resnet101}


class MaskRCNN(nn.Module):
    def __init__(self, cfg: Config = None, backbone_depth: int = 50,
                 freeze_at: int = 2, grad_checkpoint: bool = False):
        super().__init__()
        self.cfg = cfg = cfg or Config()

        # ---- 백본 + FPN
        if backbone_depth not in _BACKBONES:
            raise ValueError(
                f"backbone_depth={backbone_depth} 미지원 (가능한 값: "
                f"{sorted(_BACKBONES)})")
        backbone_fn = _BACKBONES[backbone_depth]
        self.backbone = backbone_fn(freeze_at=freeze_at,
                                    grad_checkpoint=grad_checkpoint)
        self.fpn = FPN(self.backbone.out_channels, cfg.fpn_out_channels)

        # ---- RPN
        anchor_generator = AnchorGenerator(
            cfg.anchor_sizes, cfg.anchor_ratios, cfg.feature_strides,
            cfg.anchor_offset,
        )
        self.rpn = RPN(cfg, anchor_generator)

        # ---- Box branch
        self.box_pooler = MultiScaleRoIAlign(
            cfg.box_roi_pool_size, cfg.roi_sampling_ratio,
            cfg.roi_canonical_scale, cfg.roi_canonical_level,
            cfg.roi_min_level, cfg.roi_max_level,
        )
        self.box_head = BoxHead(
            cfg.fpn_out_channels, cfg.box_roi_pool_size, cfg.box_head_fc_dim
        )
        self.box_predictor = BoxPredictor(cfg.box_head_fc_dim, cfg.num_classes)

        # ---- Mask branch
        self.mask_pooler = MultiScaleRoIAlign(
            cfg.mask_roi_pool_size, cfg.roi_sampling_ratio,
            cfg.roi_canonical_scale, cfg.roi_canonical_level,
            cfg.roi_min_level, cfg.roi_max_level,
        )
        self.mask_head = MaskHead(
            cfg.fpn_out_channels, cfg.num_classes,
            cfg.mask_head_num_convs, cfg.mask_head_conv_dim,
        )

        # ---- RoI 학습용 매칭/샘플링
        # box head는 임계값 하나(0.5) 기준: IoU>=0.5 foreground, 미만 background.
        # (애매 구간 없음 -> allow_low_quality_matches=False)
        self.box_matcher = Matcher(cfg.box_fg_iou_thresh, cfg.box_bg_iou_thresh,
                                   allow_low_quality_matches=False)
        self.box_sampler = BalancedPositiveNegativeSampler(
            cfg.box_batch_size_per_image, cfg.box_positive_fraction)

    # ------------------------------------------------------------------
    def forward(self, images: Tensor,
                image_sizes: List[Tuple[int, int]] = None,
                targets: Optional[List[Dict[str, Tensor]]] = None):
        """학습 모드면 손실 딕셔너리를, 추론 모드면 detection 리스트를 반환.

        images: (B, 3, H, W) — 정규화/패딩이 끝난 배치 텐서
        image_sizes: 패딩 전 각 이미지의 실제 (H, W). 없으면 텐서 크기 사용.
        targets: 학습 시 이미지별 {"boxes"(N,4), "labels"(N,), "masks"(N,H,W)}
        """
        if image_sizes is None:
            image_sizes = [tuple(images.shape[-2:])] * images.shape[0]

        if self.training:
            if targets is None:
                raise ValueError("학습 모드에서는 targets가 필요하다.")
            return self._forward_train(images, image_sizes, targets)
        return self._forward_inference(images, image_sizes)

    # ------------------------------------------------------------------
    def _forward_train(self, images: Tensor,
                       image_sizes: List[Tuple[int, int]],
                       targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        features = self.fpn(self.backbone(images))
        proposals, rpn_losses = self.rpn(features, image_sizes, targets)

        # proposal은 고정된 입력으로 취급 (RPN으로 그래디언트 전파 차단).
        # Detectron 관례: 초반 학습 안정화를 위해 GT 박스를 proposal에 추가.
        proposals = [
            torch.cat([p.detach(), t["boxes"]])
            for p, t in zip(proposals, targets)
        ]

        proposals, matched_idxs, labels, regression_targets = \
            self.select_training_samples(proposals, targets)

        # ---- Box branch
        box_features = self.box_pooler(features, proposals)
        class_logits, box_deltas = self.box_predictor(self.box_head(box_features))
        loss_cls, loss_box_reg = self.fastrcnn_loss(
            class_logits, box_deltas, labels, regression_targets)

        # ---- Mask branch: 이미지별 positive proposal만 사용
        mask_proposals, pos_matched_idxs, pos_labels = [], [], []
        for props, m_idx, lbl in zip(proposals, matched_idxs, labels):
            pos = torch.where(lbl >= 1)[0]
            mask_proposals.append(props[pos])
            pos_matched_idxs.append(m_idx[pos])
            pos_labels.append(lbl[pos])

        if sum(p.shape[0] for p in mask_proposals) > 0:
            mask_features = self.mask_pooler(features, mask_proposals)
            mask_logits = self.mask_head(mask_features)
            loss_mask = self.maskrcnn_loss(
                mask_logits, mask_proposals, targets, pos_matched_idxs, pos_labels)
        else:
            loss_mask = class_logits.new_zeros(())

        losses = {
            "loss_box_cls": loss_cls,
            "loss_box_reg": loss_box_reg,
            "loss_mask": loss_mask,
        }
        losses.update(rpn_losses)
        return losses

    # ------------------------------------------------------------------
    def select_training_samples(self, proposals: List[Tensor],
                                targets: List[Dict[str, Tensor]]):
        """proposal을 GT에 매칭하고 512개(25% positive)를 샘플링한다.

        반환(모두 이미지별 리스트):
            proposals          : 샘플링된 proposal (S, 4)
            matched_gt_idxs    : 각 proposal이 매칭된 GT 인덱스 (S,)
            labels             : 클래스 라벨 (S,), background=0
            regression_targets : 박스 회귀 타깃 (S, 4)
        """
        out_props, out_matched, out_labels, out_reg = [], [], [], []

        for props, target in zip(proposals, targets):
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]

            if gt_boxes.numel() == 0:
                device = props.device
                matched = torch.zeros((props.shape[0],), dtype=torch.int64,
                                      device=device)
                labels = torch.zeros((props.shape[0],), dtype=torch.int64,
                                     device=device)
            else:
                iou = box_ops.box_iou(gt_boxes, props)   # (num_gt, num_prop)
                matched = self.box_matcher(iou)          # (num_prop,)
                clamped = matched.clamp(min=0)
                labels = gt_labels[clamped].to(torch.int64)
                labels[matched == Matcher.BELOW_LOW_THRESHOLD] = 0  # background

            pos_mask, neg_mask = self.box_sampler(labels)
            sampled = torch.where(pos_mask | neg_mask)[0]

            props_s = props[sampled]
            matched_s = matched[sampled]
            labels_s = labels[sampled]

            if gt_boxes.numel() == 0:
                reg_s = torch.zeros_like(props_s)
            else:
                matched_gt = gt_boxes[matched_s.clamp(min=0)]
                reg_s = box_ops.encode_boxes(
                    matched_gt, props_s, self.cfg.box_reg_weights)

            out_props.append(props_s)
            out_matched.append(matched_s)
            out_labels.append(labels_s)
            out_reg.append(reg_s)

        return out_props, out_matched, out_labels, out_reg

    # ------------------------------------------------------------------
    def fastrcnn_loss(self, class_logits: Tensor, box_deltas: Tensor,
                      labels: List[Tensor],
                      regression_targets: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """(K+1)-way 분류 CE + positive RoI의 클래스별 박스 smooth_L1."""
        labels_cat = torch.cat(labels)
        reg_cat = torch.cat(regression_targets)

        classification_loss = F.cross_entropy(class_logits, labels_cat)

        # 박스 회귀는 foreground RoI에서, 예측된 GT 클래스 채널만 사용
        pos = torch.where(labels_cat > 0)[0]
        if pos.numel() == 0:
            box_loss = class_logits.new_zeros(())
        else:
            labels_pos = labels_cat[pos]
            n = class_logits.shape[0]
            box_deltas = box_deltas.reshape(n, -1, 4)  # (N, K+1, 4)
            box_loss = smooth_l1_loss(
                box_deltas[pos, labels_pos], reg_cat[pos],
                beta=1.0, reduction="sum") / labels_cat.numel()

        return classification_loss, box_loss

    # ------------------------------------------------------------------
    def maskrcnn_loss(self, mask_logits: Tensor, mask_proposals: List[Tensor],
                      targets: List[Dict[str, Tensor]],
                      pos_matched_idxs: List[Tensor],
                      pos_labels: List[Tensor]) -> Tensor:
        """positive RoI에서 GT 클래스 채널만 골라 per-pixel BCE.

        GT 마스크는 proposal 박스로 잘라 28x28로 리샘플한 것을 타깃으로 쓴다.
        """
        M = self.cfg.mask_resolution
        mask_targets, labels_cat = [], []
        for props, target, m_idx, lbl in zip(
                mask_proposals, targets, pos_matched_idxs, pos_labels):
            if props.shape[0] == 0:
                continue
            gt_masks = target["masks"]  # (num_gt, h_m, w_m) — 다운샘플됐을 수 있음
            # mask_downsample>1로 GT 마스크를 이미지보다 더 줄여 저장했다면
            # (호스트 RAM 절약, CocoInstanceDataset(mask_downsample=..) 참고)
            # box 좌표(원본 이미지 스케일)를 그만큼 줄여서 마스크 좌표계에
            # 맞춰야 한다 — 그 배율이 target["mask_scale"]이다(기본 1.0).
            mask_scale = float(target.get("mask_scale", 1.0))
            mask_targets.append(
                _project_masks_on_boxes(gt_masks, props, m_idx, M, mask_scale))
            labels_cat.append(lbl)

        mask_targets = torch.cat(mask_targets, dim=0)  # (P, M, M)
        labels_cat = torch.cat(labels_cat)             # (P,)

        # 라벨 1..K -> 마스크 채널 0..K-1
        idx = torch.arange(mask_logits.shape[0], device=mask_logits.device)
        selected = mask_logits[idx, labels_cat - 1]    # (P, M, M)

        return F.binary_cross_entropy_with_logits(selected, mask_targets)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_inference(self, images: Tensor,
                           image_sizes: List[Tuple[int, int]]
                           ) -> List[Dict[str, Tensor]]:
        features = self.fpn(self.backbone(images))
        proposals, _ = self.rpn(features, image_sizes)

        # ---- Box branch
        box_features = self.box_pooler(features, proposals)
        class_logits, box_deltas = self.box_predictor(self.box_head(box_features))

        detections = self._postprocess_detections(
            class_logits, box_deltas, proposals, image_sizes
        )

        # ---- Mask branch: 최종 detection 박스에 대해서만 실행 (논문 3.1절)
        detection_boxes = [d["boxes"] for d in detections]
        mask_features = self.mask_pooler(features, detection_boxes)
        mask_logits = self.mask_head(mask_features)  # (sum(N_i), K, 28, 28)

        # 예측된 클래스 채널만 선택해 sigmoid
        labels = torch.cat([d["labels"] for d in detections])
        if labels.numel() > 0:
            idx = torch.arange(labels.shape[0], device=labels.device)
            masks = mask_logits[idx, labels - 1].sigmoid()  # 라벨 1..K -> 채널 0..K-1
        else:
            masks = mask_logits.new_zeros((0, self.cfg.mask_resolution,
                                           self.cfg.mask_resolution))

        offset = 0
        for det in detections:
            n = det["boxes"].shape[0]
            det["masks"] = masks[offset:offset + n]
            offset += n

        return detections

    # ------------------------------------------------------------------
    def _postprocess_detections(self, class_logits: Tensor, box_deltas: Tensor,
                                proposals: List[Tensor],
                                image_sizes: List[Tuple[int, int]]
                                ) -> List[Dict[str, Tensor]]:
        """softmax -> 클래스별 박스 디코딩 -> score threshold
        -> 클래스별 NMS(0.5) -> 상위 100개 유지."""
        cfg = self.cfg
        num_classes = cfg.num_classes
        scores_all = F.softmax(class_logits, dim=-1)

        boxes_per_image = [p.shape[0] for p in proposals]
        scores_split = scores_all.split(boxes_per_image)
        deltas_split = box_deltas.split(boxes_per_image)

        results = []
        for scores, deltas, props, img_size in zip(
                scores_split, deltas_split, proposals, image_sizes):
            # (N, K+1, 4) 클래스별 디코딩
            boxes = box_ops.decode_boxes(deltas, props, cfg.box_reg_weights)
            boxes = box_ops.clip_boxes_to_image(boxes, img_size)

            final_boxes, final_scores, final_labels = [], [], []
            for cls in range(1, num_classes):  # 0 = 배경, 건너뜀
                cls_scores = scores[:, cls]
                keep = cls_scores > cfg.score_thresh
                if keep.sum() == 0:
                    continue
                cls_boxes = boxes[keep, cls]
                cls_scores = cls_scores[keep]

                keep = nms(cls_boxes, cls_scores, cfg.detections_nms_thresh)
                final_boxes.append(cls_boxes[keep])
                final_scores.append(cls_scores[keep])
                final_labels.append(torch.full_like(
                    cls_scores[keep], cls, dtype=torch.int64))

            if final_boxes:
                boxes_cat = torch.cat(final_boxes)
                scores_cat = torch.cat(final_scores)
                labels_cat = torch.cat(final_labels)
                # 이미지당 상위 detections_per_img개
                k = min(cfg.detections_per_img, scores_cat.shape[0])
                scores_cat, topk = scores_cat.topk(k)
                boxes_cat, labels_cat = boxes_cat[topk], labels_cat[topk]
            else:
                boxes_cat = props.new_zeros((0, 4))
                scores_cat = props.new_zeros((0,))
                labels_cat = props.new_zeros((0,), dtype=torch.int64)

            results.append(
                {"boxes": boxes_cat, "scores": scores_cat, "labels": labels_cat}
            )
        return results


def _project_masks_on_boxes(gt_masks: Tensor, boxes: Tensor,
                            matched_idxs: Tensor, M: int,
                            mask_scale: float = 1.0) -> Tensor:
    """positive proposal 박스로 GT 마스크를 잘라 MxM으로 리샘플한다.

    gt_masks: (num_gt, h_m, w_m), boxes: (P, 4) — 원본 이미지 스케일 좌표,
    matched_idxs: (P,) GT 인덱스.
    mask_scale: gt_masks가 이미지보다 이 배율만큼 더 다운샘플된 상태일 때(호스트
        RAM 절약, CocoInstanceDataset(mask_downsample=..) 참고) box 좌표를
        gt_masks 좌표계로 맞추기 위한 roi_align의 spatial_scale. 기본 1.0이면
        기존과 동일(다운샘플 없음).
    반환: (P, M, M), 값은 [0, 1] (roi_align bilinear 결과) — BCE 타깃으로 사용.
    """
    rois = torch.cat([matched_idxs[:, None].to(boxes), boxes], dim=1)
    gt_masks = gt_masks[:, None].to(boxes)  # (num_gt, 1, h_m, w_m)
    return roi_align(gt_masks, rois, (M, M), spatial_scale=mask_scale,
                     sampling_ratio=1, aligned=True)[:, 0]
