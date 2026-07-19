"""torchvision 사전학습 Mask R-CNN 가중치를 우리 구현으로 매핑한다.

목적: 구조 검증. torchvision의 maskrcnn_resnet50_fpn(COCO 사전학습)과
우리 모델이 레이어 단위로 동형이면, 모든 텐서가 1:1로 (shape까지 일치하며)
매핑되어야 한다. 매핑 실패나 shape 불일치는 곧 구조 차이를 뜻한다.

주의:
  - torchvision은 91-클래스(원본 COCO id 0~90) 체계이므로 우리 모델도
    Config(num_classes=91)로 만들어야 한다.
  - torchvision mask predictor는 배경 포함 91채널, 우리는 배경 제외
    90채널이므로 채널 0(배경)을 잘라내고 매핑한다.
  - torchvision RPN 앵커는 offset 0으로 배치되므로
    Config(anchor_offset=0.0)을 사용해야 출력이 일치한다.
"""

import re
from typing import Dict, Tuple

import torch
from torch import Tensor


def _map_key(tv_key: str) -> Tuple[str, bool]:
    """torchvision state_dict 키 -> (우리 키, 배경 채널 슬라이스 여부).

    매핑 대상이 아니면 (None, False)를 반환한다.
    """
    k = tv_key

    # ResNet 백본: backbone.body.X -> backbone.X (블록/다운샘플 이름 동일)
    if k.startswith("backbone.body."):
        return "backbone." + k[len("backbone.body."):], False

    # FPN: inner(1x1 lateral)/layer(3x3 output) 블록.
    # 버전에 따라 ".{i}.weight" 또는 ".{i}.0.weight" 형태.
    m = re.match(r"backbone\.fpn\.inner_blocks\.(\d+)(?:\.0)?\.(weight|bias)$", k)
    if m:
        return f"fpn.lateral_convs.{m.group(1)}.{m.group(2)}", False
    m = re.match(r"backbone\.fpn\.layer_blocks\.(\d+)(?:\.0)?\.(weight|bias)$", k)
    if m:
        return f"fpn.output_convs.{m.group(1)}.{m.group(2)}", False

    # RPN 헤드
    m = re.match(r"rpn\.head\.conv(?:\.0\.0)?\.(weight|bias)$", k)
    if m:
        return f"rpn.head.conv.{m.group(1)}", False
    m = re.match(r"rpn\.head\.cls_logits\.(weight|bias)$", k)
    if m:
        return f"rpn.head.objectness.{m.group(1)}", False
    m = re.match(r"rpn\.head\.bbox_pred\.(weight|bias)$", k)
    if m:
        return f"rpn.head.bbox_deltas.{m.group(1)}", False

    # Box head / predictor
    m = re.match(r"roi_heads\.box_head\.fc([67])\.(weight|bias)$", k)
    if m:
        fc = "fc1" if m.group(1) == "6" else "fc2"
        return f"box_head.{fc}.{m.group(2)}", False
    m = re.match(r"roi_heads\.box_predictor\.(cls_score|bbox_pred)\.(weight|bias)$", k)
    if m:
        return f"box_predictor.{m.group(1)}.{m.group(2)}", False

    # Mask head convs: "mask_fcn{n}" 또는 Sequential 인덱스 "{n-1}.0"
    m = re.match(r"roi_heads\.mask_head\.mask_fcn(\d)\.(weight|bias)$", k)
    if m:
        conv_idx = (int(m.group(1)) - 1) * 2  # ReLU가 홀수 인덱스
        return f"mask_head.convs.{conv_idx}.{m.group(2)}", False
    m = re.match(r"roi_heads\.mask_head\.(\d)\.0\.(weight|bias)$", k)
    if m:
        return f"mask_head.convs.{int(m.group(1)) * 2}.{m.group(2)}", False

    # Mask predictor
    m = re.match(r"roi_heads\.mask_predictor\.conv5_mask\.(weight|bias)$", k)
    if m:
        return f"mask_head.deconv.{m.group(1)}", False
    m = re.match(r"roi_heads\.mask_predictor\.mask_fcn_logits\.(weight|bias)$", k)
    if m:
        # tv는 배경 포함 91채널 -> 채널 0을 버리고 90채널로
        return f"mask_head.predictor.{m.group(1)}", True

    return None, False


def load_torchvision_maskrcnn(model) -> Dict[str, int]:
    """COCO 사전학습 가중치를 다운로드(캐시)해 model에 로드한다.

    반환: {"mapped": 매핑된 텐서 수, "skipped": 건너뛴 tv 텐서 수}
    모든 우리 파라미터가 채워지지 않으면 RuntimeError.
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn,
    )

    tv_state = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    ).state_dict()

    our_state = model.state_dict()
    new_state: Dict[str, Tensor] = {}
    skipped = []

    for tv_key, tensor in tv_state.items():
        if tv_key.endswith("num_batches_tracked"):
            continue
        our_key, slice_bg = _map_key(tv_key)
        if our_key is None:
            skipped.append(tv_key)
            continue
        if our_key not in our_state:
            raise RuntimeError(f"매핑된 키가 모델에 없음: {tv_key} -> {our_key}")
        if slice_bg:
            tensor = tensor[1:]  # 배경 채널 제거
        if tensor.shape != our_state[our_key].shape:
            raise RuntimeError(
                f"shape 불일치: {tv_key} {tuple(tensor.shape)} -> "
                f"{our_key} {tuple(our_state[our_key].shape)}"
            )
        new_state[our_key] = tensor

    missing = set(our_state) - set(new_state)
    if missing:
        raise RuntimeError(f"채워지지 않은 파라미터 {len(missing)}개: "
                           f"{sorted(missing)[:10]} ...")

    model.load_state_dict(new_state)
    return {"mapped": len(new_state), "skipped": len(skipped)}


def load_torchvision_pretrained(model) -> Dict[str, int]:
    """COCO 사전학습 가중치 중 shape가 맞는 텐서만 로드한다 (파인튜닝용).

    클래스 수가 다른 파인튜닝(예: num_classes != 91)에서는 분류/박스회귀/마스크
    predictor의 shape가 달라 매핑되지 않는다. 그런 텐서는 건너뛰고 백본·FPN·RPN·
    box_head·mask_head conv 등 클래스 독립 가중치만 옮긴다.

    반환: {"loaded": 옮긴 텐서 수, "kept": 무작위 초기화로 남은 우리 파라미터 수}
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn,
    )

    tv_state = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    ).state_dict()

    our_state = model.state_dict()
    new_state = dict(our_state)
    loaded = 0

    for tv_key, tensor in tv_state.items():
        if tv_key.endswith("num_batches_tracked"):
            continue
        our_key, slice_bg = _map_key(tv_key)
        if our_key is None or our_key not in our_state:
            continue
        if slice_bg:
            tensor = tensor[1:]
        if tensor.shape != our_state[our_key].shape:
            continue  # 클래스 의존 predictor 등: 무작위 초기화 유지
        new_state[our_key] = tensor
        loaded += 1

    model.load_state_dict(new_state)
    return {"loaded": loaded, "kept": len(our_state) - loaded}
