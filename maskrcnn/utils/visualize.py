"""Detection 결과 시각화 (PIL만 사용).

mask head의 28x28 출력은 박스 좌표계이므로, 박스 크기로 리사이즈해
원본 이미지에 붙여넣는다 (paste_masks_in_image, Detectron 방식).
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import Tensor

# 클래스별 색상 (반복 사용)
_COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 190), (0, 128, 128), (170, 110, 40),
]


def paste_mask(mask: Tensor, box: Tensor, image_size) -> Tensor:
    """28x28 마스크를 박스 영역에 맞게 원본 좌표계 (H, W)로 확장한다."""
    h, w = image_size
    x1, y1, x2, y2 = box.round().int().tolist()
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    out = torch.zeros(h, w)
    if x2 <= x1 or y2 <= y1:
        return out
    resized = F.interpolate(mask[None, None], size=(y2 - y1, x2 - x1),
                            mode="bilinear", align_corners=False)[0, 0]
    out[y1:y2, x1:x2] = resized
    return out


def draw_detections(image: Image.Image, detection: Dict[str, Tensor],
                    class_names: Dict[int, str], score_thresh: float = 0.5,
                    mask_thresh: float = 0.5) -> Image.Image:
    """박스/라벨/마스크를 이미지에 그려 반환한다."""
    image = image.convert("RGB").copy()
    overlay = np.array(image, dtype=np.float32)
    w, h = image.size

    keep = detection["scores"] > score_thresh
    boxes = detection["boxes"][keep]
    labels = detection["labels"][keep]
    scores = detection["scores"][keep]
    masks = detection.get("masks")
    masks = masks[keep] if masks is not None else None

    # 마스크 오버레이
    if masks is not None:
        for i, (box, label) in enumerate(zip(boxes, labels)):
            color = np.array(_COLORS[int(label) % len(_COLORS)], dtype=np.float32)
            m = paste_mask(masks[i], box, (h, w)).numpy() > mask_thresh
            overlay[m] = overlay[m] * 0.5 + color * 0.5

    image = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        color = _COLORS[int(label) % len(_COLORS)]
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        name = class_names.get(int(label), str(int(label)))
        draw.text((x1 + 2, max(y1 - 12, 0)), f"{name} {score:.2f}", fill=color)
    return image
