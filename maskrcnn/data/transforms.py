"""전처리: 리사이즈 + ImageNet 정규화 + 배치 패딩 (Mask R-CNN 논문 3.1절).

- 리사이즈: 짧은 변을 800으로, 단 긴 변이 1333을 넘지 않도록 하는 단일 스케일
  (논문/Detectron 표준). 스칼라 배율 하나로 이미지·박스·마스크를 함께 변환하므로
  추론 후 박스를 원본 좌표로 되돌릴 때 그 배율로 나누기만 하면 된다.
- 정규화: ImageNet 평균/표준편차 (백본이 ImageNet 사전학습이므로).
- 패딩: 배치 내 최대 크기(32의 배수)로 오른쪽/아래를 0 패딩해 텐서로 쌓는다.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_SIZE = 800
MAX_SIZE = 1333
SIZE_DIVISIBLE = 32


def resize_image_and_target(image: Tensor,
                            target: Optional[Dict[str, Tensor]],
                            min_size: int = MIN_SIZE,
                            max_size: int = MAX_SIZE
                            ) -> Tuple[Tensor, Optional[Dict[str, Tensor]], float]:
    """이미지(그리고 있으면 target)를 단일 스케일로 리사이즈한다.

    image: (3, H, W) float [0, 1]
    반환: (resized_image, resized_target, scale)
    """
    h, w = image.shape[-2:]
    scale = min_size / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)

    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    image = F.interpolate(image[None], size=(new_h, new_w),
                          mode="bilinear", align_corners=False)[0]

    if target is not None:
        target = dict(target)
        boxes = target.get("boxes")
        if boxes is not None and boxes.numel() > 0:
            target["boxes"] = boxes * scale
        masks = target.get("masks")
        if masks is not None and masks.numel() > 0:
            masks = F.interpolate(masks[None].float(), size=(new_h, new_w),
                                  mode="nearest")[0]
            target["masks"] = masks.to(target["masks"].dtype)
        elif masks is not None:
            target["masks"] = masks.new_zeros((0, new_h, new_w))

    return image, target, scale


def normalize_image(image: Tensor) -> Tensor:
    """ImageNet 평균/표준편차로 채널별 정규화."""
    mean = torch.as_tensor(IMAGENET_MEAN, dtype=image.dtype, device=image.device)
    std = torch.as_tensor(IMAGENET_STD, dtype=image.dtype, device=image.device)
    return (image - mean[:, None, None]) / std[:, None, None]


def batch_images(images: List[Tensor],
                 size_divisible: int = SIZE_DIVISIBLE
                 ) -> Tuple[Tensor, List[Tuple[int, int]]]:
    """서로 다른 크기의 이미지들을 오른쪽/아래 0 패딩해 하나의 배치로 쌓는다.

    반환: (batch (B,3,Hmax,Wmax), image_sizes) — image_sizes는 패딩 전 (H,W).
    """
    image_sizes = [(int(img.shape[-2]), int(img.shape[-1])) for img in images]
    max_h = max(s[0] for s in image_sizes)
    max_w = max(s[1] for s in image_sizes)

    def _ceil(x: int) -> int:
        return int((x + size_divisible - 1) // size_divisible * size_divisible)

    max_h, max_w = _ceil(max_h), _ceil(max_w)

    batch = images[0].new_zeros((len(images), 3, max_h, max_w))
    for i, img in enumerate(images):
        batch[i, :, : img.shape[-2], : img.shape[-1]] = img
    return batch, image_sizes


def collate_fn(batch: List[Tuple[Tensor, Dict[str, Tensor]]]
               ) -> Tuple[Tensor, List[Tuple[int, int]], List[Dict[str, Tensor]]]:
    """DataLoader용 collate: (정규화 -> 패딩) 후 (images, image_sizes, targets) 반환.

    Dataset이 이미 리사이즈까지 마친 (image, target)을 준다고 가정한다.
    """
    images = [normalize_image(img) for img, _ in batch]
    targets = [t for _, t in batch]
    batched, image_sizes = batch_images(images)
    return batched, image_sizes, targets
