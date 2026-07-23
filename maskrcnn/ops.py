"""RoIAlign과 NMS의 직접 구현 (torchvision.ops 비의존).

이 프로젝트는 원래 두 연산만 torchvision.ops에서 빌려 썼는데
(roi_align, nms), 여기서 논문 정의 그대로 직접 구현해 그 의존성을 없앤다.
torch 텐서/autograd는 그대로 쓰므로 GPU 학습(DDP 포함)에는 영향이 없다.

- roi_align: Mask R-CNN 논문 3절 'RoIAlign'. RoIPool의 양자화를 제거하고,
  각 bin 안 규칙적 샘플 포인트에서 bilinear interpolation 후 평균한다.
  torchvision.ops.roi_align(aligned=True)과 수치적으로 일치하도록 좌표
  규약(half-pixel offset, 경계 밖 0 처리, 모서리 클램프)을 맞췄다.
  features에 대해 미분 가능하다(advanced-index gather의 backward).
- nms: Fast R-CNN/Faster R-CNN의 greedy NMS. score 내림차순으로 훑으며
  이미 남긴 박스와 IoU가 임계값을 넘는 박스를 제거한다. torchvision.ops.nms와
  같은 시맨틱(IoU > threshold 제거, 남긴 인덱스를 score 내림차순으로 반환).
"""

from typing import Tuple, Union

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------
def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """Greedy non-maximum suppression.

    boxes: (N, 4) xyxy, scores: (N,). 반환: 남긴 박스 인덱스 (score 내림차순).

    torchvision.ops.nms와 동일하게 IoU > iou_threshold인 박스를 제거한다.
    순수 파이썬 루프라 CUDA 커널보다 느리다 — N이 큰 RPN proposal(레벨당
    수천 개)에서는 pre-NMS top-k로 이미 줄인 뒤 호출되므로 실용상 문제없지만,
    torchvision 커널만큼 빠르진 않다는 점만 알아둘 것.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]

        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter)

        order = rest[iou <= iou_threshold]

    return torch.stack(keep)


# ---------------------------------------------------------------------------
# RoIAlign
# ---------------------------------------------------------------------------
def _bilinear_sample(feature: Tensor, batch_idx: Tensor,
                     y: Tensor, x: Tensor) -> Tensor:
    """feature(B,C,H,W)에서 연속 좌표 (y,x)의 값을 bilinear로 샘플한다.

    batch_idx: (K,), y: (K, OH), x: (K, OW). 반환: (K, C, OH, OW).
    torchvision 규약: y<-1 or y>H or x<-1 or x>W면 0; 그 외에는 0으로 클램프한
    뒤 low=floor, high=low+1, 단 low>=size-1이면 high=low=size-1(경계).
    """
    B, C, H, W = feature.shape
    K, OH = y.shape
    OW = x.shape[1]

    Y = y[:, :, None].expand(K, OH, OW)   # (K, OH, OW)
    X = x[:, None, :].expand(K, OH, OW)

    valid = (Y >= -1.0) & (Y <= H) & (X >= -1.0) & (X <= W)

    Yc = Y.clamp(min=0.0)
    Xc = X.clamp(min=0.0)

    y_low = Yc.floor().long()
    x_low = Xc.floor().long()

    # low >= size-1이면 경계로 클램프하고 보간 가중치가 0이 되도록 좌표도 맞춘다
    y_edge = y_low >= (H - 1)
    x_edge = x_low >= (W - 1)
    y_low = y_low.clamp(max=H - 1)
    x_low = x_low.clamp(max=W - 1)
    y_high = torch.where(y_edge, y_low, y_low + 1)
    x_high = torch.where(x_edge, x_low, x_low + 1)
    Yc = torch.where(y_edge, y_low.to(Yc.dtype), Yc)
    Xc = torch.where(x_edge, x_low.to(Xc.dtype), Xc)

    ly = Yc - y_low.to(Yc.dtype)
    lx = Xc - x_low.to(Xc.dtype)
    hy = 1.0 - ly
    hx = 1.0 - lx

    # 네 모서리 가중치 (K, OH, OW)
    w1 = (hy * hx)[:, None]   # top-left     -> (K,1,OH,OW)
    w2 = (hy * lx)[:, None]   # top-right
    w3 = (ly * hx)[:, None]   # bottom-left
    w4 = (ly * lx)[:, None]   # bottom-right

    b = batch_idx[:, None, None].expand(K, OH, OW)

    # advanced indexing으로 필요한 지점만 모은다 (feature[b, :, y, x] -> (K,OH,OW,C))
    def gather(yy: Tensor, xx: Tensor) -> Tensor:
        return feature[b, :, yy, xx].permute(0, 3, 1, 2)  # (K, C, OH, OW)

    v1 = gather(y_low, x_low)
    v2 = gather(y_low, x_high)
    v3 = gather(y_high, x_low)
    v4 = gather(y_high, x_high)

    out = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
    return out * valid[:, None].to(out.dtype)


def roi_align(input: Tensor, boxes: Tensor,
              output_size: Union[int, Tuple[int, int]],
              spatial_scale: float = 1.0,
              sampling_ratio: int = -1,
              aligned: bool = False) -> Tensor:
    """torchvision.ops.roi_align 호환 RoIAlign (직접 구현).

    input: (B, C, H, W)
    boxes: (K, 5) — 각 행이 [batch_idx, x1, y1, x2, y2] (input 픽셀 좌표).
    output_size: int 또는 (OH, OW)
    spatial_scale: box 좌표를 feature 좌표로 바꾸는 배율 (예: 1/stride)
    sampling_ratio: bin당 한 변의 샘플 수(>=1). 이 프로젝트는 항상 1 또는 2를
        쓴다. <=0(적응적)은 지원하지 않는다 — 호출부가 모두 양수를 준다.
    aligned: True면 half-pixel offset(-0.5) 적용 (논문/Detectron2 방식).

    반환: (K, C, OH, OW). input에 대해 미분 가능.
    """
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    B, C, H, W = input.shape
    K = boxes.shape[0]
    if K == 0:
        return input.new_zeros((0, C, out_h, out_w))
    if sampling_ratio <= 0:
        raise ValueError("이 구현은 sampling_ratio >= 1만 지원한다 "
                         f"(받은 값: {sampling_ratio}).")

    device = input.device
    batch_idx = boxes[:, 0].round().long()

    # 좌표/보간은 항상 float32로 계산한다. AMP(bfloat16 autocast)에서 input이
    # bf16이어도 box 좌표 정밀도를 유지하고, 마지막에 input dtype으로 되돌린다
    # (pooler가 output[idx] = roi_align(...)로 대입하므로 dtype이 반드시
    # input과 같아야 한다 — 안 맞으면 index_put dtype mismatch 에러).
    dtype = torch.float32
    boxes = boxes.float()

    offset = 0.5 if aligned else 0.0
    x1 = boxes[:, 1] * spatial_scale - offset
    y1 = boxes[:, 2] * spatial_scale - offset
    x2 = boxes[:, 3] * spatial_scale - offset
    y2 = boxes[:, 4] * spatial_scale - offset

    roi_w = x2 - x1
    roi_h = y2 - y1
    if not aligned:
        roi_w = roi_w.clamp(min=1.0)
        roi_h = roi_h.clamp(min=1.0)

    bin_w = roi_w / out_w        # (K,)
    bin_h = roi_h / out_h
    sr_h = sr_w = int(sampling_ratio)
    count = sr_h * sr_w

    ph = torch.arange(out_h, device=device, dtype=dtype)  # (OH,)
    pw = torch.arange(out_w, device=device, dtype=dtype)  # (OW,)

    output = torch.zeros((K, C, out_h, out_w), dtype=dtype, device=device)
    # bin당 sr_h x sr_w 서브샘플을 순회하며 누적 — 모든 샘플을 한 번에
    # 펼치지 않아 peak memory가 낮다.
    for iy in range(sr_h):
        # yy: (K, OH) — 각 출력 행 중심의 iy번째 서브샘플 y좌표
        yy = (y1[:, None]
              + ph[None, :] * bin_h[:, None]
              + (iy + 0.5) * bin_h[:, None] / sr_h)
        for ix in range(sr_w):
            xx = (x1[:, None]
                  + pw[None, :] * bin_w[:, None]
                  + (ix + 0.5) * bin_w[:, None] / sr_w)
            output = output + _bilinear_sample(input, batch_idx, yy, xx)

    return (output / count).to(input.dtype)
