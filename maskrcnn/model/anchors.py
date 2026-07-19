"""앵커 생성 (Faster R-CNN + FPN 방식).

FPN 논문 4.1절: 피라미드가 이미 멀티스케일이므로 레벨당 단일 스케일만 사용.
P2~P6에 각각 {32^2, 64^2, 128^2, 256^2, 512^2} 크기, 종횡비 {1:2, 1:1, 2:1}.
따라서 위치당 앵커는 레벨마다 3개(A=3)다.
"""

from typing import List, Tuple

import torch
from torch import Tensor


class AnchorGenerator:
    def __init__(self, sizes: Tuple[int, ...], ratios: Tuple[float, ...],
                 strides: Tuple[int, ...], offset: float = 0.5):
        assert len(sizes) == len(strides), "레벨 수와 stride 수가 일치해야 함"
        self.strides = strides
        self.ratios = ratios
        self.offset = offset
        # 레벨별 기본 앵커 (0,0) 중심, (A, 4)
        self.cell_anchors = [self._make_cell_anchors(s, ratios) for s in sizes]

    @staticmethod
    def _make_cell_anchors(size: int, ratios: Tuple[float, ...]) -> Tensor:
        """면적 size^2를 유지하면서 종횡비만 바꾼 앵커들을 만든다."""
        anchors = []
        area = float(size * size)
        for ratio in ratios:
            # h / w = ratio, w * h = area
            w = (area / ratio) ** 0.5
            h = w * ratio
            anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, dtype=torch.float32)

    def __call__(self, feature_shapes: List[Tuple[int, int]],
                 device: torch.device) -> List[Tensor]:
        """각 피라미드 레벨의 전체 앵커를 반환한다.

        feature_shapes: 레벨별 (H, W)
        반환: 레벨별 (H*W*A, 4) 텐서 리스트, 입력 이미지 좌표계
        """
        all_anchors = []
        for (h, w), stride, cell in zip(feature_shapes, self.strides,
                                        self.cell_anchors):
            cell = cell.to(device)
            shifts_x = (torch.arange(w, device=device, dtype=torch.float32) + self.offset) * stride
            shifts_y = (torch.arange(h, device=device, dtype=torch.float32) + self.offset) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack(
                [shift_x.reshape(-1), shift_y.reshape(-1),
                 shift_x.reshape(-1), shift_y.reshape(-1)], dim=1
            )
            # (H*W, 1, 4) + (1, A, 4) -> (H*W*A, 4)
            anchors = (shifts[:, None, :] + cell[None, :, :]).reshape(-1, 4)
            all_anchors.append(anchors)
        return all_anchors

    @property
    def num_anchors_per_location(self) -> int:
        return len(self.ratios)
