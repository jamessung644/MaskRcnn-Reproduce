"""Feature Pyramid Network (Lin et al., CVPR 2017, 3절).

백본의 C2~C5를 받아 P2~P6을 만든다.

  - lateral: 각 Ci에 1x1 conv를 적용해 채널을 d=256으로 맞춘다.
  - top-down: 상위 레벨을 nearest neighbor로 2배 업샘플해 lateral과 element-wise 합.
  - output: 합쳐진 맵에 3x3 conv를 적용해 업샘플링의 앨리어싱을 줄인다.
  - P6: P5에 stride 2 subsampling(maxpool)을 적용 (FPN 논문 4.1절, RPN 전용).

피라미드에는 비선형성이 없다 (논문 3절: "there are no non-linearities").
"""

from typing import Dict, List

import torch.nn.functional as F
from torch import Tensor, nn


class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1)
             for _ in in_channels_list]
        )
        self.out_channels = out_channels

        # FPN 논문 3절: 새로 추가된 conv는 Xavier 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        c2, c3, c4, c5 = features["c2"], features["c3"], features["c4"], features["c5"]

        # top-down 경로: 가장 거친 레벨(C5)부터 시작
        laterals = [
            lateral(c) for lateral, c in
            zip(self.lateral_convs, (c2, c3, c4, c5))
        ]

        merged = [laterals[-1]]  # M5
        for lateral in laterals[-2::-1]:  # M4, M3, M2
            top_down = F.interpolate(merged[0], size=lateral.shape[-2:], mode="nearest")
            merged.insert(0, lateral + top_down)

        outputs = [conv(m) for conv, m in zip(self.output_convs, merged)]
        p2, p3, p4, p5 = outputs
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2)

        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}
