"""Box head와 Mask head (Mask R-CNN 논문 3절, 그림 4 우측 'FPN' 구성).

Box head (FPN 백본용):
    RoIAlign 7x7x256 -> flatten -> FC 1024 -> FC 1024
    -> {classification (K+1), box regression (4 per class)}

Mask head (FPN 백본용):
    RoIAlign 14x14x256 -> [3x3 conv 256, ReLU] x4
    -> 2x2 deconv stride 2 (28x28) -> 1x1 conv -> K개의 클래스별 마스크

논문 3.1절: 마스크는 클래스별로 독립적으로 예측하고(sigmoid), 클래스 간
경쟁이 없다. 어떤 채널을 쓸지는 box branch의 분류 결과가 정한다.
"""

import torch
from torch import Tensor, nn


class BoxHead(nn.Module):
    """2개의 1024-d FC로 이루어진 공유 특징 추출기."""

    def __init__(self, in_channels: int, pool_size: int, fc_dim: int = 1024):
        super().__init__()
        in_dim = in_channels * pool_size * pool_size
        self.fc1 = nn.Linear(in_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.relu = nn.ReLU(inplace=True)

        for fc in (self.fc1, self.fc2):
            # Detectron: FC는 Xavier 초기화
            nn.init.xavier_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x


class BoxPredictor(nn.Module):
    """분류 로짓 (K+1) + 클래스별 박스 회귀 (4*(K+1))."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_dim, num_classes)
        self.bbox_pred = nn.Linear(in_dim, num_classes * 4)

        # Fast R-CNN 논문: cls는 N(0, 0.01), box reg는 N(0, 0.001)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.zeros_(self.bbox_pred.bias)

    def forward(self, x: Tensor):
        return self.cls_score(x), self.bbox_pred(x)


class MaskHead(nn.Module):
    """4x conv -> deconv(2x) -> 1x1 conv, 출력 (N, K, 28, 28) 로짓."""

    def __init__(self, in_channels: int, num_classes: int,
                 num_convs: int = 4, conv_dim: int = 256):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(num_convs):
            layers.append(nn.Conv2d(ch, conv_dim, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch = conv_dim
        self.convs = nn.Sequential(*layers)

        self.deconv = nn.ConvTranspose2d(conv_dim, conv_dim, 2, stride=2)
        self.predictor = nn.Conv2d(conv_dim, num_classes - 1, 1)  # 배경 제외 K개

        # Mask R-CNN 논문 4.1절 각주: mask head는 MSRA(He) 초기화
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x)
        x = torch.relu(self.deconv(x))
        return self.predictor(x)
