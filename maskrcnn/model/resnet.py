"""ResNet 백본 (He et al., CVPR 2016).

Mask R-CNN 논문 3절 'Network Architecture'의 표기로 C2~C5, 즉
conv2_x ~ conv5_x 각 스테이지의 마지막 residual block 출력을 반환한다.

ResNet-50 구성 (He et al. 2016, Table 1):
    stem : 7x7 conv, 64ch, stride 2  ->  3x3 maxpool, stride 2
    conv2_x: bottleneck x3, 출력  256ch, stride  4 (C2)
    conv3_x: bottleneck x4, 출력  512ch, stride  8 (C3)
    conv4_x: bottleneck x6, 출력 1024ch, stride 16 (C4)
    conv5_x: bottleneck x3, 출력 2048ch, stride 32 (C5)

detection 학습은 배치가 작아 BN 통계가 불안정하므로, Faster/Mask R-CNN
관례대로 BN을 고정(FrozenBatchNorm)한다.
"""

from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class FrozenBatchNorm2d(nn.Module):
    """통계와 아핀 파라미터를 모두 고정한 BatchNorm.

    사전학습 가중치를 로드해 y = (x - mean) / sqrt(var + eps) * weight + bias
    를 상수 아핀 변환으로만 적용한다.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x: Tensor) -> Tensor:
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        shift = self.bias - self.running_mean * scale
        return x * scale[None, :, None, None] + shift[None, :, None, None]


class Bottleneck(nn.Module):
    """1x1 -> 3x3 -> 1x1 bottleneck residual block (He et al. 2016, 그림 5 우)."""

    expansion = 4

    def __init__(self, in_channels: int, bottleneck_channels: int, stride: int = 1):
        super().__init__()
        out_channels = bottleneck_channels * self.expansion

        # 다운샘플은 3x3 conv에서 수행 (torchvision 'ResNet v1.5' 방식,
        # Detectron의 기본 구성과 동일)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1, bias=False)
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(
            bottleneck_channels, bottleneck_channels, 3,
            stride=stride, padding=1, bias=False,
        )
        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1, bias=False)
        self.bn3 = FrozenBatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                FrozenBatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


class ResNet(nn.Module):
    """C2~C5 멀티스케일 피처를 반환하는 ResNet 백본.

    freeze_at: Detectron 관례(FPN 논문/Mask R-CNN 구현체 기본값)대로 stem과
        초반 stage는 학습하지 않는다. requires_grad=False로 두면 그 구간은
        autograd가 activation을 저장할 필요가 없어져(모든 입력이 grad를
        요구하지 않으므로) 메모리를 크게 아낀다 — 특히 layer1(C2)은 stride 4로
        해상도가 가장 커서 activation 메모리 비중이 가장 크다.
            0: 고정 없음, 1: stem만, 2: stem+layer1(기본, Detectron 기본값),
            3: stem+layer1+layer2, 4: +layer3, 5: 전체 고정
    grad_checkpoint: True면 layer1~layer4를 torch.utils.checkpoint로 감싸
        forward activation을 저장하지 않고 backward 시 재계산한다(연산량↑,
        메모리↓). 고정된(freeze) stage는 어차피 activation을 안 남기므로
        checkpoint 대상에서 제외한다.
    """

    def __init__(self, stage_blocks: List[int], freeze_at: int = 2,
                 grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint

        # stem: conv1
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = FrozenBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_stage(64, 64, stage_blocks[0], stride=1)      # C2
        self.layer2 = self._make_stage(256, 128, stage_blocks[1], stride=2)    # C3
        self.layer3 = self._make_stage(512, 256, stage_blocks[2], stride=2)    # C4
        self.layer4 = self._make_stage(1024, 512, stage_blocks[3], stride=2)   # C5

        # C2~C5 채널 수 (FPN lateral 연결에 필요)
        self.out_channels = [256, 512, 1024, 2048]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        self._freeze_stages(freeze_at)

    def _freeze_stages(self, freeze_at: int):
        stem = [self.conv1, self.bn1]
        stages = [self.layer1, self.layer2, self.layer3, self.layer4]
        modules_to_freeze = []
        if freeze_at >= 1:
            modules_to_freeze += stem
        modules_to_freeze += stages[: max(freeze_at - 1, 0)]
        for m in modules_to_freeze:
            for p in m.parameters():
                p.requires_grad = False

    @staticmethod
    def _make_stage(in_channels: int, bottleneck_channels: int,
                    num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [Bottleneck(in_channels, bottleneck_channels, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(
                Bottleneck(bottleneck_channels * Bottleneck.expansion,
                           bottleneck_channels)
            )
        return nn.Sequential(*blocks)

    def _run_stage(self, stage: nn.Sequential, x: Tensor) -> Tensor:
        # 고정된 stage는 어차피 grad를 안 남기니 checkpoint가 무의미하다.
        needs_grad = any(p.requires_grad for p in stage.parameters())
        if self.grad_checkpoint and self.training and needs_grad:
            return checkpoint(stage, x, use_reentrant=False)
        return stage(x)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self._run_stage(self.layer1, x)
        c3 = self._run_stage(self.layer2, c2)
        c4 = self._run_stage(self.layer3, c3)
        c5 = self._run_stage(self.layer4, c4)
        return {"c2": c2, "c3": c3, "c4": c4, "c5": c5}


def resnet50(freeze_at: int = 2, grad_checkpoint: bool = False) -> ResNet:
    return ResNet([3, 4, 6, 3], freeze_at=freeze_at, grad_checkpoint=grad_checkpoint)


def resnet101(freeze_at: int = 2, grad_checkpoint: bool = False) -> ResNet:
    return ResNet([3, 4, 23, 3], freeze_at=freeze_at, grad_checkpoint=grad_checkpoint)
