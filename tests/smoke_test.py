"""전체 파이프라인 스모크 테스트.

각 단계의 출력 shape이 논문 스펙과 일치하는지 확인한다.
  - ResNet-50: C2~C5 채널 256/512/1024/2048, stride 4/8/16/32
  - FPN: P2~P6 전부 256채널, stride 4/8/16/32/64
  - RPN: 이미지당 proposal <= 1000 (test 모드)
  - Box/Mask branch: detection <= 100, mask는 (N, 28, 28)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from maskrcnn import Config, MaskRCNN


def main():
    torch.manual_seed(0)
    cfg = Config()
    model = MaskRCNN(cfg).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {n_params / 1e6:.1f}M")

    images = torch.randn(2, 3, 512, 640)

    with torch.no_grad():
        # ---- 백본 / FPN shape 확인
        c_feats = model.backbone(images)
        for name, ch, stride in zip("c2 c3 c4 c5".split(),
                                    (256, 512, 1024, 2048), (4, 8, 16, 32)):
            f = c_feats[name]
            assert f.shape[1] == ch, f"{name} 채널 {f.shape[1]} != {ch}"
            assert f.shape[2] == 512 // stride, f"{name} stride 불일치"
            print(f"{name.upper()}: {tuple(f.shape)}")

        p_feats = model.fpn(c_feats)
        for name, stride in zip("p2 p3 p4 p5 p6".split(), (4, 8, 16, 32, 64)):
            f = p_feats[name]
            assert f.shape[1] == 256, f"{name} 채널 {f.shape[1]} != 256"
            assert f.shape[2] == 512 // stride, f"{name} stride 불일치"
            print(f"{name.upper()}: {tuple(f.shape)}")

        # ---- RPN proposal
        image_sizes = [(512, 640)] * 2
        proposals, _ = model.rpn(p_feats, image_sizes)
        for i, p in enumerate(proposals):
            assert p.shape[0] <= cfg.rpn_post_nms_top_n_test
            print(f"이미지 {i} proposals: {tuple(p.shape)}")

        # ---- 전체 forward
        detections = model(images, image_sizes)
        for i, det in enumerate(detections):
            n = det["boxes"].shape[0]
            assert n <= cfg.detections_per_img
            assert det["masks"].shape[1:] == (28, 28)
            print(f"이미지 {i}: boxes {tuple(det['boxes'].shape)}, "
                  f"labels {tuple(det['labels'].shape)}, "
                  f"masks {tuple(det['masks'].shape)}")

    print("\n스모크 테스트 통과 [OK]")


if __name__ == "__main__":
    main()
