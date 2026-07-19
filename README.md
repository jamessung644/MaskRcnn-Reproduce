# Mask R-CNN 리프로덕션 (PyTorch)

He et al., *Mask R-CNN* (ICCV 2017) + Lin et al., *Feature Pyramid Networks* (CVPR 2017)
논문 기반 구현. 백본은 ResNet-50-FPN.

## 구조

```
maskrcnn/
├── config.py              # 논문/Detectron 하이퍼파라미터 (출처 주석 포함)
├── data/
│   ├── coco.py            # COCO 포맷 데이터셋 (커스텀 데이터셋도 COCO 포맷이면 재사용)
│   └── transforms.py      # 리사이즈(800/1333) + ImageNet 정규화 + 배치 패딩
├── model/
│   ├── resnet.py          # ResNet-50/101 백본 (C2~C5, FrozenBN)
│   ├── fpn.py             # FPN (P2~P6, d=256)
│   ├── anchors.py         # 레벨당 단일 스케일 {32..512} × 종횡비 {0.5,1,2}
│   ├── rpn.py             # RPN 헤드(레벨 공유) + proposal 생성 + RPN 손실
│   ├── pooler.py          # RoIAlign + FPN 레벨 할당 k=⌊4+log₂(√wh/224)⌋
│   ├── heads.py           # Box head (2×FC 1024), Mask head (4×conv → deconv → 28×28)
│   ├── matcher.py         # IoU 매처 + 균형 샘플러 + smooth_L1 (학습 타깃)
│   └── mask_rcnn.py       # 전체 조립 + 추론 후처리 + 학습(RoI 샘플링/손실)
├── evaluate.py            # 체크포인트 로드 + 원본좌표 예측 + COCO mAP 평가
├── utils/
│   ├── box_ops.py         # 박스 인코딩/디코딩, IoU, clip
│   ├── tv_weights.py      # torchvision 사전학습 가중치 매핑 (전체/부분 로드)
│   └── visualize.py       # 마스크 페이스트 + detection 시각화
tools/
├── train.py               # SGD 학습 루프 (warmup + step decay, 체크포인트, 선택적 mAP)
├── evaluate.py            # 체크포인트 -> COCO mAP (bbox/segm)
└── infer.py               # 체크포인트 -> 추론 -> detection 시각화 저장
tests/
├── smoke_test.py          # 추론 경로 shape 검증
├── train_smoke_test.py    # 학습 경로 검증 (합성 타깃)
└── test_coco.py           # COCO 데이터 + 사전학습 가중치 + 레퍼런스 비교
```

## 논문 대응표

| 구성 요소 | 논문 스펙 | 구현 위치 |
|---|---|---|
| 백본 | ResNet-50, C2~C5 (stride 4/8/16/32) | `resnet.py` |
| FPN | lateral 1×1 + top-down 2× + 3×3, P6=maxpool(P5) | `fpn.py` |
| 앵커 | P2~P6에 {32²..512²}, 비율 {1:2,1:1,2:1} | `anchors.py` |
| RPN | 3×3 conv + 1×1 ×2, 전 레벨 파라미터 공유 | `rpn.py` |
| RoIAlign | bilinear 샘플링, box 7×7 / mask 14×14 | `pooler.py` |
| Box head | 2× FC 1024 → (K+1) cls + 클래스별 4 box | `heads.py` |
| Mask head | 4× conv256 + deconv → 1×1 → K개 28×28, sigmoid | `heads.py` |
| 추론 | proposal 1000 → detection 100 → 마스크는 100개에만 | `mask_rcnn.py` |

## 실행

### 테스트

```bash
python tests/smoke_test.py        # 추론 경로 shape 검증 (랜덤 가중치)
python tests/train_smoke_test.py  # 학습 경로 검증 (합성 타깃, 데이터 불필요)
python tests/test_coco.py         # COCO 데이터 + 사전학습 가중치 검증
```

`test_coco.py`는 `data/coco/` 아래에 instances_val2017.json과 val2017 샘플
이미지가 필요하다 (http://images.cocodataset.org 에서 다운로드).

### 학습

COCO 포맷 데이터셋(커스텀 데이터도 변환하면 재사용 가능)으로 학습한다:

```bash
python tools/train.py \
    --train-images data/lung/train \
    --train-ann    data/lung/annotations/train.json \
    --num-classes  3 \          # 배경 포함 (전경 K개면 K+1)
    --epochs 20 --batch-size 2 --lr 0.005 \
    --pretrained                # COCO 백본/FPN/RPN 가중치로 초기화(권장)
```

- 손실: `L = L_rpn(obj+box) + L_cls + L_box + L_mask` 를 모두 합산해 역전파.
- optimizer: SGD(momentum 0.9, weight decay 1e-4) + 선형 warmup + step decay.
- `--pretrained`는 클래스 독립 가중치(백본·FPN·RPN·head conv)만 옮기고
  클래스 의존 predictor는 무작위 초기화로 남긴다 → 파인튜닝에 적합.
- polygon segmentation은 의존성 없이 동작. RLE 마스크는 `pycocotools` 필요.
- 체크포인트는 `checkpoints/`에 epoch마다 저장(`--resume`으로 재개).
- 출력: 설정 배너 + iteration별 항목 손실(이동평균)·lr·속도·ETA + epoch 요약.

**epoch마다 검증 mAP를 함께 보려면** (`pycocotools` 필요):

```bash
python tools/train.py ... \
    --eval-images data/coco/val2017 \
    --eval-ann    data/coco/annotations/instances_val2017.json \
    --eval-interval 1 --eval-max-images 500   # 빠른 확인용 500장만
```

### 평가 (COCO mAP)

```bash
python tools/evaluate.py \
    --checkpoint  checkpoints/maskrcnn_epoch11.pth \
    --val-images  data/coco/val2017 \
    --val-ann     data/coco/annotations/instances_val2017.json
# bbox / segm 각각 AP, AP50, AP75 출력 (pycocotools COCOeval)
```

### 추론 + 시각화

```bash
python tools/infer.py \
    --checkpoint checkpoints/maskrcnn_epoch11.pth \
    --images data/coco/val2017 \
    --output outputs/ --max-images 20 --score-thresh 0.5
# 박스+라벨+마스크 오버레이를 outputs/det_*.jpg 로 저장
```

## 검증 결과

- torchvision COCO 사전학습 가중치 **307개 텐서 전부 1:1 매핑** 성공
  → 구조가 레퍼런스 구현과 레이어 단위로 동형
- 실제 COCO 이미지에서 torchvision 레퍼런스 detection과 **98% 일치**
  (IoU>0.5, 동일 라벨 기준, score>0.5인 144개 중 141개)
- 시각화 결과: `outputs/det_*.jpg`

주의: torchvision 가중치 사용 시 `Config(num_classes=91, anchor_offset=0.0)`
(원본 COCO id 체계 + torchvision 앵커 배치 관례).

## 현재 상태 / 다음 단계

- [x] 전체 아키텍처 + 추론 경로 (44.3M 파라미터)
- [x] COCO 데이터셋 로더 + 전처리 + 사전학습 가중치 검증
- [x] 학습 경로: RPN/RoI 타깃 매칭·샘플링 (RPN 256 anchors 1:1, RoI 512개 25% positive)
      — `model/matcher.py`, `model/rpn.py`, `model/mask_rcnn.py`
- [x] 손실 함수: L = L_cls + L_box + L_mask (mask는 GT 클래스 채널만 per-pixel BCE)
- [x] 학습 스크립트 (SGD lr 0.02, warmup + ×0.1 스케줄) — `tools/train.py`
- [ ] 폐 X-ray 데이터셋 → COCO 포맷 변환 후 `CocoInstanceDataset` 재사용
- [ ] 검증셋 mAP 평가 루프 (현재는 학습 손실만)
