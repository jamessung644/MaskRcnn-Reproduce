"""Mask R-CNN 하이퍼파라미터 (He et al., ICCV 2017 / Lin et al., CVPR 2017 기준).

논문에 명시된 값은 주석에 출처(섹션)를 표기했다. 논문에 수치가 없는 항목은
공식 구현(Detectron)의 기본값을 따르고 그렇게 표기했다.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ 데이터
    num_classes: int = 81  # COCO: 80 클래스 + 배경

    # ---------------------------------------------------------------- 백본/FPN
    # FPN 논문 3절: 모든 피라미드 레벨의 채널 수 d = 256
    fpn_out_channels: int = 256

    # ------------------------------------------------------------------ 앵커
    # FPN 논문 4.1절: 레벨당 단일 스케일 {32^2, ..., 512^2} (P2~P6),
    # 종횡비 {1:2, 1:1, 2:1} -> 레벨당 위치마다 앵커 3개
    anchor_sizes: Tuple[int, ...] = (32, 64, 128, 256, 512)
    anchor_ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)
    # P2~P6의 stride (입력 대비 다운샘플 배율)
    feature_strides: Tuple[int, ...] = (4, 8, 16, 32, 64)
    # 앵커 중심 오프셋 (셀 크기 대비). 0.5 = 픽셀 중심(Detectron2 방식).
    # torchvision 사전학습 가중치와 맞출 때는 0.0 사용.
    anchor_offset: float = 0.5

    # ------------------------------------------------------------------ RPN
    # Faster R-CNN 논문: NMS IoU 0.7
    rpn_nms_thresh: float = 0.7
    # FPN 기반 proposal 수. 원래 Detectron/논문 기본값은 train 2000/2000,
    # test 1000/1000이었으나, 여기서는 16GB급 GPU x2(Ada 2000) 환경에서
    # 처리량을 우선하는 경량화 프로파일로 절반 수준으로 낮췄다 — box/mask
    # head가 RoI마다 연산하므로 proposal 수를 줄이면 학습/추론 모두
    # 직접적으로 빨라진다. 정확도를 원래 수준으로 맞추고 싶으면 주석의
    # 원래 값으로 되돌리면 된다.
    rpn_pre_nms_top_n_train: int = 1000   # 레벨당 (원래 2000)
    rpn_pre_nms_top_n_test: int = 500     # 레벨당 (원래 1000)
    rpn_post_nms_top_n_train: int = 1000  # 이미지당 (원래 2000, Mask R-CNN 논문 4.1절)
    rpn_post_nms_top_n_test: int = 500    # 이미지당 (원래 1000, FPN 논문 4.1절)
    rpn_min_size: float = 0.0  # 이 크기보다 작은 proposal 제거

    # -------------------------------------------------- RPN 학습 타깃/샘플링
    # Faster R-CNN 논문 3.1.2절: anchor의 GT IoU >= 0.7이면 positive,
    # < 0.3이면 negative, 그 사이는 학습에서 무시한다.
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    # Faster R-CNN 논문 3.1.3절: 이미지당 256개 anchor를 positive:negative
    # = 1:1 로 샘플링한다 (positive가 부족하면 negative로 채운다).
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5

    # ---------------------------------------------------------------- RoIAlign
    # Mask R-CNN 논문 3절: box head는 7x7, mask head는 14x14
    box_roi_pool_size: int = 7
    mask_roi_pool_size: int = 14
    roi_sampling_ratio: int = 2  # bin당 샘플링 포인트 (논문 3절: 4점 = 2x2)
    # FPN 논문 식(1): k = floor(k0 + log2(sqrt(wh)/224)), k0 = 4
    roi_canonical_scale: int = 224
    roi_canonical_level: int = 4
    roi_min_level: int = 2  # P2
    roi_max_level: int = 5  # P5 (P6는 RPN 전용, FPN 논문 4.1절)

    # ------------------------------------------------------------- Box head
    # FPN 논문 4.2절: 2개의 hidden 1024-d FC
    box_head_fc_dim: int = 1024

    # ------------------------------------------------------------ Mask head
    # Mask R-CNN 논문 그림 4(우) 원래 값: conv 4개(256ch) + deconv + 1x1, 출력 28x28.
    # 여기서는 경량화 프로파일로 conv 2개(128ch)로 줄였다 — mask branch는
    # 14x14 해상도에서 conv를 여러 번 돌리는 부분이라 연산 비중이 커서,
    # 줄이면 처리량에 직접적인 이득이 있다(대신 마스크 품질은 다소 낮아질
    # 수 있다). 정확도를 우선하면 4/256으로 되돌리면 된다.
    mask_head_num_convs: int = 2   # 원래 4
    mask_head_conv_dim: int = 128  # 원래 256
    mask_resolution: int = 28

    # ------------------------------------------------------- 추론 후처리
    # Detectron 기본값: score 0.05로 거르고 클래스별 NMS 0.5, 상위 100개 유지
    # (mask branch를 상위 100개 detection에 적용하는 것은 논문 3.1절 'Inference')
    score_thresh: float = 0.05
    detections_nms_thresh: float = 0.5
    detections_per_img: int = 100

    # -------------------------------------------------- RoI 학습 타깃/샘플링
    # Fast R-CNN 논문 / Mask R-CNN 3.1절: proposal의 GT IoU >= 0.5면
    # foreground(해당 GT 클래스), < 0.5면 background(클래스 0)로 둔다.
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    # Mask R-CNN 논문 3.1절 원래 값: 이미지당 512개 RoI를 positive 비율
    # 25%로 샘플링. 경량화 프로파일에서는 256개로 줄였다 — box/mask head가
    # 스텝마다 처리하는 RoI 수가 절반이 되어 iteration이 빨라진다(수렴에
    # 필요한 step 수가 다소 늘 수 있는 트레이드오프).
    box_batch_size_per_image: int = 256  # 원래 512
    box_positive_fraction: float = 0.25

    # ------------------------------------------------------- 학습 스케줄
    # Mask R-CNN 논문 3.1절/Detectron 1x 스케줄 (8-GPU, 이미지 16장 기준).
    # 단일 GPU/작은 배치에서는 lr을 선형으로 낮춰 쓰는 것을 권장한다.
    learning_rate: float = 0.02
    momentum: float = 0.9
    weight_decay: float = 1e-4
    # warmup (Detectron): 초반 lr을 선형으로 끌어올려 발산을 막는다.
    warmup_iters: int = 500
    warmup_factor: float = 1.0 / 3

    # ------------------------------------------- 박스 회귀 인코딩 가중치
    # (Detectron 기본값; 논문은 Fast/Faster R-CNN의 파라미터화를 그대로 사용)
    rpn_box_reg_weights: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    box_reg_weights: Tuple[float, ...] = (10.0, 10.0, 5.0, 5.0)
