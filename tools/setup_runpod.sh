#!/usr/bin/env bash
# RunPod(단일 GPU, 예: L40S) 초기 셋업: 의존성 설치 + COCO 2017 전체 다운로드.
#
# 전제: RunPod의 PyTorch 템플릿을 쓴다는 가정이라 torch/torchvision은 이미 설치돼
# 있다고 보고 CUDA 버전 지정 없이 이 리포가 추가로 필요로 하는 패키지만 설치한다.
# 다른 템플릿(순정 Ubuntu 등)이면 먼저 CUDA 버전에 맞는 torch부터 설치해야 한다
# (https://pytorch.org/get-started/locally/ 참고).
#
# 사용:
#   bash tools/setup_runpod.sh [DATA_DIR]
# DATA_DIR 기본값은 data/coco (리포 루트 기준).
#
# COCO train2017(~19GB) + val2017(~1GB) + annotations(~250MB) 다운로드/압축
# 해제라 디스크 40GB+ 여유를 확인하고 돌린다. 시간도 꽤 걸린다(네트워크 속도에
# 따라 다르지만 보통 10~30분).

set -euo pipefail

DATA_DIR="${1:-data/coco}"
mkdir -p "$DATA_DIR"

echo "=== 1) 파이썬 의존성 (torch/torchvision은 이미 있다고 가정) ==="
python3 -c "import torch" 2>/dev/null && echo "  torch 확인됨: $(python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())')" \
    || { echo "  [경고] torch가 없다 — https://pytorch.org/get-started/locally/ 에서 먼저 설치하라"; }
pip install --upgrade pip
pip install pycocotools matplotlib pillow numpy

echo ""
echo "=== 2) COCO 2017 다운로드 -> $DATA_DIR ==="
cd "$DATA_DIR"
for f in train2017.zip val2017.zip annotations_trainval2017.zip; do
    if [ -f "$f" ]; then
        echo "  $f 이미 있음, 건너뜀"
    else
        case "$f" in
            annotations_trainval2017.zip) url="http://images.cocodataset.org/annotations/$f" ;;
            *) url="http://images.cocodataset.org/zips/$f" ;;
        esac
        echo "  받는 중: $url"
        wget -c -q --show-progress "$url"
    fi
done

echo ""
echo "=== 3) 압축 해제 ==="
for f in train2017.zip val2017.zip annotations_trainval2017.zip; do
    unzip -q -o "$f"
done
cd - >/dev/null

echo ""
echo "=== 완료 ==="
echo "이미지: $DATA_DIR/train2017, $DATA_DIR/val2017"
echo "annotation: $DATA_DIR/annotations/instances_{train,val}2017.json"
echo ""
echo "학습 실행 예시 (README 또는 채팅에서 준 튜닝 커맨드 참고):"
echo "  python3 tools/train.py --train-images $DATA_DIR/train2017 \\"
echo "      --train-ann $DATA_DIR/annotations/instances_train2017.json \\"
echo "      --num-classes 81 --pretrained --amp ..."
