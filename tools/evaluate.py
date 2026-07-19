"""체크포인트로 COCO mAP 평가 (bbox + segm). pycocotools 필요.

사용 예:

    python tools/evaluate.py \
        --checkpoint checkpoints/maskrcnn_epoch11.pth \
        --val-images data/coco/val2017 \
        --val-ann    data/coco/annotations/instances_val2017.json \
        --max-images 500

`--max-images`로 일부만 빠르게 평가할 수 있다 (생략 시 전체).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from maskrcnn.data import CocoInstanceDataset
from maskrcnn.evaluate import evaluate_coco, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Mask R-CNN COCO mAP 평가")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--val-ann", required=True)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--iou-types", nargs="+", default=["bbox", "segm"],
                   choices=["bbox", "segm"])
    p.add_argument("--device", default=None)
    return p.parse_args()


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    model, _ = load_checkpoint(args.checkpoint, device)
    dataset = CocoInstanceDataset(
        args.val_images, args.val_ann, contiguous_ids=True,
        min_size=args.min_size, max_size=args.max_size, skip_empty=False)
    print(f"평가 이미지 {len(dataset)}장"
          + (f" (앞 {args.max_images}장만)" if args.max_images else ""))

    metrics = evaluate_coco(
        model, dataset, device, iou_types=tuple(args.iou_types),
        max_images=args.max_images, min_size=args.min_size,
        max_size=args.max_size)

    print("\n===== 요약 =====")
    for k, v in metrics.items():
        print(f"  {k:12s} = {v:.4f}")


if __name__ == "__main__":
    main()
