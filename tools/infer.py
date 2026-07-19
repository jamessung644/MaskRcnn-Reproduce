"""체크포인트로 추론 -> detection 시각화 이미지 저장.

사용 예:

    python tools/infer.py \
        --checkpoint checkpoints/maskrcnn_epoch11.pth \
        --images data/coco/val2017 \
        --output outputs/ \
        --max-images 20 --score-thresh 0.5

`--images`는 디렉토리(하위 이미지 전부) 또는 이미지 파일 하나를 받는다.
클래스 이름은 체크포인트에 저장된 label_to_name을 사용한다.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from maskrcnn.evaluate import load_checkpoint, predict_original
from maskrcnn.utils.visualize import draw_detections

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser(description="Mask R-CNN 추론 + 시각화")
    p.add_argument("--checkpoint", required=True, help="train.py 체크포인트 (.pth)")
    p.add_argument("--images", required=True, help="이미지 디렉토리 또는 파일")
    p.add_argument("--output", default="outputs", help="시각화 저장 경로")
    p.add_argument("--score-thresh", type=float, default=0.5)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--device", default=None, help="cuda/mps/cpu (기본 자동)")
    return p.parse_args()


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gather_images(path: Path):
    if path.is_file():
        return [path]
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)
    return files


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    model, label_to_name = load_checkpoint(args.checkpoint, device)
    print(f"체크포인트 로드: {args.checkpoint} (클래스 {len(label_to_name)}개)")

    image_paths = gather_images(Path(args.images))
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        print("이미지를 찾지 못했다.")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        pil = Image.open(img_path).convert("RGB")
        det = predict_original(model, pil, device, args.min_size, args.max_size)
        n = int((det["scores"] > args.score_thresh).sum())
        vis = draw_detections(pil, det, label_to_name,
                              score_thresh=args.score_thresh)
        out_path = out_dir / f"det_{img_path.stem}.jpg"
        vis.save(out_path)
        print(f"  [{i + 1}/{len(image_paths)}] {img_path.name}: "
              f"검출 {n}개 -> {out_path}", flush=True)

    print(f"\n시각화 {len(image_paths)}장 저장 완료 -> {out_dir}")


if __name__ == "__main__":
    main()
