"""Mask R-CNN 학습 스크립트 (SGD + warmup + step decay).

논문 3.1절 / Detectron 관례를 따른다:
    - optimizer: SGD (momentum 0.9, weight decay 1e-4)
    - lr: 초반 warmup 후 milestone에서 x0.1
    - loss: RPN(obj+box) + RoI(cls+box) + mask, 전부 합산해 역전파

사용 예 (COCO 포맷 커스텀 데이터, 폐 X-ray 등):

    python tools/train.py \
        --train-images data/lung/train \
        --train-ann    data/lung/annotations/train.json \
        --num-classes  3 \
        --epochs 20 --batch-size 2 --lr 0.005 \
        --pretrained            # COCO 백본/FPN/RPN 가중치로 초기화(권장)

`--num-classes`는 배경을 포함한 값이다 (전경 클래스 K개면 K+1).
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from maskrcnn import Config, MaskRCNN
from maskrcnn.data import CocoInstanceDataset, collate_fn


def parse_args():
    p = argparse.ArgumentParser(description="Mask R-CNN 학습")
    p.add_argument("--train-images", required=True, help="학습 이미지 디렉토리")
    p.add_argument("--train-ann", required=True, help="COCO 포맷 annotation JSON")
    p.add_argument("--num-classes", type=int, required=True,
                   help="배경 포함 클래스 수 (전경 K개면 K+1)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=None,
                   help="기본값: Config.learning_rate")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--milestones", type=float, nargs="*", default=[0.7, 0.9],
                   help="전체 epoch 대비 lr 감쇠 시점 (기본 70%%, 90%%)")
    p.add_argument("--pretrained", action="store_true",
                   help="COCO 사전학습 가중치로 클래스 독립 파라미터 초기화")
    p.add_argument("--output", default="checkpoints", help="체크포인트 저장 경로")
    p.add_argument("--resume", default=None, help="이어서 학습할 체크포인트")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--device", default=None, help="cuda/mps/cpu (기본 자동)")
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_targets(targets, device):
    out = []
    for t in targets:
        out.append({k: v.to(device) for k, v in t.items()})
    return out


def main():
    args = parse_args()
    device = pick_device(args.device)
    print(f"device: {device}")

    cfg = Config(num_classes=args.num_classes)
    if args.lr is not None:
        cfg.learning_rate = args.lr

    # ---- 데이터
    dataset = CocoInstanceDataset(
        args.train_images, args.train_ann,
        contiguous_ids=True, min_size=args.min_size, max_size=args.max_size)
    print(f"학습 이미지 {len(dataset)}장, 전경 클래스 {len(dataset.label_to_name)}개")
    if args.num_classes != len(dataset.label_to_name) + 1:
        print(f"  [경고] --num-classes={args.num_classes} 이지만 데이터셋 클래스는 "
              f"{len(dataset.label_to_name)}개다 (배경 포함 {len(dataset.label_to_name)+1} 권장).")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=collate_fn, drop_last=True)

    # ---- 모델
    model = MaskRCNN(cfg)
    if args.pretrained:
        from maskrcnn.utils.tv_weights import load_torchvision_pretrained
        stats = load_torchvision_pretrained(model)
        print(f"사전학습 가중치 로드: {stats['loaded']}개 텐서 "
              f"(무작위 초기화 유지 {stats['kept']}개 — 클래스 의존 predictor)")
    model.to(device).train()

    # ---- optimizer / 스케줄
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.learning_rate, momentum=cfg.momentum,
        weight_decay=cfg.weight_decay)

    iters_per_epoch = max(len(loader), 1)
    total_iters = args.epochs * iters_per_epoch
    milestones = sorted(int(m * total_iters) for m in args.milestones)

    def lr_at(it):
        # warmup (선형) 후 milestone마다 x0.1
        if it < cfg.warmup_iters:
            alpha = it / max(cfg.warmup_iters, 1)
            factor = cfg.warmup_factor * (1 - alpha) + alpha
        else:
            factor = 1.0
        for m in milestones:
            if it >= m:
                factor *= 0.1
        return factor

    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", start_epoch * iters_per_epoch)
        print(f"체크포인트에서 재개: epoch {start_epoch}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 학습 루프
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        for i, (images, image_sizes, targets) in enumerate(loader):
            images = images.to(device)
            targets = move_targets(targets, device)

            for g in optimizer.param_groups:
                g["lr"] = cfg.learning_rate * lr_at(global_step)

            losses = model(images, image_sizes, targets)
            loss = sum(losses.values())

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            if i % args.log_interval == 0:
                parts = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items())
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[epoch {epoch} it {i}/{iters_per_epoch}] "
                      f"loss={loss.item():.3f} lr={lr_now:.5f} | {parts}")
            global_step += 1

        ckpt_path = out_dir / f"maskrcnn_epoch{epoch}.pth"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "num_classes": args.num_classes,
            "label_to_name": dataset.label_to_name,
        }, ckpt_path)
        print(f"epoch {epoch} 완료 ({time.time() - epoch_start:.0f}s) "
              f"-> {ckpt_path}")

    print("학습 종료 [OK]")


if __name__ == "__main__":
    main()
