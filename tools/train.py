"""Mask R-CNN 학습 스크립트 (SGD + warmup + step decay).

논문 3.1절 / Detectron 관례를 따른다:
    - optimizer: SGD (momentum 0.9, weight decay 1e-4)
    - lr: 초반 warmup 후 milestone에서 x0.1
    - loss: RPN(obj+box) + RoI(cls+box) + mask, 전부 합산해 역전파

사용 예 1) COCO (80 클래스 -> 배경 포함 81):

    python tools/train.py \
        --train-images data/coco/val2017 \
        --train-ann    data/coco/annotations/instances_val2017.json \
        --num-classes  81 \
        --epochs 12 --batch-size 2 --lr 0.005 \
        --pretrained            # COCO 백본/FPN/RPN 가중치로 초기화(권장)

사용 예 2) 커스텀 데이터 (폐 X-ray 등, 전경 2 클래스 -> 배경 포함 3):

    python tools/train.py \
        --train-images data/lung/train \
        --train-ann    data/lung/annotations/train.json \
        --num-classes  3 \
        --epochs 20 --batch-size 2 --lr 0.005 --pretrained

`--num-classes`는 배경을 포함한 값이다 (전경 클래스 K개면 K+1).
"""

import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from maskrcnn import Config, MaskRCNN
from maskrcnn.data import CocoInstanceDataset, collate_fn


class SmoothedValue:
    """최근 window개의 이동평균과 전체 평균을 함께 추적한다."""

    def __init__(self, window: int = 50):
        self.deque = deque(maxlen=window)
        self.total = 0.0
        self.count = 0

    def update(self, value: float):
        self.deque.append(value)
        self.total += value
        self.count += 1

    @property
    def avg(self) -> float:
        return sum(self.deque) / max(len(self.deque), 1)

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)


def format_time(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


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
    # ---- 선택: epoch마다 val mAP 평가 (pycocotools 필요)
    p.add_argument("--eval-images", default=None, help="검증 이미지 디렉토리")
    p.add_argument("--eval-ann", default=None, help="검증 annotation JSON")
    p.add_argument("--eval-interval", type=int, default=1,
                   help="몇 epoch마다 평가할지 (기본 1)")
    p.add_argument("--eval-max-images", type=int, default=None,
                   help="평가에 쓸 이미지 수 제한 (빠른 확인용)")
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

    # ---- 선택: 검증셋 (epoch마다 mAP 평가)
    eval_dataset = None
    if args.eval_images and args.eval_ann:
        eval_dataset = CocoInstanceDataset(
            args.eval_images, args.eval_ann, contiguous_ids=True,
            min_size=args.min_size, max_size=args.max_size, skip_empty=False)
        print(f"검증 이미지 {len(eval_dataset)}장 "
              f"(epoch {args.eval_interval}마다 mAP 평가)")

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

    # ---- 설정 배너
    print("=" * 66)
    print("  Mask R-CNN 학습 설정")
    print("-" * 66)
    print(f"  device            : {device}")
    print(f"  학습 이미지       : {len(dataset)} 장")
    print(f"  클래스(배경 포함) : {args.num_classes}")
    print(f"  epochs            : {args.epochs}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  iters / epoch     : {iters_per_epoch}")
    print(f"  총 iterations     : {total_iters}")
    print(f"  base lr           : {cfg.learning_rate}")
    print(f"  lr 감쇠 시점(iter): {milestones}  (x0.1)")
    print(f"  warmup iters      : {cfg.warmup_iters}")
    print(f"  입력 해상도       : min {args.min_size} / max {args.max_size}")
    print(f"  pretrained init   : {args.pretrained}")
    print(f"  grad clip         : {args.grad_clip}")
    print(f"  체크포인트 경로   : {out_dir}")
    print("=" * 66, flush=True)

    # ---- 학습 루프
    iter_time = SmoothedValue(window=50)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        meters = defaultdict(SmoothedValue)  # epoch별 손실 이동평균
        epoch_start = time.time()

        for i, (images, image_sizes, targets) in enumerate(loader):
            step_start = time.time()
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

            # 손실/시간 집계
            meters["loss"].update(loss.item())
            for k, v in losses.items():
                meters[k].update(v.item())
            iter_time.update(time.time() - step_start)
            global_step += 1

            if i % args.log_interval == 0 or i == iters_per_epoch - 1:
                lr_now = optimizer.param_groups[0]["lr"]
                eta = format_time(iter_time.avg * (total_iters - global_step))
                mem = ""
                if device.type == "cuda":
                    mem = f" mem={torch.cuda.max_memory_allocated() / 1e9:.1f}G"
                print(
                    f"E{epoch}/{args.epochs} "
                    f"[{i:>4}/{iters_per_epoch}] "
                    f"step {global_step}/{total_iters}  "
                    f"loss={meters['loss'].avg:.3f}  "
                    f"cls={meters['loss_box_cls'].avg:.3f} "
                    f"box={meters['loss_box_reg'].avg:.3f} "
                    f"mask={meters['loss_mask'].avg:.3f} "
                    f"rpn_obj={meters['rpn_objectness'].avg:.3f} "
                    f"rpn_box={meters['rpn_box_reg'].avg:.3f}  "
                    f"lr={lr_now:.5f}  {iter_time.avg:.2f}s/it  eta {eta}{mem}",
                    flush=True,
                )

        ckpt_path = out_dir / f"maskrcnn_epoch{epoch}.pth"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "num_classes": args.num_classes,
            "label_to_name": dataset.label_to_name,
        }, ckpt_path)
        epoch_time = time.time() - epoch_start
        print("-" * 66)
        print(
            f"[epoch {epoch} 완료] "
            f"avg_loss={meters['loss'].global_avg:.3f}  "
            f"time={format_time(epoch_time)}  "
            f"-> {ckpt_path}"
        )
        print("-" * 66, flush=True)

        # ---- epoch마다 검증셋 mAP 평가 (선택)
        if eval_dataset is not None and (epoch + 1) % args.eval_interval == 0:
            from maskrcnn.evaluate import evaluate_coco
            print(f"[epoch {epoch}] 검증셋 평가 중...", flush=True)
            model.eval()
            metrics = evaluate_coco(
                model, eval_dataset, device,
                max_images=args.eval_max_images,
                min_size=args.min_size, max_size=args.max_size,
                verbose=False)
            summary = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
            print(f"[epoch {epoch} mAP] {summary}", flush=True)
            model.train()

    print("학습 종료 [OK]")


if __name__ == "__main__":
    main()
