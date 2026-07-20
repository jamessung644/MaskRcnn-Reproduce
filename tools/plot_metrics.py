"""체크포인트 -> Precision/Recall/F1 + AP breakdown 성능지표 시각화 (COCOeval 기반).

tools/evaluate.py가 뽑는 AP/AP50/AP75는 스칼라 요약값이라 "어느 정도
confidence threshold에서 precision/recall 균형이 가장 좋은가", "클래스별로는
어디가 약한가" 같은 걸 안 보여준다. 이 스크립트는 COCOeval이 이미 계산해 둔
카테고리별 precision-recall-score 배열을 재활용해 두 장을 저장한다:

    <output>            : bbox/segm 각각 Precision-Recall curve + F1 curve
                          (best F1 지점과 근사 confidence threshold 표시)
    <output>_breakdown  : (1) AP/AP50/AP75/크기별 AP/AR@{1,10,100} 등
                          COCOeval 표준 12개 지표 막대그래프
                          (2) 클래스별 AP50 막대그래프 (약한 클래스 바로 확인)

matplotlib, pycocotools가 필요하다.

사용 예:
    python tools/plot_metrics.py \
        --checkpoint checkpoints/maskrcnn_epoch11.pth \
        --val-images data/coco/val2017 \
        --val-ann    data/coco/annotations/instances_val2017.json \
        --output outputs/metrics.png --max-images 500
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # 헤드리스 서버에서도 그리기만 하고 파일로 저장
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import torch

# 커스텀 데이터셋 카테고리 이름이 한글일 수 있어서(예: 폐 X-ray 프로젝트),
# 서버에 CJK 폰트가 있으면 그걸 쓴다 — 없으면 조용히 기본 폰트로 넘어간다
# (그 경우 한글 라벨만 네모(tofu)로 깨지고 그래프 자체는 정상 출력된다).
for _cjk in ("NanumGothic", "Noto Sans CJK KR", "Noto Sans KR", "Malgun Gothic", "AppleGothic"):
    if any(_cjk.lower() in f.name.lower() for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _cjk
        plt.rcParams["axes.unicode_minus"] = False
        break

from maskrcnn.data import CocoInstanceDataset
from maskrcnn.evaluate import (evaluate_coco, load_checkpoint,
                               per_class_ap, precision_recall_f1_curve)

# bbox/segm 2계열 고정 색상 (파랑/주황 — validate_palette.js로 확인한 조합:
# CVD ΔE 31.1, normal-vision ΔE 36.6, 둘 다 여유 있게 통과)
COLORS = {"bbox": "#2f6fed", "segm": "#e2711d"}

# pycocotools COCOeval.summarize()가 채우는 e.stats의 고정 순서 (bbox/segm 공통)
STAT_NAMES = ["AP", "AP50", "AP75", "AP_S", "AP_M", "AP_L",
             "AR1", "AR10", "AR100", "AR_S", "AR_M", "AR_L"]


def parse_args():
    p = argparse.ArgumentParser(description="Precision/Recall/F1 시각화")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--val-ann", required=True)
    p.add_argument("--output", default="outputs/metrics.png")
    p.add_argument("--iou-thresh", type=float, default=0.5,
                   help="PR/F1 curve를 뽑을 IoU 기준 (기본 0.5, COCO AP50에 해당)")
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

    metrics, raw_evals = evaluate_coco(
        model, dataset, device, iou_types=tuple(args.iou_types),
        max_images=args.max_images, min_size=args.min_size,
        max_size=args.max_size, return_raw=True)

    curves = {}
    summary = {}
    for iou_type, e in raw_evals.items():
        c = precision_recall_f1_curve(e, iou_thresh=args.iou_thresh)
        curves[iou_type] = c
        i = c["best_idx"]
        summary[iou_type] = {
            "AP": metrics[f"{iou_type}/AP"],
            "AP50": metrics[f"{iou_type}/AP50"],
            "AP75": metrics[f"{iou_type}/AP75"],
            "best_f1": float(c["f1"][i]),
            "precision_at_best_f1": float(c["precision"][i]),
            "recall_at_best_f1": float(c["recall"][i]),
            "score_thresh_at_best_f1": float(c["score"][i]),
        }

    print("\n===== F1 요약 (IoU>=%.2f) =====" % args.iou_thresh)
    for iou_type, s in summary.items():
        print(f"[{iou_type}] best F1={s['best_f1']:.4f} "
              f"(P={s['precision_at_best_f1']:.4f}, R={s['recall_at_best_f1']:.4f}) "
              f"@ score_thresh~={s['score_thresh_at_best_f1']:.3f}  "
              f"| AP={s['AP']:.4f} AP50={s['AP50']:.4f} AP75={s['AP75']:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot(curves, args.iou_thresh, out_path)
    print(f"\n그래프 저장 -> {out_path}")

    per_class = {iou_type: per_class_ap(e, iou_thresh=args.iou_thresh)
                for iou_type, e in raw_evals.items()}
    breakdown_path = out_path.with_name(out_path.stem + "_breakdown" + out_path.suffix)
    _plot_breakdown(raw_evals, per_class, dataset, args.iou_thresh, breakdown_path)
    print(f"breakdown 그래프 저장 -> {breakdown_path}")

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"수치 요약 저장 -> {json_path}")


def _plot(curves: dict, iou_thresh: float, out_path: Path):
    fig, (ax_pr, ax_f1) = plt.subplots(1, 2, figsize=(11, 4.5))

    # 두 계열(bbox/segm)의 best-F1 지점이 서로 가까울 때 라벨이 겹치지 않도록
    # 계열마다 반대 방향으로 떨어뜨리고, 점과 라벨을 얇은 선으로 이어준다.
    label_offsets = [(10, 16), (10, -28)]

    for idx, (iou_type, c) in enumerate(curves.items()):
        color = COLORS.get(iou_type, "#555555")
        i = c["best_idx"]

        ax_pr.plot(c["recall"], c["precision"], color=color, linewidth=2,
                  label=iou_type)
        ax_pr.scatter([c["recall"][i]], [c["precision"][i]], color=color,
                     s=36, zorder=3)

        ax_f1.plot(c["recall"], c["f1"], color=color, linewidth=2,
                  label=iou_type)
        ax_f1.scatter([c["recall"][i]], [c["f1"][i]], color=color,
                     s=36, zorder=3)
        xytext = label_offsets[idx % len(label_offsets)]
        ax_f1.annotate(f"F1={c['f1'][i]:.2f} @{c['score'][i]:.2f}",
                      (c["recall"][i], c["f1"][i]),
                      textcoords="offset points", xytext=xytext,
                      fontsize=8, color=color, ha="left",
                      arrowprops=dict(arrowstyle="-", color=color,
                                      lw=0.75, alpha=0.6))

    for ax, title, ylabel in (
        (ax_pr, f"Precision-Recall (IoU>={iou_thresh:.2f})", "Precision"),
        (ax_f1, f"F1 vs Recall (IoU>={iou_thresh:.2f})", "F1"),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Recall")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_breakdown(raw_evals: dict, per_class: dict, dataset,
                    iou_thresh: float, out_path: Path):
    iou_types = list(raw_evals)
    # 클래스 이름은 원본 COCO category_id 기준 (COCOeval.params.catIds와 같은 축)
    cat_name = {cid: c["name"] for cid, c in dataset.coco.cats.items()}
    n_classes = max((len(pc) for pc in per_class.values()), default=0)

    fig = plt.figure(figsize=(11, 4.2 + 0.22 * n_classes))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, max(2, 0.22 * n_classes)],
                          hspace=0.35)

    # ---- (1) COCOeval 표준 12개 지표 (bbox/segm 나란히)
    ax_stats = fig.add_subplot(gs[0])
    n_series = len(iou_types)
    width = 0.8 / max(n_series, 1)
    x = range(len(STAT_NAMES))
    for idx, iou_type in enumerate(iou_types):
        stats = raw_evals[iou_type].stats
        offs = (idx - (n_series - 1) / 2) * width
        ax_stats.bar([xi + offs for xi in x], stats, width=width,
                    color=COLORS.get(iou_type, "#555555"), label=iou_type)
    ax_stats.set_xticks(list(x))
    ax_stats.set_xticklabels(STAT_NAMES, fontsize=8, rotation=30, ha="right")
    ax_stats.set_ylim(0, 1.0)
    ax_stats.set_title("COCOeval summary stats")
    ax_stats.grid(True, axis="y", linewidth=0.4, alpha=0.3)
    ax_stats.legend(frameon=False, fontsize=9)

    # ---- (2) 클래스별 AP (IoU>=iou_thresh), bbox 기준 내림차순 정렬
    ax_cls = fig.add_subplot(gs[1])
    if n_classes:
        ref_type = "bbox" if "bbox" in per_class else iou_types[0]
        cat_ids = sorted(per_class[ref_type], key=lambda c: per_class[ref_type][c])
        y = range(len(cat_ids))
        for idx, iou_type in enumerate(iou_types):
            vals = [per_class[iou_type].get(cid, 0.0) for cid in cat_ids]
            offs = (idx - (n_series - 1) / 2) * width
            ax_cls.barh([yi + offs for yi in y], vals, height=width,
                       color=COLORS.get(iou_type, "#555555"), label=iou_type)
        ax_cls.set_yticks(list(y))
        ax_cls.set_yticklabels([cat_name.get(c, str(c)) for c in cat_ids],
                              fontsize=7)
        ax_cls.set_xlim(0, 1.0)
        ax_cls.set_xlabel(f"AP (IoU>={iou_thresh:.2f})")
        ax_cls.set_title("Per-class AP (classes with GT only, ascending)")
        ax_cls.grid(True, axis="x", linewidth=0.4, alpha=0.3)
        ax_cls.legend(frameon=False, fontsize=9)
    else:
        ax_cls.text(0.5, 0.5, "No classes with GT", ha="center", va="center")
        ax_cls.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
