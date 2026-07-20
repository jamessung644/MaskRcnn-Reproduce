"""train.py 학습 로그(stdout)를 파싱해 loss/lr 학습곡선을 그린다.

idalog 등으로 저장한 텍스트 로그 파일을 그대로 읽는다 — 재실행/재개
(--resume)로 로그가 여러 파일에 나뉘어 있으면 --log에 시간순으로 여러 개
넘기면 된다 (크래시로 겹치는 step 구간은 나중 파일 값으로 덮어써 정리한다).

파싱하는 두 줄 형식 (train.py의 실제 출력 포맷):
    E4/12 [ 200/500] step 2201/6000  loss=0.531  cls=0.133 box=0.106 \
        mask=0.240 rpn_obj=0.011 rpn_box=0.042  lr=0.00500  ...
    [epoch 4 완료] avg_loss=0.528  time=0:03:10  -> checkpoints/...

6개 loss 성분을 한 축에 겹쳐 그리면 스케일(mask~0.2 vs rpn_obj~0.01)도 안
맞고 색으로 6개를 다 구분하는 것도 무리라, 성분별로 작은 서브플롯을 따로
두는 small-multiples 방식을 쓴다 — 서브플롯 하나 = 시계열 하나라 색 구분이
필요 없다(전부 동일한 파란색, 제목으로 구분).

사용 예:
    python tools/plot_training_curve.py \
        --log /var/tmp/iida_logs/shsung_20260719_142136_train.log \
        --output outputs/training_curve.png
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 단일 시계열 서브플롯이라 계열 구분용 색이 필요 없다 — sequential 기본 hue
# 하나로 고정 (dataviz 팔레트 기준 blue #2a78d6).
LINE_COLOR = "#2a78d6"
EPOCH_LINE_COLOR = "#9a9990"  # muted, epoch 경계 표시용

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ITER_RE = re.compile(
    r"E(?P<epoch>\d+)/(?P<total_epochs>\d+)\s*\[\s*\d+/\s*\d+\]\s*"
    r"step\s*(?P<step>\d+)/(?P<total_steps>\d+)\s*"
    r"loss=(?P<loss>[\d.]+)\s*cls=(?P<cls>[\d.]+)\s*box=(?P<box>[\d.]+)\s*"
    r"mask=(?P<mask>[\d.]+)\s*rpn_obj=(?P<rpn_obj>[\d.]+)\s*"
    r"rpn_box=(?P<rpn_box>[\d.]+)\s*lr=(?P<lr>[\d.]+)"
)
EPOCH_RE = re.compile(
    r"\[epoch\s*(?P<epoch>\d+)\s*완료\]\s*avg_loss=(?P<avg_loss>[\d.]+)"
)

LOSS_FIELDS = ["loss", "cls", "box", "mask", "rpn_obj", "rpn_box"]


def parse_args():
    p = argparse.ArgumentParser(description="학습 로그 -> loss/lr 곡선 시각화")
    p.add_argument("--log", nargs="+", required=True,
                   help="train.py stdout 로그 파일(들). 재개로 여러 개면 "
                        "시간순으로 나열 (겹치는 step은 나중 파일이 이긴다)")
    p.add_argument("--output", default="outputs/training_curve.png")
    return p.parse_args()


def parse_logs(log_paths):
    """(step -> 필드 dict, epoch -> avg_loss dict, total_steps) 반환.

    step/epoch 키로 저장해두면 재개로 겹치는 구간이 있어도 나중 파일이 자동으로
    앞 파일의 값을 덮어써 정리된다.
    """
    by_step = {}
    epoch_avg = {}
    total_steps = None

    for path in log_paths:
        with open(path, "r", errors="ignore") as f:
            for raw_line in f:
                line = ANSI_RE.sub("", raw_line)

                m = ITER_RE.search(line)
                if m:
                    step = int(m.group("step"))
                    total_steps = int(m.group("total_steps"))
                    by_step[step] = {
                        "epoch": int(m.group("epoch")),
                        **{k: float(m.group(k)) for k in LOSS_FIELDS},
                        "lr": float(m.group("lr")),
                    }
                    continue

                m = EPOCH_RE.search(line)
                if m:
                    epoch_avg[int(m.group("epoch"))] = float(m.group("avg_loss"))

    if not by_step:
        raise SystemExit("로그에서 학습 iteration 줄을 하나도 못 찾았다 — "
                         "포맷이 바뀌었거나 잘못된 파일인지 확인.")
    return by_step, epoch_avg, total_steps


def _epoch_boundaries(by_step):
    """각 epoch가 시작되는 첫 step 목록 (배경 경계선용)."""
    steps_sorted = sorted(by_step)
    boundaries = []
    seen_epochs = set()
    for s in steps_sorted:
        e = by_step[s]["epoch"]
        if e not in seen_epochs:
            seen_epochs.add(e)
            boundaries.append(s)
    return boundaries


def plot_training_curve(by_step, epoch_avg, total_steps, out_path: Path):
    steps = sorted(by_step)
    boundaries = _epoch_boundaries(by_step)

    panels = LOSS_FIELDS + ["lr"]
    n = len(panels) + (1 if epoch_avg else 0)
    ncols = 4
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.0 * nrows))
    axes = axes.flatten()

    for ax, field in zip(axes, panels):
        ys = [by_step[s][field] for s in steps]
        ax.plot(steps, ys, color=LINE_COLOR, linewidth=1.5)
        for b in boundaries:
            ax.axvline(b, color=EPOCH_LINE_COLOR, linewidth=0.6, alpha=0.5)
        ax.set_title(field, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.tick_params(labelsize=8)
        if total_steps:
            ax.set_xlim(0, total_steps)

    if epoch_avg:
        ax = axes[len(panels)]
        epochs = sorted(epoch_avg)
        ax.plot(epochs, [epoch_avg[e] for e in epochs],
               color=LINE_COLOR, linewidth=1.5, marker="o", markersize=3)
        ax.set_title("avg_loss / epoch", fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(True, linewidth=0.4, alpha=0.3)
        ax.tick_params(labelsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    by_step, epoch_avg, total_steps = parse_logs(args.log)

    steps = sorted(by_step)
    print(f"iteration 로그 {len(steps)}개 (step {steps[0]}~{steps[-1]}"
          f"{f'/{total_steps}' if total_steps else ''}), "
          f"epoch 요약 {len(epoch_avg)}개")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_training_curve(by_step, epoch_avg, total_steps, out_path)
    print(f"학습곡선 저장 -> {out_path}")


if __name__ == "__main__":
    main()
