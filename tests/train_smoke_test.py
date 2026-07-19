"""학습 경로 스모크 테스트 (COCO 데이터 없이 합성 타깃으로 검증).

확인 항목:
  - training 모드 forward가 5개 손실(rpn obj/box, roi cls/box, mask)을 반환
  - 모든 손실이 유한하고 0 이상
  - loss.backward()로 그래디언트가 흐르고 SGD 한 스텝이 돈다
  - GT가 없는 이미지(배경만)도 예외 없이 처리된다
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from maskrcnn import Config, MaskRCNN
from maskrcnn.data.transforms import batch_images, normalize_image


def _make_target(num_obj, H, W, num_classes, device, seed):
    g = torch.Generator().manual_seed(seed)
    if num_obj == 0:
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "labels": torch.zeros((0,), dtype=torch.int64, device=device),
            "masks": torch.zeros((0, H, W), dtype=torch.uint8, device=device),
        }
    # 랜덤하지만 유효한 박스(x2>x1, y2>y1)
    x1 = torch.randint(0, W - 40, (num_obj, 1), generator=g)
    y1 = torch.randint(0, H - 40, (num_obj, 1), generator=g)
    w = torch.randint(20, 40, (num_obj, 1), generator=g)
    h = torch.randint(20, 40, (num_obj, 1), generator=g)
    boxes = torch.cat([x1, y1, x1 + w, y1 + h], dim=1).float().to(device)
    labels = torch.randint(1, num_classes, (num_obj,), generator=g).to(device)
    masks = torch.zeros((num_obj, H, W), dtype=torch.uint8, device=device)
    for i in range(num_obj):
        bx1, by1, bx2, by2 = boxes[i].int().tolist()
        masks[i, by1:by2, bx1:bx2] = 1
    return {"boxes": boxes, "labels": labels, "masks": masks}


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 4  # 배경 + 전경 3
    cfg = Config(num_classes=num_classes)
    model = MaskRCNN(cfg).to(device).train()

    H, W = 256, 320
    # 이미지 2장: 하나는 객체 3개, 하나는 객체 0개(배경만)
    imgs = [normalize_image(torch.rand(3, H, W, device=device)),
            normalize_image(torch.rand(3, H, W, device=device))]
    images, image_sizes = batch_images(imgs)
    targets = [
        _make_target(3, H, W, num_classes, device, seed=1),
        _make_target(0, H, W, num_classes, device, seed=2),
    ]

    expected = {"rpn_objectness", "rpn_box_reg",
                "loss_box_cls", "loss_box_reg", "loss_mask"}

    # ---- 1) forward가 손실 딕셔너리를 낸다
    losses = model(images, image_sizes, targets)
    assert set(losses) == expected, f"손실 키 불일치: {set(losses)}"
    for k, v in losses.items():
        assert torch.isfinite(v), f"{k} 비유한: {v}"
        assert v.item() >= 0, f"{k} 음수: {v}"
        print(f"  {k:16s} = {v.item():.4f}")

    total = sum(losses.values())
    print(f"  total loss = {total.item():.4f}")

    # ---- 2) backward + optimizer step (train.py와 동일하게 grad clip 적용)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    optimizer.zero_grad()
    total.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "그래디언트가 하나도 없음"
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(total_norm), "그래디언트 비유한"
    optimizer.step()
    print(f"  grad norm = {total_norm.item():.4f} (clip 1.0), SGD 스텝 완료")

    # ---- 3) 손실이 실제로 감소하는지(같은 배치 몇 스텝 오버핏)
    for step in range(15):
        optimizer.zero_grad()
        loss = sum(model(images, image_sizes, targets).values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        assert torch.isfinite(loss), f"{step}스텝에서 loss 비유한"
    print(f"  15스텝 오버핏 후 total loss = {loss.item():.4f}")
    assert loss.item() < total.item(), "오버핏으로 손실이 줄지 않음"

    # ---- 4) eval 모드 추론도 여전히 동작
    model.eval()
    with torch.no_grad():
        det = model(images, image_sizes)
    assert len(det) == 2 and "masks" in det[0]
    print(f"  eval 추론 OK: 이미지0 detection {det[0]['boxes'].shape[0]}개")

    print("\n학습 스모크 테스트 통과 [OK]")


if __name__ == "__main__":
    main()
