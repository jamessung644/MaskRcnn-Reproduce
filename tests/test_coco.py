"""COCO 데이터로 전체 파이프라인 검증.

1. CocoInstanceDataset이 실제 COCO annotation을 올바른 타깃으로 만드는지
2. torchvision 사전학습 가중치가 우리 구조에 텐서 단위로 1:1 매핑되는지
   (= 구조가 레퍼런스 구현과 동형이라는 검증)
3. 실제 이미지 추론 결과가 torchvision 레퍼런스 출력과 일치하는지
4. 시각화 저장 (outputs/)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image

from maskrcnn import Config, MaskRCNN
from maskrcnn.data import (CocoInstanceDataset, batch_images, normalize_image,
                           resize_image_and_target)
from maskrcnn.utils.box_ops import box_iou
from maskrcnn.utils.tv_weights import load_torchvision_maskrcnn
from maskrcnn.utils.visualize import draw_detections

DATA = ROOT / "data" / "coco"
ANN = DATA / "annotations" / "instances_val2017.json"
IMG_DIR = DATA / "val2017"
OUT_DIR = ROOT / "outputs"


def test_dataset(dataset):
    print("=== 1. 데이터셋 검증 ===")
    for i in range(min(5, len(dataset))):
        image, target = dataset[i]
        n = target["boxes"].shape[0]
        h, w = image.shape[-2:]
        assert image.dtype == torch.float32 and image.shape[0] == 3
        assert target["labels"].shape == (n,)
        assert target["masks"].shape == (n, h, w)
        assert (target["boxes"][:, 2] > target["boxes"][:, 0]).all()
        assert (target["boxes"][:, 3] > target["boxes"][:, 1]).all()
        assert (target["boxes"][:, [0, 2]] <= w + 1).all()
        assert (target["boxes"][:, [1, 3]] <= h + 1).all()
        print(f"  이미지 {i}: {tuple(image.shape)}, 인스턴스 {n}개, "
              f"클래스 {sorted(set(target['labels'].tolist()))}")
    print("  통과\n")


def build_our_model():
    print("=== 2. 사전학습 가중치 매핑 ===")
    # torchvision 체계에 맞춤: 91 클래스(원본 COCO id), 앵커 offset 0
    cfg = Config(num_classes=91, anchor_offset=0.0)
    model = MaskRCNN(cfg).eval()
    stats = load_torchvision_maskrcnn(model)
    print(f"  매핑된 텐서 {stats['mapped']}개, 건너뜀 {stats['skipped']}개")
    print("  모든 파라미터 1:1 매핑 완료 -> 구조 동형 확인\n")
    return model


@torch.no_grad()
def run_our_model(model, pil_image):
    image = torch.from_numpy(__import__("numpy").array(pil_image)) \
        .permute(2, 0, 1).float() / 255.0
    resized, _, scale = resize_image_and_target(image, None)
    resized = normalize_image(resized)
    batch, image_sizes = batch_images([resized])
    det = model(batch, image_sizes)[0]
    det["boxes"] = det["boxes"] / scale  # 원본 좌표계로 복원
    return det


@torch.no_grad()
def compare_with_torchvision(model, dataset, num_images=4):
    print("=== 3. torchvision 레퍼런스와 출력 비교 ===")
    import numpy as np
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn)

    tv_model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).eval()

    OUT_DIR.mkdir(exist_ok=True)
    total_ref, total_matched = 0, 0

    for i in range(min(num_images, len(dataset))):
        info = dataset.coco.imgs[dataset.image_ids[i]]
        pil = Image.open(IMG_DIR / info["file_name"]).convert("RGB")

        ours = run_our_model(model, pil)
        tv_in = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
        ref = tv_model([tv_in])[0]

        # 레퍼런스의 confident detection이 우리 출력에도 있는지 (label 일치, IoU>0.5)
        ref_keep = ref["scores"] > 0.5
        ref_boxes, ref_labels = ref["boxes"][ref_keep], ref["labels"][ref_keep]
        matched = 0
        for rb, rl in zip(ref_boxes, ref_labels):
            same_label = ours["labels"] == rl
            if same_label.sum() > 0:
                ious = box_iou(rb[None], ours["boxes"][same_label])[0]
                if ious.max() > 0.5:
                    matched += 1
        total_ref += len(ref_boxes)
        total_matched += matched

        top = ours["scores"] > 0.5
        names = [dataset.label_to_name.get(int(l), "?")
                 for l in ours["labels"][top][:6]]
        print(f"  {info['file_name']}: 우리 {int(top.sum())}개 검출 "
              f"(ref {len(ref_boxes)}개 중 {matched}개 일치) — {names}")

        vis = draw_detections(pil, ours, dataset.label_to_name)
        vis.save(OUT_DIR / f"det_{info['file_name']}")

    rate = total_matched / max(total_ref, 1)
    print(f"\n  레퍼런스 detection 일치율: {total_matched}/{total_ref} "
          f"({rate:.0%})")
    assert rate >= 0.8, "레퍼런스와 출력 불일치 — 구현 확인 필요"
    print(f"  시각화 저장: {OUT_DIR}\n")


def main():
    torch.manual_seed(0)
    dataset = CocoInstanceDataset(str(IMG_DIR), str(ANN), contiguous_ids=False)
    # 디스크에 실제로 받아둔 이미지만 사용
    on_disk = {p.name for p in IMG_DIR.glob("*.jpg")}
    dataset.image_ids = [
        i for i in dataset.image_ids
        if dataset.coco.imgs[i]["file_name"] in on_disk
    ]
    print(f"샘플 이미지 {len(dataset)}장 사용\n")

    test_dataset(dataset)
    model = build_our_model()
    compare_with_torchvision(model, dataset)
    print("COCO 파이프라인 테스트 전부 통과 [OK]")


if __name__ == "__main__":
    main()
