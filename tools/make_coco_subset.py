"""COCO annotation JSON에서 이미지 N장만 무작위로 뽑아 서브셋 JSON을 만든다.

이미지 파일 자체는 옮기지 않는다 — CocoInstanceDataset은 JSON에 있는
file_name만 읽으므로, --train-images는 원본 폴더(예: train2017/) 그대로 두고
--train-ann만 이 스크립트가 만든 서브셋 JSON으로 바꾸면 된다.

사용 예:
    python tools/make_coco_subset.py \
        --ann    data/coco/annotations/instances_train2017.json \
        --output data/coco/annotations/instances_train2017_2k.json \
        --num-images 2000 --seed 0
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="COCO annotation 서브셋 생성")
    p.add_argument("--ann", required=True, help="원본 COCO annotation JSON")
    p.add_argument("--output", required=True, help="서브셋 JSON 저장 경로")
    p.add_argument("--num-images", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-empty", action="store_true",
                   help="유효 annotation이 없는 이미지도 후보에 포함한다 "
                        "(기본은 학습 스크립트의 skip_empty와 맞춰 제외)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.ann, "r") as f:
        data = json.load(f)

    img_to_anns = {}
    for a in data["annotations"]:
        img_to_anns.setdefault(a["image_id"], []).append(a)

    def has_valid_ann(img_id: int) -> bool:
        for a in img_to_anns.get(img_id, []):
            if a.get("iscrowd", 0):
                continue
            _, _, w, h = a["bbox"]
            if w > 0 and h > 0:
                return True
        return False

    candidates = [img["id"] for img in data["images"]]
    if not args.allow_empty:
        candidates = [i for i in candidates if has_valid_ann(i)]

    if len(candidates) < args.num_images:
        raise SystemExit(
            f"후보 이미지가 {len(candidates)}장뿐이라 {args.num_images}장을 "
            f"뽑을 수 없다 (--allow-empty로 후보를 늘릴 수 있다)."
        )

    rng = random.Random(args.seed)
    selected = set(rng.sample(candidates, args.num_images))

    images = [img for img in data["images"] if img["id"] in selected]
    annotations = [a for a in data["annotations"] if a["image_id"] in selected]

    out = {
        "images": images,
        "annotations": annotations,
        "categories": data["categories"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"이미지 {len(images)}장, annotation {len(annotations)}개 -> {out_path}")


if __name__ == "__main__":
    main()
