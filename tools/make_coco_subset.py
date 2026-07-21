"""COCO annotation JSON에서 이미지 N장만 무작위로 뽑거나, 실제로 디스크에
있는 파일만 걸러낸 서브셋 JSON을 만든다.

이미지 파일 자체는 옮기지 않는다 — CocoInstanceDataset은 JSON에 있는
file_name만 읽으므로, --train-images는 원본 폴더(예: train2017/) 그대로 두고
--train-ann만 이 스크립트가 만든 서브셋 JSON으로 바꾸면 된다.

사용 예 1) 무작위 N장 서브셋(빠른 실험용):
    python tools/make_coco_subset.py \
        --ann    data/coco/annotations/instances_train2017.json \
        --output data/coco/annotations/instances_train2017_2k.json \
        --num-images 2000 --seed 0

사용 예 2) 다운로드/압축해제가 아직 덜 끝났을 때, 지금까지 실제로 받아진
파일만으로 바로 학습 시작(--num-images 생략하면 있는 만큼 전부 사용):
    python tools/make_coco_subset.py \
        --ann        data/coco/annotations/instances_train2017.json \
        --output     data/coco/annotations/instances_train2017_partial.json \
        --images-dir data/coco/train2017
"""

import argparse
import json
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="COCO annotation 서브셋 생성")
    p.add_argument("--ann", required=True, help="원본 COCO annotation JSON")
    p.add_argument("--output", required=True, help="서브셋 JSON 저장 경로")
    p.add_argument("--num-images", type=int, default=None,
                   help="이 수만큼 무작위로 뽑는다. 생략하면(--images-dir과 "
                        "같이 쓸 때 특히 유용) 후보 전부를 그대로 쓴다.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-empty", action="store_true",
                   help="유효 annotation이 없는 이미지도 후보에 포함한다 "
                        "(기본은 학습 스크립트의 skip_empty와 맞춰 제외)")
    p.add_argument("--images-dir", default=None,
                   help="이 폴더에 실제로 파일이 존재하는 이미지만 후보로 "
                        "남긴다 — 다운로드/압축해제가 덜 끝난 상태에서 지금 "
                        "있는 것만으로 바로 학습을 시작하고 싶을 때 쓴다.")
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

    if args.images_dir:
        # os.path.exists를 이미지마다 호출하면(네트워크 스토리지에서 특히)
        # 느리니, 디렉토리를 한 번만 통째로 읽어서 메모리에서 비교한다.
        existing_files = {p.name for p in Path(args.images_dir).iterdir()}
        file_name_by_id = {img["id"]: img["file_name"] for img in data["images"]}
        before = len(candidates)
        candidates = [
            i for i in candidates
            if Path(file_name_by_id[i]).name in existing_files
        ]
        print(f"--images-dir 필터: {before} -> {len(candidates)}장 "
              f"(실제 파일 존재하는 것만)")

    if args.num_images is None:
        selected = set(candidates)
    else:
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
