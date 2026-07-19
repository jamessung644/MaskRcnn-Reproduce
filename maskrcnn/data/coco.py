"""COCO instance-segmentation 포맷 데이터셋.

커스텀 데이터셋(예: 폐 X-ray)도 COCO 포맷(images/annotations/categories)으로
변환해 두면 이 로더를 그대로 재사용할 수 있다.

pycocotools에 의존하지 않도록 annotation JSON을 직접 파싱하고, polygon
segmentation은 PIL로 래스터화한다. RLE 마스크는 pycocotools가 설치돼 있으면
그것으로 디코딩하고, 없으면 명확한 오류를 낸다.

각 __getitem__은 (image, target)을 리사이즈까지 마쳐 반환한다:
    image  : (3, H, W) float32 [0, 1]
    target : {"boxes"(N,4) xyxy, "labels"(N,) int64, "masks"(N,H,W) uint8,
              "image_id"(1,)}
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import MAX_SIZE, MIN_SIZE, resize_image_and_target


class _CocoIndex:
    """pycocotools.COCO의 최소 대체: imgs/cats/이미지별 annotation 인덱스."""

    def __init__(self, ann_file: str):
        with open(ann_file, "r") as f:
            data = json.load(f)
        self.imgs: Dict[int, dict] = {img["id"]: img for img in data["images"]}
        self.cats: Dict[int, dict] = {c["id"]: c for c in data["categories"]}
        self.img_to_anns: Dict[int, List[dict]] = defaultdict(list)
        for ann in data.get("annotations", []):
            self.img_to_anns[ann["image_id"]].append(ann)


class CocoInstanceDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str,
                 contiguous_ids: bool = True,
                 min_size: int = MIN_SIZE, max_size: int = MAX_SIZE,
                 skip_empty: bool = True):
        self.img_dir = Path(img_dir)
        self.coco = _CocoIndex(ann_file)
        self.min_size = min_size
        self.max_size = max_size

        image_ids = sorted(self.coco.imgs.keys())
        if skip_empty:
            image_ids = [i for i in image_ids if self._has_valid_ann(i)]
        self.image_ids = image_ids

        # category id -> 학습용 라벨. contiguous면 1..K(배경 0), 아니면 원본 id 유지.
        cat_ids = sorted(self.coco.cats.keys())
        if contiguous_ids:
            self.cat_id_to_label = {cid: i + 1 for i, cid in enumerate(cat_ids)}
        else:
            self.cat_id_to_label = {cid: cid for cid in cat_ids}
        self.label_to_name = {
            lbl: self.coco.cats[cid]["name"]
            for cid, lbl in self.cat_id_to_label.items()
        }

    # ------------------------------------------------------------------
    def _has_valid_ann(self, img_id: int) -> bool:
        for a in self.coco.img_to_anns.get(img_id, []):
            if a.get("iscrowd", 0):
                continue
            x, y, w, h = a["bbox"]
            if w > 0 and h > 0:
                return True
        return False

    def __len__(self) -> int:
        return len(self.image_ids)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        info = self.coco.imgs[img_id]
        pil = Image.open(self.img_dir / info["file_name"]).convert("RGB")
        W, H = pil.size
        image = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0

        boxes, labels, masks = [], [], []
        for a in self.coco.img_to_anns.get(img_id, []):
            if a.get("iscrowd", 0):
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_label[a["category_id"]])
            masks.append(self._ann_to_mask(a.get("segmentation"), H, W))

        if boxes:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "masks": torch.as_tensor(np.stack(masks), dtype=torch.uint8),
            }
        else:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, H, W), dtype=torch.uint8),
            }
        target["image_id"] = torch.tensor([img_id])

        image, target, _ = resize_image_and_target(
            image, target, self.min_size, self.max_size)
        return image, target

    # ------------------------------------------------------------------
    @staticmethod
    def _ann_to_mask(segm, height: int, width: int) -> np.ndarray:
        """segmentation -> (H, W) uint8 이진 마스크."""
        if segm is None:
            return np.zeros((height, width), dtype=np.uint8)

        # polygon: [[x1,y1,x2,y2,...], ...]
        if isinstance(segm, list):
            mask = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(mask)
            for poly in segm:
                if len(poly) < 6:
                    continue
                xy = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                draw.polygon(xy, outline=1, fill=1)
            return np.asarray(mask, dtype=np.uint8)

        # RLE (dict): pycocotools가 있으면 사용
        try:
            from pycocotools import mask as mask_utils
        except ImportError as e:
            raise RuntimeError(
                "RLE 형식 마스크를 디코딩하려면 pycocotools가 필요하다 "
                "(`pip install pycocotools`). polygon 포맷 데이터는 의존성 없이 동작한다."
            ) from e
        rle = segm
        if isinstance(segm["counts"], list):
            rle = mask_utils.frPyObjects(segm, height, width)
        return mask_utils.decode(rle).astype(np.uint8)
