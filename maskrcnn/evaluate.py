"""추론/평가 엔진: 체크포인트 로드, 원본 좌표 예측, COCO mAP 평가.

- load_checkpoint: train.py가 저장한 체크포인트로 모델을 복원한다.
- predict_original: 이미지 한 장을 리사이즈/정규화 -> 추론 -> 박스를 원본
  좌표계로 되돌려 반환한다 (마스크는 28x28 그대로, paste 시 박스 크기로 확장).
- evaluate_coco: 데이터셋 전체를 추론해 pycocotools COCOeval로 bbox/segm AP를
  계산한다 (pycocotools 필요).
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from .config import Config
from .data.transforms import (batch_images, normalize_image,
                              resize_image_and_target)
from .model.mask_rcnn import MaskRCNN
from .utils.visualize import paste_mask


def load_checkpoint(path: str, device, anchor_offset: float = 0.5):
    """train.py 체크포인트 -> (model(eval), label_to_name)."""
    ckpt = torch.load(path, map_location=device)
    num_classes = ckpt["num_classes"]
    backbone_depth = ckpt.get("backbone_depth", 50)  # 구버전 체크포인트 호환
    cfg = Config(num_classes=num_classes, anchor_offset=anchor_offset)
    model = MaskRCNN(cfg, backbone_depth=backbone_depth)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    label_to_name = ckpt.get("label_to_name", {})
    return model, label_to_name


@torch.no_grad()
def predict_original(model, pil: Image.Image, device,
                     min_size: int = 800, max_size: int = 1333
                     ) -> Dict[str, Tensor]:
    """PIL 이미지 -> 원본 좌표계의 {"boxes","labels","scores","masks"}."""
    image = torch.from_numpy(np.array(pil.convert("RGB"))) \
        .permute(2, 0, 1).float() / 255.0
    resized, _, scale = resize_image_and_target(image, None, min_size, max_size)
    resized = normalize_image(resized)
    batch, image_sizes = batch_images([resized])
    det = model(batch.to(device), image_sizes)[0]
    det = {k: v.detach().cpu() for k, v in det.items()}
    det["boxes"] = det["boxes"] / scale  # 원본 좌표로 복원
    return det


def evaluate_coco(model, dataset, device,
                  iou_types=("bbox", "segm"),
                  max_images: Optional[int] = None,
                  min_size: int = 800, max_size: int = 1333,
                  verbose: bool = True, return_raw: bool = False):
    """COCO mAP 평가. pycocotools가 필요하다.

    반환: {"bbox/AP": .., "bbox/AP50": .., "segm/AP": .., ...}
    return_raw=True면 (metrics, {iou_type: COCOeval}) 튜플을 반환한다 —
    COCOeval.eval['precision']/['recall']/['scores']에서 F1/PR curve 등
    scalar AP로는 안 보이는 지표를 뽑아낼 때 쓴다 (tools/plot_metrics.py 참고).
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from pycocotools import mask as mask_utils
    except ImportError as e:
        raise RuntimeError(
            "mAP 평가에는 pycocotools가 필요하다 (`pip install pycocotools`)."
        ) from e

    # 학습용 라벨(1..K) -> 원본 COCO category_id 역매핑
    label_to_cat = {lbl: cid for cid, lbl in dataset.cat_id_to_label.items()}

    image_ids = dataset.image_ids
    if max_images is not None:
        image_ids = image_ids[:max_images]

    results_bbox: List[dict] = []
    results_segm: List[dict] = []
    model.eval()

    # 이미지 1장씩 순차 추론이라 전체가 끝날 때까지 아무 출력도 없으면
    # 멈춘 것처럼 보인다 — 데이터셋 크기에 상관없이 대략 10번 정도
    # 진행 상황을 찍도록 간격을 크기에 맞춰 계산한다(200장이든 5000장이든).
    print_interval = max(1, len(image_ids) // 10)

    for n, img_id in enumerate(image_ids):
        info = dataset.coco.imgs[img_id]
        H, W = info["height"], info["width"]
        pil = Image.open(dataset.img_dir / info["file_name"]).convert("RGB")
        det = predict_original(model, pil, device, min_size, max_size)

        boxes, labels = det["boxes"], det["labels"]
        scores, masks = det["scores"], det.get("masks")
        for i in range(boxes.shape[0]):
            cat_id = label_to_cat.get(int(labels[i]), int(labels[i]))
            x1, y1, x2, y2 = boxes[i].tolist()
            score = float(scores[i])
            if "bbox" in iou_types:
                results_bbox.append({
                    "image_id": img_id, "category_id": cat_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score,
                })
            if "segm" in iou_types and masks is not None:
                full = (paste_mask(masks[i], boxes[i], (H, W)) > 0.5).numpy()
                rle = mask_utils.encode(np.asfortranarray(full.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                results_segm.append({
                    "image_id": img_id, "category_id": cat_id,
                    "segmentation": rle, "score": score,
                })

        if verbose and (n + 1) % print_interval == 0:
            print(f"  [eval] {n + 1}/{len(image_ids)} 이미지 추론 완료", flush=True)

    coco_gt = COCO(dataset.ann_file)
    metrics: Dict[str, float] = {}
    raw_evals: Dict[str, "COCOeval"] = {}
    for iou_type, results in (("bbox", results_bbox), ("segm", results_segm)):
        if iou_type not in iou_types:
            continue
        if not results:
            metrics[f"{iou_type}/AP"] = 0.0
            continue
        coco_dt = coco_gt.loadRes(results)
        e = COCOeval(coco_gt, coco_dt, iou_type)
        e.params.imgIds = list(image_ids)
        e.evaluate(); e.accumulate()
        if verbose:
            print(f"\n----- {iou_type} -----", flush=True)
        e.summarize()
        metrics[f"{iou_type}/AP"] = float(e.stats[0])     # AP @[.5:.95]
        metrics[f"{iou_type}/AP50"] = float(e.stats[1])   # AP @.50
        metrics[f"{iou_type}/AP75"] = float(e.stats[2])   # AP @.75
        raw_evals[iou_type] = e

    if return_raw:
        return metrics, raw_evals
    return metrics


def precision_recall_f1_curve(coco_eval: "COCOeval", iou_thresh: float = 0.5,
                              area: str = "all", max_dets: Optional[int] = None):
    """COCOeval.accumulate() 결과에서 카테고리 평균 PR curve와 F1 curve를 뽑는다.

    AP 계산과 동일한 방식(카테고리별 precision을 recall grid 위에서 평균)으로
    "전체" precision-recall curve를 만들고, 그 위에서 F1 = 2PR/(P+R)을 계산한다.
    scores 배열도 같은 grid 위에 있어서, best-F1 지점의 근사 confidence
    threshold도 함께 뽑을 수 있다.

    반환: dict(recall, precision, f1, score, best_idx) — 전부 numpy 배열이고
        best_idx는 f1이 최대인 recall grid 인덱스.
    """
    import numpy as np

    p = coco_eval.params
    t_idx = int(np.argmin(np.abs(np.array(p.iouThrs) - iou_thresh)))
    a_idx = p.areaRngLbl.index(area)
    m_idx = (p.maxDets.index(max_dets) if max_dets is not None
             else len(p.maxDets) - 1)

    # (R, K): recall grid x 카테고리. 유효 카테고리(-1 아닌) 없는 recall은 nan.
    precision = coco_eval.eval["precision"][t_idx, :, :, a_idx, m_idx]
    scores = coco_eval.eval["scores"][t_idx, :, :, a_idx, m_idx]
    valid = precision > -1

    recall = np.array(p.recThrs)
    mean_precision = np.full(recall.shape, np.nan)
    mean_score = np.full(recall.shape, np.nan)
    for r in range(precision.shape[0]):
        col = valid[r]
        if col.any():
            mean_precision[r] = precision[r, col].mean()
            mean_score[r] = scores[r, col].mean()

    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * mean_precision * recall / (mean_precision + recall)
    f1 = np.nan_to_num(f1, nan=0.0)

    best_idx = int(np.nanargmax(f1)) if np.isfinite(f1).any() else 0
    return {
        "recall": recall,
        "precision": np.nan_to_num(mean_precision, nan=0.0),
        "f1": f1,
        "score": np.nan_to_num(mean_score, nan=0.0),
        "best_idx": best_idx,
    }


def per_class_ap(coco_eval: "COCOeval", iou_thresh: float = 0.5,
                 area: str = "all", max_dets: Optional[int] = None
                 ) -> Dict[int, float]:
    """카테고리별 AP(주어진 IoU 기준, recall grid 평균). GT 없는 클래스는 제외.

    반환: {원본 COCO category_id: AP}. `dataset.coco.cats[cid]["name"]`으로
    이름을 붙일 수 있다 (evaluate_coco의 label_to_cat 역매핑과 같은 catId 축).
    """
    p = coco_eval.params
    t_idx = int(np.argmin(np.abs(np.array(p.iouThrs) - iou_thresh)))
    a_idx = p.areaRngLbl.index(area)
    m_idx = (p.maxDets.index(max_dets) if max_dets is not None
             else len(p.maxDets) - 1)

    precision = coco_eval.eval["precision"][t_idx, :, :, a_idx, m_idx]  # (R, K)
    out: Dict[int, float] = {}
    for k, cat_id in enumerate(p.catIds):
        col = precision[:, k]
        valid = col > -1
        if valid.any():
            out[cat_id] = float(col[valid].mean())
    return out
