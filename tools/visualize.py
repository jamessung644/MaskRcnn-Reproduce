import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from maskrcnn import Config, MaskRCNN
from maskrcnn.data import (CocoInstanceDataset, batch_images, normalize_image,
                           resize_image_and_target)
from maskrcnn.utils.visualize import draw_detections

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--images", required=True)
p.add_argument("--ann", required=True)
p.add_argument("--n", type=int, default=8)
p.add_argument("--out", default="outputs/vis")
p.add_argument("--device", default="cuda")
a = p.parse_args()

device = torch.device(a.device if torch.cuda.is_available() else "cpu")
ckpt = torch.load(a.ckpt, map_location="cpu")
model = MaskRCNN(Config(num_classes=ckpt["num_classes"]))
model.load_state_dict(ckpt["model"])
model.to(device).eval()


@torch.no_grad()
def infer(pil):
    image = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
    resized, _, scale = resize_image_and_target(image, None)
    resized = normalize_image(resized)
    batch, image_sizes = batch_images([resized])
    det = model(batch.to(device), image_sizes)[0]
    det = {k: v.cpu() for k, v in det.items()}
    det["boxes"] = det["boxes"] / scale
    return det


ds = CocoInstanceDataset(a.images, a.ann, contiguous_ids=True, skip_empty=False)
img_dir = Path(a.images)
out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

for i in range(min(a.n, len(ds))):
    info = ds.coco.imgs[ds.image_ids[i]]
    pil = Image.open(img_dir / info["file_name"]).convert("RGB")
    det = infer(pil)
    draw_detections(pil, det, ds.label_to_name).save(
        out_dir / f"det_{info['file_name']}")
    print(f"{info['file_name']}: {int((det['scores'] > 0.5).sum())}개 (score>0.5)")

print(f"저장 완료: {out_dir}")
