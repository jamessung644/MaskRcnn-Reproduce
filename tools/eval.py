import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from maskrcnn import Config, MaskRCNN
from maskrcnn.data import CocoInstanceDataset
from maskrcnn.evaluate import evaluate_coco

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default=None, help="없으면 --pretrained 로 torchvision 가중치만 로드")
p.add_argument("--pretrained", action="store_true")
p.add_argument("--num-classes", type=int, default=91)
p.add_argument("--images", required=True)
p.add_argument("--ann", required=True)
p.add_argument("--max-images", type=int, default=None)
p.add_argument("--min-size", type=int, default=800)
p.add_argument("--max-size", type=int, default=1333)
p.add_argument("--contiguous-ids", action="store_true", default=False,
               help="True면 category_id를 1..K로 압축, False면 원본 id 그대로")
a = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskRCNN(Config(num_classes=a.num_classes))

if a.ckpt:
    ckpt = torch.load(a.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"체크포인트 로드: {a.ckpt}")
elif a.pretrained:
    from maskrcnn.utils.tv_weights import load_torchvision_pretrained
    load_torchvision_pretrained(model)
    print("torchvision 사전학습 가중치 로드 (학습 0회)")

model.to(device).eval()

ds = CocoInstanceDataset(a.images, a.ann, contiguous_ids=a.contiguous_ids,
                         min_size=a.min_size, max_size=a.max_size,
                         skip_empty=False)
print(f"contiguous_ids={a.contiguous_ids}")

m = evaluate_coco(model, ds, device, max_images=a.max_images,
                  min_size=a.min_size, max_size=a.max_size, verbose=True)
for k, v in m.items():
    print(f"{k:12s} {v:.4f}")