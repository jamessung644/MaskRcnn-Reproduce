# HANDOFF — Mask R-CNN 재현 프로젝트 세션 인계 문서

이 문서는 다른 Claude 세션(또는 사람)이 이 리포의 현재 상태와 지금까지의
맥락을 빠르게 파악할 수 있도록 정리한 것이다. 최신 상태 기준이며, 이 파일
자체는 오래되면 틀릴 수 있으니 `git log`와 실제 코드로 항상 재확인할 것.

## 기본 정보

- 리포: https://github.com/jamessung644/MaskRcnn-Reproduce
- 작업 브랜치: `claude/blissful-nash-953600`
- **PR #1 (`claude/blissful-nash-953600` → `main`)이 열려있고 아직 merge 안 됨.**
  이 브랜치의 모든 커밋이 여기 들어있다. main만 보면 이 세션에서 한 작업이
  하나도 안 보이니 반드시 이 브랜치 기준으로 볼 것.
- 이 문서 작성 시점 최신 커밋: `735cd6b`
- 프로젝트 성격: He et al. *Mask R-CNN* (ICCV 2017) 논문 재현 구현.
  ResNet50-FPN backbone, RPN, RoIAlign box/mask head. 자체 구현이며
  torchvision을 참조하되 의존하지 않음 (`maskrcnn/` 패키지가 전체 구현).

## 지금 이 순간 상황 (제일 중요)

- 학습 환경을 **공유 IIDALAB 서버 → RunPod 단일 L40S(48GB, 전용 컨테이너)**로
  옮겼다. IIDALAB 서버는 호스트 RAM 14.9GB짜리 공유 머신이라 OOM/세션킬/
  좀비 프로세스 문제를 계속 겪었고(아래 "겪은 문제" 참고), RunPod은 그런
  문제가 없다.
- COCO 2017 전체(train2017 118287장)를 받는 중인데 **RunPod 네트워크
  스토리지에서 압축 해제가 매우 느려서**, 다 받기를 기다리지 않고
  **지금 다운로드된 파일만으로 서브셋을 만들어 학습을 시작**했다.
  - 서브셋 JSON: `data/coco/annotations/instances_train2017_20k.json`
    (`tools/make_coco_subset.py --num-images 20000 --images-dir
    data/coco/train2017 --seed 0`로 생성 — 그 시점에 실제 존재하던 파일
    중에서 2만 장 무작위 추출)
  - train2017.zip 압축 해제는 백그라운드에서 계속 진행 중일 수도 있다.
    더 받아졌으면 `make_coco_subset.py`를 다시 돌려 더 큰(또는 전체)
    서브셋으로 새로 학습해도 된다.
- 마지막으로 실행/재개하려던 커맨드(단일 GPU, RunPod, `torchrun` 불필요):
  ```bash
  python3 tools/train.py \
      --train-images data/coco/train2017 \
      --train-ann    data/coco/annotations/instances_train2017_20k.json \
      --num-classes  81 \
      --epochs 5 --batch-size 8 --lr 0.01 \
      --pretrained --amp --freeze-at 2 --mask-downsample 1 \
      --eval-images data/coco/val2017 \
      --eval-ann    data/coco/annotations/instances_val2017.json \
      --eval-interval 1 --eval-max-images 100 \
      --resume checkpoints/maskrcnn_epoch0.pth
  ```
  - epoch0까지 완료됨 (`checkpoints/maskrcnn_epoch0.pth` 존재, avg_loss≈0.79,
    GPU 메모리 약 20~25GB/48GB만 사용 — 배치를 더 키울 여유 있음).
  - `--epochs 5`, `20k` 서브셋을 고른 이유: "2시간 안에 끝내고 싶다"는
    요청에 맞춘 타협안. 전체 데이터로 24 epoch 제대로 돌리면 이 GPU
    기준 약 34시간 걸림(계산 근거는 대화 로그 참고). 이 20k/5epoch 결과가
    괜찮으면 나중에 더 큰 데이터/더 많은 epoch으로 이어서 확장 가능
    (epoch 수가 적어 아직 lr 감쇠 전이라 `--epochs`를 나중에 늘려 재개해도
    스케줄 꼬임 없음).

## 이번 세션에서 한 작업 (커밋 순서, 전부 이 브랜치에 있음)

1. **`ad1d8a1`** DataLoader 공유메모리 전략(`file_system`) + `persistent_workers`
   + `--prefetch-factor` — IIDALAB 서버에서 워커 프로세스가 호스트 RAM을
   계속 누적시키던 문제 완화. `--workers` 기본 4→2.
2. **`e4b6699`, `c53c7e3`** `tools/plot_metrics.py` 신규 — COCOeval 내부
   precision/recall/score 배열을 재활용해 bbox/segm별 **Precision-Recall
   curve + F1 curve**(best-F1 지점/근사 threshold 표시), `_breakdown.png`에
   COCOeval 12개 지표 + **클래스별 AP** 막대그래프. `maskrcnn/evaluate.py`에
   `precision_recall_f1_curve()`, `per_class_ap()` 추가.
3. **`4c3b9dc`** `tools/plot_training_curve.py` 신규 — train.py stdout 로그를
   파싱해 loss(loss/cls/box/mask/rpn_obj/rpn_box)+lr을 small-multiples로 시각화.
   재개로 로그가 여러 파일로 나뉘면 `--log a.log b.log ...`로 여러 개 줘도 됨
   (겹치는 step은 나중 파일이 이김).
4. **`b3fe19a`** Overfitting 대응 + RAM 절약:
   - `CocoInstanceDataset(augment=True)` — random horizontal flip(p=0.5).
     train.py 기본 켜짐, `--no-hflip`로 끔. eval용 데이터셋엔 절대 안 켜짐.
   - `CocoInstanceDataset(mask_downsample=k)` — GT 마스크를 이미지보다 k배
     더 줄여 저장(호스트 RAM 절약, 어차피 최종 28x28로 pooling되니 정보
     손실 미미 — 4배 다운샘플 검증 결과 97.6% 픽셀 일치/IoU 0.95).
     `maskrcnn/model/mask_rcnn.py`의 `_project_masks_on_boxes`가
     `target["mask_scale"]`을 `roi_align spatial_scale`로 받아 좌표 보정.
     train.py `--mask-downsample` 기본 4 (RunPod 커맨드에서는 여유 있어서 1로 끔).
   - `--workers` 기본 2, `--prefetch-factor` 기본 1로 더 낮춤.
5. **`36de302`** 학습 종료 후 **최종 리포트 자동 생성** — `--eval-images`/
   `--eval-ann` 있으면 `<output>/report_epoch<N>/`에 `metrics.png` +
   `metrics_breakdown.png` + `metrics.json` + `detections/*.jpg`(샘플 8장,
   `--report-detections`로 조절) 자동 저장. `--no-final-report`로 끔.
   `tools/plot_metrics.py`의 로직을 `generate_report()` 함수로 리팩터링해
   CLI와 train.py가 같은 코드 재사용.
6. **`fb8b5d1`** DDP NCCL timeout 버그 수정 — per-epoch 평가는 rank0만
   돌고(순차 단일이미지 추론) 나머지 rank는 `dist.barrier()`에서 기다리는데,
   `--eval-max-images` 없이 val 전체(5000장) 평가하면 10분 넘어가서 NCCL
   watchdog이 "hang"으로 오판, job 전체를 죽였다. `timeout=timedelta(minutes=60)`
   추가 + 큰 검증셋인데 `--eval-max-images` 없으면 경고 출력.
7. **`38ecfd0`** `tools/setup_runpod.sh` 신규 — pycocotools/matplotlib 등 설치
   + COCO 2017 전체 다운로드+압축해제 자동화 스크립트. **주의**: 실제로 돌려보니
   네트워크 스토리지에서 118287개 파일 압축 해제가 스크립트의 순차 `unzip`으로는
   비현실적으로 느렸다 (아래 "겪은 문제" 참고) — 병렬 추출로 우회함.
8. **`c8be081`** `plot_training_curve.py`에 **AP/AP50/AP75 bbox-vs-segm
   비교 패널** 추가 — train.py가 이미 찍는 `[epoch N mAP] bbox/AP=..` 로그
   줄을 파싱(추가 연산 없음).
9. **`ca5139a`** `tools/make_coco_subset.py` 확장 — `--images-dir` 옵션으로
   **실제 디스크에 존재하는 파일만** 후보로 필터링(`os.listdir()` 한 번으로
   비교, 파일마다 `os.path.exists()` 호출 안 함 — 네트워크 스토리지에서
   매우 느렸던 경험 때문). `--num-images` 생략 가능(있는 만큼 전부 사용).
   다운로드/압축해제가 덜 끝난 상태에서 바로 학습 시작할 때 씀.
10. **`d52c2d5`** `CocoInstanceDataset.__getitem__`이 이미지 로드 실패
    (`UnidentifiedImageError` 등, 압축해제 중 잘린 파일)하면 경고만 찍고
    다른 인덱스로 최대 5번 재시도 — 파일 하나 때문에 DataLoader worker가
    죽어서 34시간짜리 학습 전체가 멈추는 것 방지.
11. **`735cd6b`** 평가 진행상황 출력 — `evaluate_coco`의 print 간격을
    하드코딩 200장에서 `max(1, len(image_ids)//10)`으로(데이터셋 크기에
    맞춰 대략 10번 출력), train.py의 per-epoch 평가 호출도 `verbose=True`로
    바꿔서 몇 분씩 아무 출력 없이 멈춘 것처럼 보이던 문제 해결.

## 이 과정에서 겪은 문제들 (재발 가능성 있음, 참고)

- **RunPod 네트워크 스토리지 + 파일 많은 zip**: `unzip`으로 118287개 파일
  압축 해제가 시간당 5만개 수준으로 극도로 느렸음(거의 멈춘 것처럼 보임).
  - 파일 존재 확인을 `os.path.exists()`로 파일마다 하면 그 자체도 매우
    느림(네트워크 stat 콜) — `os.listdir()` 한 번으로 세트 만들어 비교할 것.
  - `zipfile.ZipFile` 객체 **하나**를 여러 스레드가 동시에 `.extract()`하면
    `BadZipFile: Overlapped entries (possible zip bomb)` 에러 발생 —
    스레드마다 **별도 ZipFile 핸들**이 필요 (`threading.local()` 패턴).
  - 압축 해제 중 프로세스를 강제 종료하면 일부 파일이 잘린 채 남아
    나중에 `BadZipFile`/`PIL.UnidentifiedImageError` 유발 → 10번 항목의
    재시도 로직으로 지금은 학습이 이걸로 안 죽는다.
  - COCO train2017.zip 자체에도 최소 몇 개 파일이 CRC 에러남(예:
    `000000310085.jpg`) — 원본 아카이브 문제로 보이고, 그냥 스킵 처리 중.
- **NCCL 기본 timeout(10분)**: DDP + rank0 전용 느린 평가 조합이면 job이
  죽을 수 있음 (6번 항목에서 이미 수정).
- **(IIDALAB 서버 한정, RunPod에선 해당 없음)** systemd `Linger=no`면 SSH
  세션이 끊기는 순간 tmux 안에 있어도 프로세스가 통째로 죽는다. RunPod은
  전용 컨테이너라 이 문제 자체가 없다.

## 코드 구조 요약

```
maskrcnn/
├── config.py           # 하이퍼파라미터 (논문/Detectron 출처 주석 포함)
├── data/
│   ├── coco.py         # CocoInstanceDataset — augment/mask_downsample 지원, 이미지 로드 실패 시 재시도
│   └── transforms.py   # 리사이즈+정규화+패딩, hflip_image_and_target, mask_downsample
├── model/               # ResNet-FPN, RPN, RoIAlign heads, 전체 조립(mask_rcnn.py)
├── evaluate.py          # load_checkpoint, predict_original, evaluate_coco,
│                        # precision_recall_f1_curve, per_class_ap
└── utils/visualize.py   # draw_detections, paste_mask

tools/
├── train.py             # 학습 루프. DDP 지원, freeze-at/grad-checkpoint/
│                        # mask-downsample/hflip, 최종 리포트 자동 생성
├── evaluate.py           # 체크포인트 -> COCO mAP만 (CLI)
├── plot_metrics.py       # PR/F1 curve + AP breakdown 시각화 (generate_report 재사용 가능)
├── plot_training_curve.py  # 학습 로그 -> loss/lr/mAP 곡선
├── make_coco_subset.py   # 서브셋 JSON 생성 (랜덤 N장 또는 --images-dir로 존재하는 것만)
├── setup_runpod.sh       # RunPod 초기 셋업(의존성+COCO 다운로드)
└── infer.py              # 체크포인트 -> 추론 -> 시각화 저장
```

## 다음에 할 일 후보

- train2017 전체 다운로드/압축 해제 끝나면 더 큰(또는 전체) 서브셋으로 재학습
- GPU 메모리 여유 있음(48GB 중 20~25GB만 사용 중) — `--batch-size` 더
  올려보고 `--lr`도 그에 맞춰 선형 스케일링(canonical: batch16=lr0.02)
- 지금 5-epoch/20k 결과 확인 후 필요하면 `--resume`으로 더 길게 확장
- `tools/plot_training_curve.py`, `tools/plot_metrics.py`로 결과 시각화
- PR #1 merge 여부는 사용자가 판단 (지금은 열려만 있음)
