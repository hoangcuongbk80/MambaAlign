# MambaAlign

**MambaAlign** — Multimodal alignment-aware anomaly detection (RGB + auxiliary modality such as surface normals / IR / depth).
This repository integrates Per-Modal Mamba Modules (PMM) with QuadSnake Visual State-Space Stack (QSVSS), Cross Mamba Interaction (CMI) and Alignment-Aware Fusion (AAF) into a single PyTorch codebase for anomaly detection and segmentation.

---

## Repository layout

```
MambaAlign/
├── MambaAlign.py                 # Main model wiring (backbone + PMM + CMI + AAF + decoder + classifier)
├── rgbx_backbone.py              # Two-branch backbone (RGB + X)
├── pmm_qsvss.py                  # Per-Modal Mamba Module + QSVSS block
├── cross_mamba_cmi.py            # Cross Mamba Interaction (CMI)
├── alignment_aware_fusion.py     # Alignment-Aware Fusion (AAF)
├── MulsenAD.py                   # Dataset loader for MulSen-AD (RGB + IR / aux)
├── RealIAD_D3.py                 # Dataset loader for Real-IAD D³ (RGB + surface normals)
├── RGBD_dataset.py               # Generic RGB-D dataset loader (optional)
├── train.py                      # Training script (synthetic anomalies, checkpointing)
├── eval.py                       # Evaluation script (AUROC, best-F1, visualizations)
├── requirements.txt              # Suggested Python packages
└── README.md                     # This file
```

---

## Quick summary / features

* Two-branch backbone with shared convolutions and per-modality normalization.
* Per-Modal Mamba Module (PMM) — QSVSS-like state-space processing per modality.
* Cross Mamba Interaction (CMI) — cross-conditioned recurrence between modalities at top-level features.
* Alignment-Aware Fusion (AAF) — top-down multi-scale fusion robust to small spatial misalignments.
* Decoder + heads: segmentation head (pixel map) and global image-level anomaly score.
* Dataset loaders included for MulSen-AD, RealIAD-D³ and generic RGB-D.
* Training & evaluation scripts:

  * `train.py` — synthetic anomaly injection (image-space default), mixed precision, optimizer, scheduler, checkpointing.
  * `eval.py` — pixel / image AUROC, best-F1 search, per-image CSV, visualization saving.

---

## Requirements

Minimum recommended:

* Python 3.8+
* PyTorch (version matching your CUDA if using GPU)
* torchvision
* numpy
* pillow
* tqdm

Optional (helpful):

* scikit-learn (AUROC)
* scipy (synthetic mask smoothing)
* open3d (point-cloud processing; not required if using normals only)

Install via pip (example):

```bash
pip install torch torchvision numpy pillow tqdm
# optional:
pip install scikit-learn scipy open3d
```

For GPU, follow the PyTorch official install instructions: [https://pytorch.org](https://pytorch.org)

---

## Preparing datasets

### MulSen-AD

Keep the original repo directory structure. Example:

```
MulSen_AD/
  <object>/
    RGB/train/...
    RGB/test/<anomaly_type>/...
    RGB/GT/<anomaly_type>/...
    Infrared/...
```

Usage:

```python
from MulsenAD import MulsenAD
ds = MulsenAD(root="/path/to/MulSen_AD", object_name="capsule", split="train", resize=(512,512))
```

### RealIAD-D³ (surface normals)

RealIAD D³ includes pseudo-3D maps (surface normals). Place category folders such that each category contains RGB and pseudo3D (normals) folders. The loader converts common encodings of normals into per-pixel unit vectors.

Usage:

```python
from RealIAD_D3 import RealIAD_D3
ds = RealIAD_D3(root="/path/to/RealIAD_D3", category="some_cat", split="train",
                resize=(512,512), require_normals=True)
```

### RGB-D

This project supports RGB-D experiments via the `RGBD_dataset.py` loader.

**Options when using RGB-D:**

1. **Depth as single-channel auxiliary**

   * Use depth directly as the auxiliary input.
   * Set `--aux_ch 1` (or `x_in_ch=1` when constructing `MambaAlign`).
   * Ensure `RGBD_dataset` is configured with the correct `depth_scale` and `max_depth` (if you normalize depth to `[0,1]`).
   * Example dataset instantiation (in Python):

   ```python
   from RGBD_dataset import RGBDDataset
   ds = RGBDDataset(
       root="/path/to/RGBD",
       category="some_cat",
       split="train",
       resize=(512,512),
       depth_scale=1000.0,    # if depth stored as uint16 mm
       max_depth=10.0,        # meters, for normalization
       normalize_depth=True,
       require_depth=True,
       return_paths=True,
   )
   ```

   Then create the model with `x_in_ch=1`:

   ```python
   from MambaAlign import MambaAlign
   model = MambaAlign(x_in_ch=1)
   ```

---

## Training

Example commands:

**MulSen-AD**

```bash
python train.py \
  --data_root /path/to/MulSen_AD \
  --dataset mulsenad \
  --object capsule \
  --out_dir outputs/mamba_capsule \
  --epochs 300 \
  --batch_size 8 \
  --img_size 512 \
  --aux_ch 1 \
  --use_amp \
  --pretrained_backbone
```

**RealIAD-D³ (normals)**

```bash
python train.py \
  --data_root /path/to/RealIAD_D3 \
  --dataset realiadd3 \
  --object some_category \
  --out_dir outputs/mamba_realiad \
  --epochs 200 \
  --batch_size 8 \
  --img_size 512 \
  --aux_ch 3 \
  --use_amp \
  --pretrained_backbone
```

**RGB-D (depth as aux)**

```bash
python train.py \
  --data_root /path/to/RGBD \
  --dataset rgbd \
  --object some_cat \
  --out_dir outputs/mamba_rgbd \
  --epochs 200 \
  --batch_size 8 \
  --img_size 512 \
  --aux_ch 1 \
  --use_amp
```

**RGB-D (normals as aux, precomputed)**

```bash
python train.py \
  --data_root /path/to/RGBD \
  --dataset rgbd \
  --object some_cat \
  --out_dir outputs/mamba_rgbd_normals \
  --epochs 200 \
  --batch_size 8 \
  --img_size 512 \
  --aux_ch 3 \
  --use_amp
```

Key flags:

* `--aux_ch` — number of channels for the auxiliary modality: normals → `3`, depth/IR → `1`.
* `--synth_prob` — probability to apply a synthetic anomaly per sample (default in `train.py`).
* `--use_amp` — enable mixed precision (recommended).
* `--pretrained_backbone` — use ImageNet-pretrained backbone weights.

See `train.py` header comments for loss terms, hyperparameters, and checkpoint settings.

---

## Evaluation

Run `eval.py` with a saved checkpoint:

```bash
python eval.py \
  --data_root /path/to/RealIAD_D3 \
  --dataset realiadd3 \
  --object some_category \
  --checkpoint outputs/mamba_realiad/best_checkpoint.pth \
  --out_dir results/mamba_realiad_eval \
  --img_size 512 \
  --batch_size 4 \
  --save_vis
```

For RGB-D specify proper `--dataset` / dataset loader and `--aux_ch` (1 for depth, 3 for normals). The script outputs:

* `eval_results.csv` — per-image scores and pixel metrics (when GT exists).
* `vis/heatmaps/` and `vis/overlays/` — saved visualizations when `--save_vis` is provided.
* `summary.txt` — simple aggregate metrics.

Notes:

* Pixel-level metrics are computed for images that contain GT masks (anomalous images typically have masks).
* Image-level AUROC requires `scikit-learn`; otherwise it won’t be computed.

---

Performance tips:

* For small batch sizes, prefer `GroupNorm` instead of `BatchNorm` in fusion/PMM blocks.
* If GPU memory is limited, reduce batch size or enable memory-friendly options (e.g., sequential route evaluation in PMM if implemented).
* Use gradient accumulation if you need larger effective batch sizes.

---

## Troubleshooting

* If the dataset loader cannot find your modality directories, check folder names and pass directory-name candidates (many dataset loaders accept name variations).
* If checkpoints won't load because keys mismatch, inspect the checkpoint dict: training script stores `ckpt["model"] = model.state_dict()`. `eval.py` accepts both raw `state_dict` and dict-wrapped checkpoints.
* If you see unstable results when training with very small batches, change BatchNorm → GroupNorm in key modules.

---