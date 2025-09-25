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
* Dataset loaders included for MulSen-AD and RealIAD-D³ (RealIAD loader converts pseudo-3D to per-pixel unit surface normals).
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

Keep original repo directory structure. Example:

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

Key flags:

* `--aux_ch` — number of channels for auxiliary modality: normals → `3`, depth/IR → `1`.
* `--synth_prob` — probability to apply a synthetic anomaly per sample (default in `train.py`).
* `--use_amp` — enable mixed precision (recommended).
* `--pretrained_backbone` — use ImageNet-pretrained backbone weights (if available).

See `train.py` top comments for loss terms, hyperparameters, and checkpoint settings.

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

Outputs:

* `eval_results.csv` — per-image scores and pixel metrics (when GT exists).
* `vis/heatmaps/` and `vis/overlays/` — saved visualizations when `--save_vis` is provided.
* `summary.txt` — simple aggregate metrics.

Notes:

* Pixel-level metrics are computed for images that contain GT masks (anomalous images typically have masks).
* Image-level AUROC requires `scikit-learn`; otherwise it won’t be computed.

---

## Implementation notes and customization points

* `MambaAlign.py` — top-level model: adjust `x_in_ch`, PMM block counts, `fused_out_ch`, and CMI settings to change capacity.
* `pmm_qsvss.py` — plug in optimized SSM/S4/S6 kernels if available; replace fallback implementations.
* `alignment_aware_fusion.py` — tune `fused_out_ch`, bottleneck ratios, or replace `LocalFusion` blocks to change fusion behavior.
* `RealIAD_D3.py` — default normal transform maps common encodings to unit vectors; if your normals encoding differs, pass a custom `transform_normals` to the dataset.

Performance tips:

* For small batch sizes, prefer `GroupNorm` instead of `BatchNorm` in fusion/PMM blocks.
* If GPU memory is limited, reduce batch size or enable memory-friendly options (e.g., sequential route evaluation in PMM if implemented).
* Use gradient accumulation if you need larger effective batch sizes.

---

## Troubleshooting

* If the dataset loader cannot find your modality directories, check folder names and pass directory-name candidates (many dataset loaders accept name variations).
* If checkpoints won't load because keys mismatch, inspect the checkpoint dict: training script stores `ckpt["model"] = model.state_dict()`. `eval.py` accepts both raw `state_dict` and dict-wrapped checkpoints.
* If you see unstable results when training with very small batches, change BatchNorm -> GroupNorm in key modules.

---