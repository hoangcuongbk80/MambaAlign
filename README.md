```markdown
# MambaAlign

**MambaAlign** — Multimodal alignment-aware anomaly detection (RGB + auxiliary modality such as surface normals / IR / depth).  
This repository integrates Per-Modal Mamba Modules (PMM) with QuadSnake Visual State-Space Stack (QSVSS), Cross Mamba Interaction (CMI), and Alignment-Aware Fusion (AAF) into a single, end-to-end PyTorch codebase for anomaly detection and segmentation.

---

## Repository layout

```

MambaAlign/
├── MambaAlign.py                 # Main model wiring (backbone + PMM + CMI + AAF + decoder + classifier)
├── rgbx\_backbone.py              # Two-branch backbone (RGB + X)
├── pmm\_qsvss.py                  # Per-Modal Mamba Module + QSVSS block
├── cross\_mamba\_cmi.py            # Cross Mamba Interaction (CMI)
├── alignment\_aware\_fusion.py     # Alignment-Aware Fusion (AAF)
├── MulsenAD.py                   # Dataset loader for MulSen-AD (RGB + IR / aux)
├── RealIAD\_D3.py                 # Dataset loader for Real-IAD D³ (RGB + surface normals)
├── RGBD\_dataset.py               # Generic RGB-D dataset loader (optional)
├── train.py                      # Training script (synthetic anomalies, checkpointing)
├── eval.py                       # Evaluation script (AUROC, best-F1, visualizations)
├── requirements.txt              # Suggested Python packages
└── README.md                     # This file

```

---

## Highlights / Features

- **Multimodal backbone** with shared convs and per-modality BatchNorm.
- **Per-Modal Mamba Module (PMM)** — spatial mixing plus per-route SSM processing (QSVSS).
- **Cross Mamba Interaction (CMI)** — content-conditioned SSM recurrence with cross-decoding across modalities.
- **Alignment-Aware Fusion (AAF)** — top-down fusion that tolerates small misalignments.
- **Flexible dataset loaders** for MulSen-AD and RealIAD D³; RealIAD loader treats pseudo-3D as surface normals and converts to per-pixel unit vectors.
- **Complete training & evaluation scripts** including synthetic anomaly injection, mixed-precision training, checkpointing, pixel/image AUROC, best-F1 computation, and visualizations.

---

## Requirements

Recommended Python packages (add or adjust versions as required):

```

python >= 3.8
torch, torchvision
numpy
Pillow
tqdm
scikit-learn          # optional (AUROC)
scipy                 # optional (smoothing for synthetic masks)
open3d                # optional (point-cloud handling; not needed if using normals only)

````

You can install the basics with:

```bash
pip install torch torchvision numpy pillow tqdm scikit-learn
# optional:
pip install scipy open3d
````

> For GPU usage, please install the PyTorch build matching your CUDA version following [https://pytorch.org](https://pytorch.org).

---

## Quick start — smoke tests

1. Put all provided `.py` files into the same folder (project root).
2. Run the module-level smoke tests (each file includes a small test when executed directly). Examples:

```bash
# Model wiring smoke test
python MambaAlign.py

# PMM smoke test
python pmm_qsvss.py

# CMI smoke test
python cross_mamba_cmi.py

# AAF smoke test
python alignment_aware_fusion.py

# Dataset smoke tests (point them at your local data folders)
python MulsenAD.py --root /path/to/MulSen_AD --split train
python RealIAD_D3.py --root /path/to/RealIAD_D3 --split test
```

---

## Preparing data

### MulSen-AD

Maintain the original repo structure:

```
MulSen_AD/
  <object>/
    RGB/train/...
    RGB/test/<anomaly_type>/...
    RGB/GT/<anomaly_type>/...
    Infrared/...
```

Use the loader:

```py
from MulsenAD import MulsenAD
ds = MulsenAD(root="/path/to/MulSen_AD", object_name="capsule", split="train", resize=(512,512))
```

### RealIAD D³ (normals)

Place category folders with RGB and pseudo-3D normals (3-channel images). The loader expects names such as `Pseudo3D`, `pseudo3d`, `Normals`, etc. The loader will convert the normals to unit vectors.

```py
from RealIAD_D3 import RealIAD_D3
ds = RealIAD_D3(root="/path/to/RealIAD_D3", category="some_cat", split="train", resize=(512,512), require_normals=True)
```

---

## Training

A working training command (example):

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

**RealIAD D³ (normals)**

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

* `--aux_ch`: auxiliary channels (normals → `3`)
* `--synth_prob`: probability per-sample to apply synthetic anomaly injection (default 0.5)
* `--use_amp`: use mixed precision (recommended)
* `--pretrained_backbone`: use ImageNet-pretrained backbone weights

See `train.py` header comments for loss weights and hyperparameters.

---

## Evaluation

Run evaluation on a saved checkpoint:

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

* `eval_results.csv` — per-image metrics (image score, pixel AUROC when GT exists, best-F1, best threshold).
* `vis/heatmaps/` and `vis/overlays/` — predicted heatmaps and overlays (when `--save_vis` is set).
* `summary.txt` — aggregated metrics.

---

## File-level notes & where to customize

* `MambaAlign.py`

  * `x_in_ch`: set to `3` for normals, `1` for depth/IR.
  * `get_param_groups()`: produces optimizer groups with backbone lr multiplier.

* `pmm_qsvss.py`

  * Replace `FallbackCausalConv1D` with an S4/S6 implementation if available (adapter should accept `(B,T,C)` input).
  * Set `sequential_routes=True` to reduce memory.

* `cross_mamba_cmi.py`

  * Configure `D` and `N` to match your capacity / compute budget.

* `alignment_aware_fusion.py`

  * `fused_out_ch` controls final feature dimension forwarded to the decoder.

* `RealIAD_D3.py`

  * Default `transform_normals` maps common encodings to per-pixel unit vectors — adjust if your normals encoding differs.

---

## Tips, troubleshooting & best practices

* **Batch size & normalization**: If running with small batches, prefer `GroupNorm` instead of `BatchNorm` along fusion / PMM blocks.
* **GPU memory**: If OOM occurs, reduce batch size, set `sequential_routes=True` in `QSVSSBlock`, or use gradient accumulation.
* **Data loader issues**: If `RealIAD_D3` fails to find normals, check folder names and pass `p3d_dirnames` explicitly when constructing the dataset.
* **Reproducibility**: Use `--seed` in `train.py` to fix RNG; set `torch.backends.cudnn.deterministic = True` if needed (may slow training).
* **Plug-in SSMs**: If you have a high-performance SSM/S4/S6 kernel, adapt its wrapper to implement `SSMInterface.forward(seq: Tensor[B,T,C]) -> Tensor[B,T,C]` and pass a factory to PMM.

---

## License & attribution

This code is provided for research and development. Add a compatible license file (e.g., MIT or Apache 2.0) to the project root before public distribution. When publishing results using MulSen-AD or RealIAD datasets, cite the original dataset papers and repositories.

---

## Contact / Next steps

If you want help with any of the following, I can provide code or patches:

* Replace the fallback SSM with a specific S4/S6 implementation (adapter wrapper).
* Add synchronized augmentations (same random crop/flip for RGB, normals and mask).
* Convert `train.py` synthetic injection to feature-space injection (paper-style) — adjust `MambaAlign` to optionally return fused latent.
* Add unit tests and CI targets.

Happy to help — tell me what you’d like next.

```
```
