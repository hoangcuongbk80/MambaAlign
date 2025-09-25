#!/usr/bin/env python3
"""

Training script for MambaAlign (RGB + auxiliary modality).

Usage example:
    python train.py \
      --data_root /path/to/MulSen_AD --dataset mulsenad --object capsule \
      --out_dir experiments/mamba_capsule --epochs 300 --batch_size 8 --gpus 0

Main features:
 - Integrates MambaAlign model (MambaAlign.py)
 - Supports MulSen-AD and RealIAD_D3 dataset classes (assumes files created earlier)
 - Synthetic anomaly injection (image-space by default)
 - Losses: segmentation focal + truncated-L1, image-level focal
 - AdamW optimizer with backbone LR multiplier (model.get_param_groups)
 - Mixed precision (torch.cuda.amp)
 - Checkpointing and optional validation (pixel- and image-level AUROC if sklearn available)
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

# import model and dataset classes (assumes they exist in same folder or installed)
from MambaAlign import MambaAlign
# datasets implemented earlier:
from MulsenAD import MulsenAD
from RealIAD_D3 import RealIAD_D3

# try to import sklearn for AUROC computation during validation (optional)
try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# -------------------------
# Utility functions
# -------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_output_dir(out_dir: str):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -------------------------
# Synthetic mask generator
# -------------------------
def generate_random_blob_mask(H: int, W: int, num_blobs: int = 3, max_radius: int = 64) -> np.ndarray:
    """
    Generate a binary mask with random elliptical/gaussian blobs.
    Returns mask as uint8 numpy array shape (H,W) with values {0,1}.
    """
    mask = np.zeros((H, W), dtype=np.uint8)
    for _ in range(num_blobs):
        # random center
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)
        # random axes
        ry = np.random.randint(max(5, max_radius // 8), max(10, max_radius))
        rx = np.random.randint(max(5, max_radius // 8), max(10, max_radius))
        theta = np.random.rand() * 2 * math.pi
        # draw ellipse via param eqn
        y, x = np.ogrid[0:H, 0:W]
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        x_rel = x - cx
        y_rel = y - cy
        # rotation
        xr = x_rel * cos_t + y_rel * sin_t
        yr = -x_rel * sin_t + y_rel * cos_t
        ellipse = (xr**2) / (rx**2) + (yr**2) / (ry**2) <= 1.0
        mask[ellipse] = 1
    # apply small gaussian smoothing to soften edges (optional)
    try:
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
        mask = (mask > 0.2).astype(np.uint8)
    except Exception:
        # no scipy -> keep binary
        pass
    return mask


def inject_image_space_anomalies(rgb: torch.Tensor, aux: torch.Tensor, mask_np: np.ndarray, noise_std: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply synthetic perturbation to rgb and aux tensors within mask areas.
    - rgb: Tensor (B,3,H,W) in [0,1] or normalized; we perturb in image space *before* normalization if possible.
    - aux: Tensor (B,C,H,W)
    - mask_np: numpy array (H,W) {0,1} or (B,H,W)
    Returns perturbed (rgb_p, aux_p)
    Note: This function assumes tensors are in floating range roughly [-1,1] or [0,1]; perturbation is additive Gaussian.
    """
    device = rgb.device
    B = rgb.shape[0]
    H, W = mask_np.shape[-2], mask_np.shape[-1] if mask_np.ndim == 3 else mask_np.shape
    # ensure mask has batch dim
    if mask_np.ndim == 2:
        mask_batch = np.stack([mask_np] * B, axis=0)
    else:
        mask_batch = mask_np  # (B,H,W)

    mask = torch.from_numpy(mask_batch).to(device).unsqueeze(1).float()  # (B,1,H,W)

    # perturb RGB: additive gaussian noise scaled by local std (approx)
    noise_rgb = torch.randn_like(rgb) * noise_std
    rgb_p = rgb * (1 - mask) + (rgb + noise_rgb) * mask

    # perturb aux modality similarly
    noise_aux = torch.randn_like(aux) * noise_std
    aux_p = aux * (1 - mask) + (aux + noise_aux) * mask

    return rgb_p, aux_p


# -------------------------
# Losses
# -------------------------
def focal_loss_logits(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
    """
    Focal loss for logits (binary).
    logits: (B,1,H,W) or (B,H,W) or (B,N,...)
    targets: same shape (binary 0/1)
    """
    # flatten last dims
    logits = logits.view(-1)
    targets = targets.view(-1).float()
    prob = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    mod = (1.0 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * mod * ce_loss
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def truncated_l1_loss(pred: torch.Tensor, target: torch.Tensor, tau: float = 0.3, reduction: str = "mean"):
    """
    Truncated L1 between pred (probabilities in [0,1]) and target (0/1)
    l = min(|pred - target|, tau)
    """
    diff = torch.abs(pred - target)
    t = torch.clamp(diff, max=tau)
    if reduction == "mean":
        return t.mean()
    elif reduction == "sum":
        return t.sum()
    return t


# -------------------------
# Training / Validation loops
# -------------------------
def validate(model: nn.Module, dataloader: DataLoader, device: torch.device, max_batches: Optional[int] = None):
    model.eval()
    total_loss = 0.0
    n = 0
    seg_auc_list = []
    img_auc_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            rgb = batch["rgb"].to(device)
            aux = batch.get("aux") or batch.get("normals") or batch.get("p3d") or batch.get("depth")
            aux = aux.to(device) if aux is not None else torch.zeros(rgb.shape[0], 1, rgb.shape[2], rgb.shape[3], device=device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)

            logits, score, _ = model(rgb, aux, return_aux=False)
            logits_sig = torch.sigmoid(logits)
            # segmentation loss metrics for monitoring
            seg_loss = F.binary_cross_entropy_with_logits(logits, mask.float())
            total_loss += float(seg_loss.item())
            n += 1

            # compute AUROC if sklearn available
            if _HAS_SKLEARN:
                try:
                    # pixel-level AUROC (flatten per-image)
                    y_true = mask.cpu().numpy().reshape(mask.shape[0], -1)
                    y_score = logits_sig.cpu().numpy().reshape(mask.shape[0], -1)
                    for j in range(mask.shape[0]):
                        try:
                            auc = roc_auc_score(y_true[j], y_score[j])
                            seg_auc_list.append(auc)
                        except Exception:
                            pass
                    # image-level AUROC
                    img_true = label.cpu().numpy()
                    img_scores = score.detach().cpu().numpy()
                    try:
                        img_auc = roc_auc_score(img_true, img_scores)
                        img_auc_list.append(img_auc)
                    except Exception:
                        pass
                except Exception:
                    pass

            if max_batches and i + 1 >= max_batches:
                break

    avg_loss = total_loss / max(1, n)
    seg_auc = np.mean(seg_auc_list) if seg_auc_list else None
    img_auc = np.mean(img_auc_list) if img_auc_list else None
    return {"val_loss": avg_loss, "seg_auc": seg_auc, "img_auc": img_auc}


def train(
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu_ids else "cpu")
    print("Using device:", device)

    # model
    model = MambaAlign(x_in_ch=args.aux_ch, pretrained_backbone=args.pretrained_backbone)
    model = model.to(device)

    # dataset selection
    if args.dataset.lower() == "mulsenad":
        dataset = MulsenAD(root=args.data_root, object_name=args.object, split="train", resize=(args.img_size, args.img_size))
        val_dataset = MulsenAD(root=args.data_root, object_name=args.object, split="test", resize=(args.img_size, args.img_size))
        aux_key = "aux"  # MulsenAD returns 'aux'
    elif args.dataset.lower() == "realiadd3":
        dataset = RealIAD_D3(root=args.data_root, category=args.object, split="train", resize=(args.img_size, args.img_size), require_normals=True)
        val_dataset = RealIAD_D3(root=args.data_root, category=args.object, split="test", resize=(args.img_size, args.img_size), require_normals=True)
        aux_key = "normals"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # optimizer param groups via model helper (deduplicated)
    param_groups = model.get_param_groups(base_lr=args.base_lr, backbone_lr_mult=args.backbone_lr_mult, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[240, 270], gamma=0.4)

    # optionally resume
    start_epoch = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    if args.resume:
        assert os.path.exists(args.resume), f"Resume checkpoint not found: {args.resume}"
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        scheduler.load_state_dict(ck["sched"])
        if "scaler" in ck and args.use_amp:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"Resumed checkpoint {args.resume} epoch {start_epoch}")

    out_dir = make_output_dir(args.out_dir)
    print("Output dir:", out_dir)

    # training loop
    best_val_metric = -1.0
    global_step = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs-1}", unit="batch")
        for batch in pbar:
            rgb = batch["rgb"].to(device)
            aux = batch.get(aux_key)
            if aux is None:
                # try alternative keys
                aux = batch.get("p3d") or batch.get("normals") or batch.get("depth")
            aux = aux.to(device) if aux is not None else torch.zeros(rgb.shape[0], args.aux_ch, rgb.shape[2], rgb.shape[3], device=device)
            mask = batch["mask"].to(device)
            # label integer for image-level
            label = batch["label"].to(device).float()

            B, _, H, W = rgb.shape

            # --- synthetic anomaly injection (image-space) ---
            # generate masks per-sample
            synthetic_masks = []
            for b in range(B):
                m = generate_random_blob_mask(H, W, num_blobs=np.random.randint(1, 4), max_radius=min(H, W)//4)
                synthetic_masks.append(m)
            synthetic_masks = np.stack(synthetic_masks, axis=0)  # (B,H,W)

            rgb_pert, aux_pert = inject_image_space_anomalies(rgb, aux, synthetic_masks, noise_std=args.synth_noise)

            # For training the segmentation head, we'll use perturbed images (rgb_pert, aux_pert) as inputs
            # and synthetic_masks as ground-truth anomalies.
            # To keep training stable, with some probability use clean images (no synthetic anomalies)
            if args.synth_prob < 1.0:
                use_mask = np.random.rand(B) < args.synth_prob
                for i in range(B):
                    if not use_mask[i]:
                        synthetic_masks[i] = np.zeros((H, W), dtype=np.uint8)
                        rgb_pert[i] = rgb[i]
                        aux_pert[i] = aux[i]

            # forward pass (mixed precision)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                logits, score, _ = model(rgb_pert, aux_pert, return_aux=False)  # logits (B,1,H,W), score (B,)
                # segmentation probabilities
                prob = torch.sigmoid(logits)

                # prepare targets
                mask_t = torch.from_numpy(synthetic_masks).to(device).unsqueeze(1).float()  # (B,1,H,W)

                # losses:
                loss_focal_seg = focal_loss_logits(logits, mask_t, alpha=args.focal_alpha, gamma=args.focal_gamma)
                loss_trunc = truncated_l1_loss(prob, mask_t, tau=args.trunc_tau)
                # image-level label: if any mask pixel present then label=1 else 0
                img_label = (mask_t.view(B, -1).sum(dim=1) > 0).float()
                loss_focal_img = F.binary_cross_entropy_with_logits(score, img_label)  # alternative: focal on logits
                # Combine losses with weights
                loss = args.w_focal_seg * loss_focal_seg + args.w_trunc * loss_trunc + args.w_img * loss_focal_img

            scaler.scale(loss).backward()
            # gradient clipping optional
            if args.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            global_step += 1
            pbar.set_postfix({'loss': epoch_loss / (global_step + 1 - (start_epoch * len(train_loader))), 'lr': optimizer.param_groups[-1]['lr']})

        scheduler.step()

        # epoch end: validation
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            val_stats = validate(model, val_loader, device, max_batches=args.val_max_batches)
            print(f"Epoch {epoch}: val stats: {val_stats}")
            # save best by image-AUC if available otherwise by val_loss
            metric = val_stats.get("img_auc") if val_stats.get("img_auc") is not None else -val_stats.get("val_loss", 1e9)
            if metric is not None and metric > best_val_metric:
                best_val_metric = metric
                ckpt = {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": vars(args),
                }
                if args.use_amp:
                    ckpt["scaler"] = scaler.state_dict()
                torch.save(ckpt, str(out_dir / "best_checkpoint.pth"))
                print("Saved best checkpoint.")

        # periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            if args.use_amp:
                ckpt["scaler"] = scaler.state_dict()
            torch.save(ckpt, str(out_dir / f"checkpoint_epoch_{epoch}.pth"))
            print(f"Saved checkpoint at epoch {epoch}")

    print("Training finished.")


# -------------------------
# CLI and defaults
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--dataset", type=str, default="mulsenad", choices=["mulsenad", "realiadd3"])
    parser.add_argument("--object", type=str, default=None, help="Object/category name (if dataset requires it)")
    parser.add_argument("--out_dir", type=str, default="outputs/mamba", help="Output directory for checkpoints/logs")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=1e-4)
    parser.add_argument("--backbone_lr_mult", type=float, default=0.25)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--pretrained_backbone", action="store_true", default=False)
    parser.add_argument("--aux_ch", type=int, default=3, help="channels of auxiliary modality (normals -> 3)")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume")
    parser.add_argument("--gpus", type=str, default="", help="comma-separated GPU ids (not used if single GPU)")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--val_max_batches", type=int, default=200)
    parser.add_argument("--save_best_metric", type=str, default="img_auc")
    parser.add_argument("--synth_prob", type=float, default=0.5, help="Per-sample probability to apply synthetic anomaly")
    parser.add_argument("--synth_noise", type=float, default=0.2, help="std of synthetic gaussian noise")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--trunc_tau", type=float, default=0.3)
    parser.add_argument("--w_focal_seg", type=float, default=1.0)
    parser.add_argument("--w_trunc", type=float, default=1.0)
    parser.add_argument("--w_img", type=float, default=0.5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    # GPU selection (simple)
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    # create out dir
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    train(args)
