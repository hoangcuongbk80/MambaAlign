#!/usr/bin/env python3
"""
Evaluation script for MambaAlign anomaly detection & segmentation.

Outputs:
 - CSV summary with per-image file names, image-level score, pixel AUROC (if mask exists),
   best-F1, best-threshold, IoU at best threshold (when GT mask available).
 - Optional saved heatmaps and overlay visualizations.

Usage example:
    python eval.py \
      --data_root /data/RealIAD_D3 --dataset realiadd3 --object capsule \
      --checkpoint outputs/mamba_capsule/best_checkpoint.pth \
      --out_dir results/mamba_capsule_eval --batch_size 4 --save_vis
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageOps, ImageFilter
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# local imports (assumes these exist in same folder / PYTHONPATH)
from MambaAlign import MambaAlign
from MulsenAD import MulsenAD
from RealIAD_D3 import RealIAD_D3

# sklearn optional
try:
    from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def save_heatmap(prob_map: np.ndarray, out_path: str):
    """
    Save single-channel float probability map [0,1] as 8-bit heatmap PNG (grayscale).
    """
    arr = np.clip(prob_map * 255.0, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    im.save(out_path)


def overlay_heatmap_on_rgb(rgb_img: Image.Image, prob_map: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlay a grayscale/probability heatmap onto RGB image.
    - rgb_img: PIL RGB
    - prob_map: numpy array shape (H,W) in [0,1]
    returns PIL RGB
    """
    # create colorized heat by mapping prob to red channel (simple)
    h, w = prob_map.shape
    heat = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat).convert("L").resize(rgb_img.size, resample=Image.BILINEAR)
    # colorize: use red channel as heat
    black = Image.new("RGB", rgb_img.size, (0, 0, 0))
    heat_rgb = Image.merge("RGB", (heat_img, Image.new("L", rgb_img.size, 0), Image.new("L", rgb_img.size, 0)))
    out = Image.blend(rgb_img.convert("RGB"), heat_rgb, alpha=alpha)
    return out


def compute_best_threshold_metrics(y_true_flat: np.ndarray, y_score_flat: np.ndarray, thresholds: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    """
    For a single image (flattened arrays), compute best F1 over grid of thresholds.
    Return (best_f1, best_thresh, iou_at_best).
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101)
    best_f1 = -1.0
    best_t = 0.5
    best_iou = 0.0
    for t in thresholds:
        pred = (y_score_flat >= t).astype(np.uint8)
        # skip trivial cases where GT all zeros
        if y_true_flat.sum() == 0 and pred.sum() == 0:
            f1 = 1.0
            iou = 1.0
        else:
            # compute F1 safely
            tp = int(((pred == 1) & (y_true_flat == 1)).sum())
            fp = int(((pred == 1) & (y_true_flat == 0)).sum())
            fn = int(((pred == 0) & (y_true_flat == 1)).sum())
            prec = tp / (tp + fp + 1e-12)
            rec = tp / (tp + fn + 1e-12)
            if (prec + rec) > 0:
                f1 = 2 * prec * rec / (prec + rec)
            else:
                f1 = 0.0
            den = tp + fp + fn
            iou = tp / (den + 1e-12) if den > 0 else 1.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_iou = iou
    return best_f1, best_t, best_iou


# -------------------------
# Main evaluation
# -------------------------
def evaluate(
    args,
):
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    print("Device:", device)

    # load model
    print("Loading model...")
    model = MambaAlign(x_in_ch=args.aux_ch, pretrained_backbone=False)
    ck = torch.load(args.checkpoint, map_location=device)
    if "model" in ck:
        state = ck["model"]
    else:
        state = ck
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    # dataset selection
    if args.dataset.lower() == "mulsenad":
        dataset = MulsenAD(root=args.data_root, object_name=args.object, split="test", resize=(args.img_size, args.img_size))
        aux_key = "aux"
    elif args.dataset.lower() == "realiadd3":
        dataset = RealIAD_D3(root=args.data_root, category=args.object, split="test", resize=(args.img_size, args.img_size), require_normals=True)
        aux_key = "normals"
    else:
        raise ValueError("Unsupported dataset: choose 'mulsenad' or 'realiadd3'")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # prepare output folders
    out_dir = Path(args.out_dir)
    ensure_dir(str(out_dir))
    if args.save_vis:
        vis_dir = out_dir / "vis"
        ensure_dir(str(vis_dir))
        heat_dir = vis_dir / "heatmaps"
        overlay_dir = vis_dir / "overlays"
        ensure_dir(str(heat_dir))
        ensure_dir(str(overlay_dir))

    # accumulators
    image_scores = []
    image_labels = []
    per_image_pixel_auc = []
    per_image_best_f1 = []
    per_image_best_thr = []
    per_image_iou = []
    entries = []  # CSV lines

    # iterate
    print("Running inference on dataset ({} images)".format(len(dataset)))
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval"):
            rgb = batch["rgb"].to(device)
            aux = batch.get(aux_key)
            if aux is None:
                aux = batch.get("p3d") or batch.get("normals") or batch.get("depth") or batch.get("aux")
            aux = aux.to(device) if aux is not None else torch.zeros(rgb.shape[0], args.aux_ch, rgb.shape[2], rgb.shape[3], device=device)
            masks = batch["mask"]  # CPU tensor (B,1,H,W)
            labels = batch["label"].numpy().astype(int)  # (B,)
            paths_meta = batch.get("meta", None)
            # forward
            logits, scores, aux_info = model(rgb, aux, return_aux=False)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)
            scores_np = scores.detach().cpu().numpy()  # (B,)

            B = probs.shape[0]
            for b in range(B):
                prob_map = probs[b, 0]  # (H,W), float
                score_val = float(scores_np[b])
                label_val = int(labels[b])
                image_scores.append(score_val)
                image_labels.append(label_val)

                # prepare file id/name
                meta = paths_meta[b] if paths_meta is not None else {}
                fname = meta.get("rgb_path", f"img_{len(entries)}")
                fname_only = Path(fname).name

                # If mask exists and contains at least one positive pixel, compute pixel AUROC & best-F1
                mask_b = masks[b].cpu().numpy()
                mask_flat = mask_b.reshape(-1)
                prob_flat = prob_map.reshape(-1)

                pixel_auc = None
                best_f1 = None
                best_thr = None
                best_iou = None
                # Only evaluate pixel metrics if GT mask contains both pos and neg pixels
                if mask_flat.sum() > 0 and mask_flat.sum() < mask_flat.size:
                    # pixel AUROC
                    if _HAS_SKLEARN:
                        try:
                            pixel_auc = float(roc_auc_score(mask_flat, prob_flat))
                        except Exception:
                            pixel_auc = None
                    else:
                        pixel_auc = None

                    # compute best threshold metrics
                    best_f1, best_thr, best_iou = compute_best_threshold_metrics(mask_flat.astype(np.uint8), prob_flat, thresholds=np.linspace(0.0, 1.0, 101))
                    per_image_pixel_auc.append(pixel_auc if pixel_auc is not None else np.nan)
                    per_image_best_f1.append(best_f1)
                    per_image_best_thr.append(best_thr)
                    per_image_iou.append(best_iou)
                else:
                    # no positive pixels -> skip pixel-level metrics but keep entries
                    per_image_pixel_auc.append(np.nan)
                    per_image_best_f1.append(np.nan)
                    per_image_best_thr.append(np.nan)
                    per_image_iou.append(np.nan)

                # optionally save heatmap and overlay
                if args.save_vis:
                    heat_path = heat_dir / (fname_only + "_heat.png")
                    overlay_path = overlay_dir / (fname_only + "_overlay.png")
                    save_heatmap(prob_map, str(heat_path))
                    # attempt overlay; need an RGB PIL image
                    try:
                        rgb_path = meta.get("rgb_path", None)
                        if rgb_path is not None and Path(rgb_path).exists():
                            rgb_pil = Image.open(rgb_path).convert("RGB")
                            # resize to model size if needed
                            rgb_pil = rgb_pil.resize((prob_map.shape[1], prob_map.shape[0]), resample=Image.BILINEAR)
                        else:
                            # reconstruct RGB from tensor
                            # rgb[b] is normalized; try to unnormalize using ImageNet stats if appropriate
                            rgb_tensor = rgb[b].cpu()
                            # attempt approximate unnormalize
                            mean = np.array([0.485, 0.456, 0.406])
                            std = np.array([0.229, 0.224, 0.225])
                            rgb_np = rgb_tensor.numpy().transpose(1, 2, 0)
                            rgb_np = (rgb_np * std) + mean
                            rgb_np = np.clip(rgb_np * 255.0, 0, 255).astype(np.uint8)
                            rgb_pil = Image.fromarray(rgb_np)
                        ov = overlay_heatmap_on_rgb(rgb_pil, prob_map, alpha=args.overlay_alpha)
                        ov.save(str(overlay_path))
                    except Exception as e:
                        # still save heatmap only (overlay failed)
                        pass

                # record CSV entry
                entries.append({
                    "fname": fname_only,
                    "full_path": str(fname),
                    "image_label": int(label_val),
                    "image_score": float(score_val),
                    "pixel_auc": float(pixel_auc) if pixel_auc is not None else "",
                    "best_f1": float(best_f1) if best_f1 is not None else "",
                    "best_thr": float(best_thr) if best_thr is not None else "",
                    "best_iou": float(best_iou) if best_iou is not None else "",
                })

    # Aggregate results
    image_scores_arr = np.array(image_scores)
    image_labels_arr = np.array(image_labels)
    # image-level AUROC
    image_auc = None
    if _HAS_SKLEARN:
        try:
            image_auc = float(roc_auc_score(image_labels_arr, image_scores_arr))
        except Exception:
            image_auc = None

    # pixel-level AUROC mean (exclude NaNs)
    pix_auc_arr = np.array(per_image_pixel_auc)
    pix_valid = ~np.isnan(pix_auc_arr)
    pixel_auc_mean = float(np.nanmean(pix_auc_arr)) if pix_valid.any() else None

    # mean best-F1 (only where computed)
    f1_arr = np.array(per_image_best_f1)
    f1_valid = ~np.isnan(f1_arr)
    mean_best_f1 = float(np.nanmean(f1_arr[f1_valid])) if f1_valid.any() else None

    print("=== Evaluation summary ===")
    print("Images:", len(entries))
    print("Image-level AUROC:", image_auc)
    print("Pixel-level mean AUROC (over images with GT):", pixel_auc_mean)
    print("Mean best-F1 (over images with GT):", mean_best_f1)

    # write CSV
    csv_path = out_dir / "eval_results.csv"
    keys = ["fname", "full_path", "image_label", "image_score", "pixel_auc", "best_f1", "best_thr", "best_iou"]
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for e in entries:
            row = {k: e.get(k, "") for k in keys}
            writer.writerow(row)
    print("Wrote per-image CSV to", str(csv_path))

    # write summary text
    with open(str(out_dir / "summary.txt"), "w") as f:
        f.write("Evaluation summary\n")
        f.write("==================\n")
        f.write(f"Num images: {len(entries)}\n")
        f.write(f"Image AUROC: {image_auc}\n")
        f.write(f"Pixel AUROC mean: {pixel_auc_mean}\n")
        f.write(f"Mean best-F1: {mean_best_f1}\n")
    print("Saved summary to", str(out_dir / "summary.txt"))

    print("Done.")


# -------------------------
# CLI
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="realiadd3", choices=["mulsenad", "realiadd3"])
    parser.add_argument("--object", type=str, default=None, help="category/object name if dataset requires it")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/mamba_eval")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--aux_ch", type=int, default=3, help="auxiliary input channels (normals -> 3)")
    parser.add_argument("--device", type=str, default="cuda", help="'cuda' or 'cpu'")
    parser.add_argument("--save_vis", action="store_true", help="save heatmaps and overlays")
    parser.add_argument("--overlay_alpha", type=float, default=0.45, help="alpha for overlay visualization")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    evaluate(args)
