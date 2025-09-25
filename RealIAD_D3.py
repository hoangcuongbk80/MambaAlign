"""
RealIAD_D3 dataset focused on RGB + surface normals (pseudo-3D).

- Loads RGB images and corresponding pseudo-3D images that encode surface normals.
- Converts pseudo-3D images into 3-channel tensors where each pixel is a unit normal vector in [-1,1].
- Handles common encodings:
    * 8-bit RGB normals (0..255) -> mapped to [-1,1]
    * 16-bit normals -> scaled appropriately
    * float images (mode 'F') that already store normals in [-1,1] or 0..1 (attempt to infer)
- Returns dict: { "rgb": Tensor(3,H,W), "normals": Tensor(3,H,W), "mask": Tensor(1,H,W), "label": int, "meta": {...} }

Usage example:
    ds = RealIAD_D3(root="/data/RealIAD", category="some_cat", split="train", resize=(512,512))
    sample = ds[0]
    rgb = sample["rgb"]         # (3,H,W)
    normals = sample["normals"] # (3,H,W), per-pixel unit vectors

Author: Assistant (as senior CV/DL engineer)
"""
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Any, List
import os
import warnings

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# Supported image extensions
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


def _is_image_file(fn: str) -> bool:
    return Path(fn).suffix.lower() in _IMAGE_EXTS


def _pil_to_rgb(img: Image.Image) -> Image.Image:
    """Ensure PIL image is RGB"""
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _load_pil(path: Optional[str]) -> Optional[Image.Image]:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = Image.open(str(p))
        return img
    except Exception:
        return None


def normals_pil_to_tensor_unit(img: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Convert a PIL image that encodes surface normals to a tensor of shape (3,H,W)
    with per-pixel unit length and values in [-1,1].

    Heuristics used:
    - If dtype is uint8: assume range [0,255] -> divide by 255 -> [0,1] -> map to [-1,1] via *2 - 1
    - If dtype is uint16: divide by 65535 -> [0,1] -> map to [-1,1]
    - If dtype is float:
        - If values appear in [-1.5, 1.5] assume already in [-1,1] and clamp.
        - Else if values in [0,1.5] assume [0,1] encoding -> map to [-1,1]
        - Else scale by max absolute value then map/clip.
    - If the image is single-channel (L), replicate to 3 channels (not typical for normals).
    - After mapping to [-1,1] we renormalize each pixel vector to unit length (avoid divide-by-zero).
    """
    if img is None:
        raise ValueError("normals_pil_to_tensor_unit received None image")

    # Optionally resize before processing (caller can resize via PIL .resize beforehand)
    # Convert to numpy array for robust dtype handling
    arr = np.array(img)

    # If single-channel, replicate into 3-channels (rare for normals)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    # arr shape now (H, W, C) expected C == 3
    if arr.shape[-1] != 3:
        # If there is an alpha channel, drop it
        if arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            # replicate channels to get 3
            arr = np.repeat(arr[..., :1], 3, axis=-1)

    # Convert to float32
    arr_f = arr.astype(np.float32)

    # Detect dtype-based scaling
    if arr.dtype == np.uint8:
        arr_f = arr_f / 255.0  # now in [0,1]
        arr_f = arr_f * 2.0 - 1.0  # -> [-1,1]
    elif arr.dtype == np.uint16:
        arr_f = arr_f / 65535.0
        arr_f = arr_f * 2.0 - 1.0
    else:
        # float images: try to infer
        max_val = float(np.nanmax(arr_f))
        min_val = float(np.nanmin(arr_f))
        # If already in [-1.5,1.5], assume normals already encoded in [-1,1]
        if max_val <= 1.5 and min_val >= -1.5:
            arr_f = np.clip(arr_f, -1.0, 1.0)
        else:
            # If values appear non-negative and <= 1.5, likely in [0,1] range
            if min_val >= 0.0 and max_val <= 1.5:
                arr_f = arr_f / max(1.0, max_val)  # scale to [0,1]
                arr_f = arr_f * 2.0 - 1.0
            else:
                # fallback: scale by max absolute value then clamp
                max_abs = max(abs(min_val), abs(max_val), 1.0)
                arr_f = arr_f / max_abs
                arr_f = np.clip(arr_f, -1.0, 1.0)

    # Now arr_f should be in [-1,1], shape (H,W,3)
    # Convert to tensor (C,H,W)
    t = torch.from_numpy(arr_f).permute(2, 0, 1).contiguous().float()  # (3,H,W)

    # Per-pixel normalization to unit length
    # compute norm across channel dim
    eps = 1e-6
    norms = torch.norm(t, p=2, dim=0, keepdim=True)  # (1,H,W)
    norms = torch.clamp(norms, min=eps)
    t_unit = t / norms  # (3,H,W)

    return t_unit


class RealIAD_D3(Dataset):
    """
    Dataset for Real-IAD D3 restricted to RGB + surface normals (pseudo-3D).
    Only returns RGB + normals + mask + label + meta.

    Args:
        root: path to dataset root (folder containing category subfolders)
        category: optional category name (if None -> iterate all categories)
        split: "train" or "test"
        rgb_dirnames: folder name candidates for RGB images inside category (default tries common names)
        p3d_dirnames: folder name candidates for pseudo-3D normals
        resize: optional (H, W) to resize images/masks before transforms (PIL order: (W,H))
        transform_rgb: callable(PIL->Tensor). Default: ToTensor + ImageNet norm
        transform_normals: callable(PIL->Tensor). Default: converts to unit normals via normals_pil_to_tensor_unit
        mask_transform: callable(PIL->Tensor). Default: ToTensor (binary)
        require_normals: if True skip samples missing normals (default True)
        return_paths: if True returned sample includes file paths in meta
    """

    def __init__(
        self,
        root: str,
        category: Optional[str] = None,
        split: str = "train",
        rgb_dirnames: Optional[List[str]] = None,
        p3d_dirnames: Optional[List[str]] = None,
        resize: Optional[Tuple[int, int]] = (512, 512),
        transform_rgb: Optional[Callable] = None,
        transform_normals: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        require_normals: bool = True,
        return_paths: bool = False,
    ):
        assert split in {"train", "test"}
        self.root = Path(root)
        self.category = category
        self.split = split
        self.resize = resize
        self.require_normals = require_normals
        self.return_paths = return_paths

        # default folder name variants
        self.rgb_dirnames = rgb_dirnames or ["RGB", "rgb", "Image", "image"]
        self.p3d_dirnames = p3d_dirnames or ["Pseudo3D", "pseudo3d", "pseudo_3d", "p3d", "Normals", "normals", "surface_normals"]

        # default transforms
        if transform_rgb is None:
            self.transform_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform_rgb = transform_rgb

        if transform_normals is None:
            # default: use the robust converter (normals_pil_to_tensor_unit) as a callable wrapper
            def _normals_transform(pil_img: Image.Image) -> torch.Tensor:
                return normals_pil_to_tensor_unit(pil_img)
            self.transform_normals = _normals_transform
        else:
            self.transform_normals = transform_normals

        if mask_transform is None:
            self.mask_transform = T.Compose([T.ToTensor()])
        else:
            self.mask_transform = mask_transform

        # build index of samples
        self.samples = []  # each entry: dict with rgb_path, normals_path, mask_path, label, category
        self._build_index()

    # -------------------------
    # directory & file helpers
    # -------------------------
    def _list_categories(self) -> List[str]:
        return [p.name for p in sorted(self.root.iterdir()) if p.is_dir()]

    def _find_dir(self, obj_root: Path, candidates: List[str]) -> Optional[Path]:
        for c in candidates:
            p = obj_root / c
            if p.exists() and p.is_dir():
                return p
        return None

    def _collect_images_in_split(self, base_dir: Path) -> Dict[str, List[Path]]:
        """
        Look under base_dir/<split> and return mapping anomaly_name -> list of image paths.
        If images are directly under base_dir/<split> (no subfolders), map "root" -> list.
        """
        out = {}
        split_dir = base_dir / self.split
        if not split_dir.exists():
            return out
        # check subfolders
        has_subdirs = any((p.is_dir() for p in split_dir.iterdir()))
        if not has_subdirs:
            images = [p for p in sorted(split_dir.rglob("*")) if p.is_file() and _is_image_file(str(p))]
            out["root"] = images
            return out
        for sub in sorted(split_dir.iterdir()):
            if not sub.is_dir():
                continue
            imgs = [p for p in sorted(sub.rglob("*")) if p.is_file() and _is_image_file(str(p))]
            out[sub.name] = imgs
        return out

    def _try_find_mask(self, rgb_path: Path, rgb_root: Path, anomaly_name: str) -> Optional[Path]:
        """
        Search for GT mask under rgb_root/GT/<anomaly_name> with the same filename or stem.
        """
        gt_root = rgb_root / "GT"
        if not gt_root.exists():
            return None
        candidate = gt_root / anomaly_name / rgb_path.name
        if candidate.exists():
            return candidate
        candidate2 = gt_root / anomaly_name / (rgb_path.stem + ".png")
        if candidate2.exists():
            return candidate2
        # fallback: any file with same stem inside gt_root/anomaly_name
        matches = list((gt_root / anomaly_name).rglob(rgb_path.stem + ".*")) if (gt_root / anomaly_name).exists() else []
        return matches[0] if matches else None

    # -------------------------
    # index builder
    # -------------------------
    def _build_index(self):
        cats = [self.category] if self.category else self._list_categories()
        if not cats:
            raise RuntimeError(f"No categories found in RealIAD root: {self.root}")

        for cat in cats:
            obj_root = self.root / cat
            if not obj_root.exists():
                continue

            # locate rgb and p3d directories
            rgb_dir = self._find_dir(obj_root, self.rgb_dirnames)
            p3d_dir = self._find_dir(obj_root, self.p3d_dirnames)

            if rgb_dir is None:
                # try fallback: category/<split> directly
                if (obj_root / self.split).exists():
                    rgb_dir = obj_root
                else:
                    # give up on this category
                    continue

            rgb_map = self._collect_images_in_split(rgb_dir)
            if not rgb_map:
                continue

            # p3d_map (may be empty)
            p3d_map = {}
            if p3d_dir and p3d_dir.exists():
                p3d_map = self._collect_images_in_split(p3d_dir)

            # assemble samples
            for anomaly_name, rgb_list in rgb_map.items():
                label = 0 if anomaly_name.lower() in {"good", "normal", "ok"} else 1
                for rgb_path in rgb_list:
                    rgb_path = Path(rgb_path)
                    # try to match p3d by same filename inside the anomaly folder
                    normals_candidate = None
                    if p3d_map:
                        lst = p3d_map.get(anomaly_name, [])
                        for cand in lst:
                            if Path(cand).name == rgb_path.name or Path(cand).stem == rgb_path.stem:
                                normals_candidate = Path(cand)
                                break
                        # fallback search whole p3d split
                        if normals_candidate is None:
                            all_p3d = []
                            p3d_split_dir = p3d_dir / self.split if p3d_dir else None
                            if p3d_split_dir and p3d_split_dir.exists():
                                all_p3d = [p for p in sorted(p3d_split_dir.rglob("*")) if p.is_file() and _is_image_file(str(p))]
                            for cand in all_p3d:
                                if cand.stem == rgb_path.stem:
                                    normals_candidate = cand
                                    break

                    if self.require_normals and normals_candidate is None:
                        # skip
                        continue

                    # mask lookup
                    mask_candidate = self._try_find_mask(rgb_path, rgb_dir, anomaly_name)
                    # fallback try any GT in category
                    if mask_candidate is None and (obj_root / "GT").exists():
                        candidates = list((obj_root / "GT").rglob(rgb_path.name))
                        if candidates:
                            mask_candidate = candidates[0]

                    entry = {
                        "category": cat,
                        "rgb_path": str(rgb_path),
                        "normals_path": str(normals_candidate) if normals_candidate is not None else None,
                        "mask_path": str(mask_candidate) if mask_candidate is not None else None,
                        "label": int(label),
                    }
                    self.samples.append(entry)

        if not self.samples:
            raise RuntimeError(f"No samples found for split={self.split} under {self.root} (category={self.category})")

    # -------------------------
    # dataset interface
    # -------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.samples[idx]
        rgb_p = entry["rgb_path"]
        normals_p = entry["normals_path"]
        mask_p = entry["mask_path"]
        label = entry["label"]

        # load images
        rgb_img = _load_pil(rgb_p)
        normals_img = _load_pil(normals_p) if normals_p else None
        mask_img = _load_pil(mask_p) if mask_p else None

        # if rgb missing, raise (RGB required)
        if rgb_img is None:
            raise RuntimeError(f"Missing RGB image at {rgb_p}")

        # resize images if requested (PIL size expects (W,H))
        if self.resize is not None:
            tgt = (self.resize[1], self.resize[0])
            rgb_img = rgb_img.resize(tgt, resample=Image.BILINEAR)
            if normals_img is not None:
                normals_img = normals_img.resize(tgt, resample=Image.BILINEAR)
            if mask_img is not None:
                mask_img = mask_img.resize(tgt, resample=Image.NEAREST)

        # load mask: if missing produce zero mask
        if mask_img is None:
            w, h = rgb_img.size
            mask_img = Image.fromarray((np.zeros((h, w), dtype=np.uint8)))

        # compute tensors
        rgb_t = self.transform_rgb(_pil_to_rgb(rgb_img))

        # normals: robust conversion to unit vectors
        if normals_img is None:
            # zero normals fallback (should not happen if require_normals=True)
            H, W = rgb_t.shape[1], rgb_t.shape[2]
            normals_t = torch.zeros(3, H, W, dtype=torch.float32)
        else:
            # When using the provided default transform_normals, it expects a PIL image and returns a (3,H,W) tensor
            normals_t = self.transform_normals(normals_img)
            # If the transform returns numpy array ensure tensor
            if isinstance(normals_t, np.ndarray):
                normals_t = torch.from_numpy(normals_t).float()
            # Ensure shape (3,H,W)
            if normals_t.ndim == 2:
                normals_t = normals_t.unsqueeze(0).repeat(3, 1, 1)
            elif normals_t.shape[0] != 3:
                # if transform produced (H,W,3) array, convert
                if normals_t.ndim == 3 and normals_t.shape[2] == 3:
                    normals_t = torch.from_numpy(normals_t).permute(2, 0, 1).float()
                else:
                    # as a last resort, replicate or pad
                    C = normals_t.shape[0]
                    if C < 3:
                        # pad channels
                        pad = torch.zeros(3 - C, normals_t.shape[1], normals_t.shape[2])
                        normals_t = torch.cat([normals_t, pad], dim=0)
                    else:
                        normals_t = normals_t[:3, :, :]

        # mask tensor
        mask_t = self.mask_transform(mask_img)
        # ensure binary
        mask_t = (mask_t > 0.5).float()
        # if mask is (H,W) convert to (1,H,W)
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)

        sample = {
            "rgb": rgb_t.float(),
            "normals": normals_t.float(),
            "mask": mask_t.float(),
            "label": int(label),
            "meta": {"rgb_path": rgb_p, "normals_path": normals_p, "mask_path": mask_p, "category": entry.get("category")},
        }
        if self.return_paths:
            sample["paths"] = (rgb_p, normals_p, mask_p)
        return sample


# -------------------------
# Quick smoke test
# -------------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to RealIAD D3 root")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--resize", type=int, nargs=2, default=(512, 512))
    args = parser.parse_args()

    ds = RealIAD_D3(root=args.root, category=args.category, split=args.split, resize=tuple(args.resize), require_normals=False, return_paths=True)
    print(f"Found {len(ds)} samples")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    for batch in dl:
        print("rgb:", batch["rgb"].shape)
        print("normals:", batch["normals"].shape)
        print("mask:", batch["mask"].shape)
        print("label:", batch["label"])
        # sanity checks on normals
        n = batch["normals"][0]  # (3,H,W)
        norms = torch.norm(n, p=2, dim=0)
        print("norms stats: min, mean, max:", float(norms.min()), float(norms.mean()), float(norms.max()))
        break

    print("Smoke test done.")
