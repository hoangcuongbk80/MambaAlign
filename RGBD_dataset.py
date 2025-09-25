"""
RGB-D Dataset loader for paired RGB and depth images (train/test splits).

Features:
 - Auto-detects RGB and depth folders inside category subfolders
 - Handles depth encodings:
     * uint16 (common: depth in millimeters) -> convert to meters via `depth_scale` (default 1000.0)
     * float32 (single-channel) interpret as meters directly
     * uint8 -> treated as scaled depth (0..255) and scaled by depth_scale if provided, else normalized by max_depth
     * 3-channel depth images (rare) -> will average channels
 - Optional normalization of depth to [0,1] via `max_depth` (recommended)
 - Optional mask loading; fallback to zero mask if missing
 - Synchronized resizing for rgb, depth and mask
 - Returns dict { "rgb", "depth", "mask", "label", "meta" }

Usage:
    ds = RGBDDataset(root="/data/RGBD", category="somecat", split="train", resize=(512,512))
    sample = ds[0]
"""

from pathlib import Path
from typing import Optional, Tuple, List, Callable, Dict, Any
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

# supported image extensions for RGB/depth/mask
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _is_image_file(fn: str) -> bool:
    return Path(fn).suffix.lower() in _IMAGE_EXTS


def _pil_open(path: str) -> Optional[Image.Image]:
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


def _load_depth_image_as_float(path: str, depth_scale: float = 1000.0) -> Optional[np.ndarray]:
    """
    Load an image using PIL and return a float32 numpy array with depth in meters (or raw floats if float image).
    Heuristics:
      - If image dtype is uint16: assume depth is in millimeters -> divide by depth_scale (default 1000.0) -> meters
      - If image dtype is uint8: convert to float [0,255]; user should set max_depth to normalize later
      - If image dtype is float32: use as-is (assume meters)
      - If multi-channel, average channels (rare)
    Returns:
      depth_arr: np.ndarray (H, W) dtype float32 or None if load fails
    """
    if path is None:
        return None
    try:
        with Image.open(path) as im:
            arr = np.array(im)
    except Exception:
        return None

    if arr.ndim == 3:
        # multi-channel depth (RGB-encoded) -> average channels
        arr = arr[..., :3].mean(axis=2)

    arr = arr.astype(np.float32)

    # dtype-based heuristics are less reliable after read by PIL (it returns uint8/uint16/float)
    # We check ranges to guess encoding:
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        # already float (likely meters) -> just cast
        return arr.astype(np.float32)
    # If array values exceed 1000, likely uint16 in mm
    maxv = float(np.nanmax(arr)) if arr.size > 0 else 0.0
    if maxv > 1000.0:
        # assume mm (uint16) -> convert to meters
        return (arr / float(depth_scale)).astype(np.float32)
    else:
        # treat as 8-bit or small-range depth; return as float (user decides how to normalize)
        return arr.astype(np.float32)


class RGBDDataset(Dataset):
    """
    Dataset class for RGB-D image pairs.

    Args:
        root: dataset root directory with category subfolders
        category: optional category to restrict to (if None, iterate all categories)
        split: 'train' or 'test'
        rgb_dirnames: possible names for RGB folder under category (default common names)
        depth_dirnames: possible names for depth folder (default 'Depth' variants)
        resize: (H, W) target size to resize RGB/depth/mask before transforms (PIL size uses (W,H))
        transform_rgb: callable(PIL->Tensor), default: ToTensor + ImageNet normalization
        transform_depth: callable(np.ndarray or PIL->Tensor -> Tensor) ; default: convert float array to Tensor and optionally normalize
        mask_transform: callable(PIL->Tensor) ; default ToTensor
        depth_scale: if uint16 depth in millimeters, divide by this scale to get meters (default 1000.0)
        max_depth: used when normalize_depth=True to clip/scale depth into [0,1]; default 10.0 meters
        normalize_depth: if True convert depth (meters) -> depth / max_depth clipped to [0,1]
        require_depth: if True skip samples missing a depth file
        return_paths: if True include paths in sample['meta']
    """

    def __init__(
        self,
        root: str,
        category: Optional[str] = None,
        split: str = "train",
        rgb_dirnames: Optional[List[str]] = None,
        depth_dirnames: Optional[List[str]] = None,
        resize: Optional[Tuple[int, int]] = (512, 512),
        transform_rgb: Optional[Callable] = None,
        transform_depth: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        depth_scale: float = 1000.0,
        max_depth: float = 10.0,
        normalize_depth: bool = True,
        require_depth: bool = True,
        return_paths: bool = False,
    ):
        assert split in {"train", "test"}
        self.root = Path(root)
        self.category = category
        self.split = split
        self.resize = resize
        self.depth_scale = depth_scale
        self.max_depth = max_depth
        self.normalize_depth = normalize_depth
        self.require_depth = require_depth
        self.return_paths = return_paths

        self.rgb_dirnames = rgb_dirnames or ["RGB", "rgb", "Image", "image"]
        self.depth_dirnames = depth_dirnames or ["Depth", "depth", "D", "depth_maps"]

        # default transforms
        if transform_rgb is None:
            self.transform_rgb = T.Compose(
                [T.ToTensor(),
                 T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
        else:
            self.transform_rgb = transform_rgb

        if transform_depth is None:
            # Default depth transform: input is np.ndarray (H,W) in meters or raw,
            # We convert to torch tensor (1,H,W) and optionally normalize by max_depth
            def _depth_transform(depth_arr: np.ndarray) -> torch.Tensor:
                # depth_arr: float32 HxW (meters or raw units)
                if depth_arr is None:
                    # return zeros with resize shape
                    if self.resize:
                        H, W = self.resize
                    else:
                        H, W = 512, 512
                    return torch.zeros(1, H, W, dtype=torch.float32)
                # Convert to float32 numpy
                darr = depth_arr.astype(np.float32)
                # If normalization requested, assume darr in meters and divide by max_depth
                if self.normalize_depth:
                    darr = darr / float(self.max_depth)
                    darr = np.clip(darr, 0.0, 1.0)
                # Expand dims -> (1,H,W)
                t = torch.from_numpy(darr).unsqueeze(0).float()
                return t
            self.transform_depth = _depth_transform
        else:
            self.transform_depth = transform_depth

        if mask_transform is None:
            self.mask_transform = T.Compose([T.ToTensor()])
        else:
            self.mask_transform = mask_transform

        # build sample index
        self.samples = []
        self._build_index()

    # -------------------------
    # helpers to find modality folders and collect images
    # -------------------------
    def _list_categories(self) -> List[str]:
        return [p.name for p in sorted(self.root.iterdir()) if p.is_dir()]

    def _find_subdir(self, obj_root: Path, candidates: List[str]) -> Optional[Path]:
        for c in candidates:
            p = obj_root / c
            if p.exists() and p.is_dir():
                return p
        return None

    def _collect_images_in_split(self, base_dir: Path) -> Dict[str, List[Path]]:
        """
        Collect images under base_dir/<split> and return mapping anomaly_name -> list of image Path
        """
        out = {}
        split_dir = base_dir / self.split
        if not split_dir.exists():
            return out
        # check for subfolders
        subdirs = [p for p in sorted(split_dir.iterdir()) if p.is_dir()]
        if not subdirs:
            imgs = [p for p in sorted(split_dir.rglob("*")) if p.is_file() and _is_image_file(str(p))]
            out["root"] = imgs
            return out
        for sub in subdirs:
            imgs = [p for p in sorted(sub.rglob("*")) if p.is_file() and _is_image_file(str(p))]
            out[sub.name] = imgs
        return out

    def _try_find_mask(self, rgb_path: Path, rgb_root: Path, anomaly_name: str) -> Optional[Path]:
        gt_root = rgb_root / "GT"
        if not gt_root.exists():
            return None
        candidate = gt_root / anomaly_name / rgb_path.name
        if candidate.exists():
            return candidate
        candidate2 = gt_root / anomaly_name / (rgb_path.stem + ".png")
        if candidate2.exists():
            return candidate2
        matches = list((gt_root / anomaly_name).rglob(rgb_path.stem + ".*")) if (gt_root / anomaly_name).exists() else []
        return matches[0] if matches else None

    # -------------------------
    # index builder
    # -------------------------
    def _build_index(self):
        cats = [self.category] if self.category else self._list_categories()
        if not cats:
            raise RuntimeError(f"No categories found in root: {self.root}")

        for cat in cats:
            obj_root = self.root / cat
            if not obj_root.exists():
                continue

            rgb_dir = self._find_subdir(obj_root, self.rgb_dirnames)
            depth_dir = self._find_subdir(obj_root, self.depth_dirnames)

            if rgb_dir is None:
                # fallback: images directly under obj_root/<split>
                if (obj_root / self.split).exists():
                    rgb_dir = obj_root
                else:
                    continue

            rgb_map = self._collect_images_in_split(rgb_dir)
            if not rgb_map:
                continue

            # depth_map may be empty
            depth_map = {}
            if depth_dir and depth_dir.exists():
                depth_map = self._collect_images_in_split(depth_dir)

            for anomaly_name, rgb_list in rgb_map.items():
                label = 0 if anomaly_name.lower() in {"good", "normal", "ok"} else 1
                for rgb_path in rgb_list:
                    rgb_path = Path(rgb_path)
                    # attempt to find matching depth by filename in same anomaly folder
                    depth_candidate = None
                    if depth_map:
                        lst = depth_map.get(anomaly_name, [])
                        for cand in lst:
                            if Path(cand).name == rgb_path.name or Path(cand).stem == rgb_path.stem:
                                depth_candidate = Path(cand)
                                break
                        # fallback search entire depth split
                        if depth_candidate is None:
                            depth_split_dir = depth_dir / self.split if depth_dir else None
                            if depth_split_dir and depth_split_dir.exists():
                                all_depths = [p for p in sorted(depth_split_dir.rglob("*")) if p.is_file() and _is_image_file(str(p))]
                                for cand in all_depths:
                                    if cand.stem == rgb_path.stem:
                                        depth_candidate = cand
                                        break

                    if self.require_depth and depth_candidate is None:
                        continue

                    # mask
                    mask_candidate = self._try_find_mask(rgb_path, rgb_dir, anomaly_name)
                    if mask_candidate is None and (obj_root / "GT").exists():
                        candidates = list((obj_root / "GT").rglob(rgb_path.name))
                        if candidates:
                            mask_candidate = candidates[0]

                    entry = {
                        "category": cat,
                        "rgb_path": str(rgb_path),
                        "depth_path": str(depth_candidate) if depth_candidate is not None else None,
                        "mask_path": str(mask_candidate) if mask_candidate is not None else None,
                        "label": int(label),
                    }
                    self.samples.append(entry)

        if not self.samples:
            raise RuntimeError(f"No samples found for split={self.split} at {self.root}")

    # -------------------------
    # helper to load images
    # -------------------------
    def _load_rgb_pil(self, path: Optional[str]) -> Optional[Image.Image]:
        if path is None:
            return None
        try:
            im = Image.open(path).convert("RGB")
            return im
        except Exception:
            return None

    def _load_mask_pil(self, path: Optional[str]) -> Optional[Image.Image]:
        if path is None:
            return None
        try:
            im = Image.open(path).convert("L")
            return im
        except Exception:
            return None

    # -------------------------
    # __getitem__
    # -------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.samples[idx]
        rgb_p = entry["rgb_path"]
        depth_p = entry["depth_path"]
        mask_p = entry["mask_path"]
        label = entry["label"]

        rgb_img = self._load_rgb_pil(rgb_p)
        if rgb_img is None:
            raise RuntimeError(f"Missing RGB image {rgb_p}")

        depth_arr = None
        if depth_p is not None:
            depth_arr = _load_depth_image_as_float(depth_p, depth_scale=self.depth_scale)

        mask_img = self._load_mask_pil(mask_p) if mask_p else None

        # resizing
        if self.resize is not None:
            tgt = (self.resize[1], self.resize[0])  # PIL uses (W,H)
            rgb_img = rgb_img.resize(tgt, resample=Image.BILINEAR)
            if depth_arr is not None:
                # resize depth via numpy interpolation
                depth_arr = np.array(Image.fromarray(depth_arr).resize(tgt, resample=Image.BILINEAR))
            if mask_img is not None:
                mask_img = mask_img.resize(tgt, resample=Image.NEAREST)
            else:
                # create zero mask if missing
                pass

        # default mask fallback to zero
        if mask_img is None:
            h, w = rgb_img.size[1], rgb_img.size[0]
            mask_img = Image.fromarray(np.zeros((h, w), dtype=np.uint8))

        # transforms
        rgb_t = self.transform_rgb(rgb_img)

        depth_t = self.transform_depth(depth_arr)  # returns (1,H,W) tensor

        mask_t = self.mask_transform(mask_img)
        # ensure binary
        mask_t = (mask_t > 0.5).float()
        if mask_t.dim() == 2:
            mask_t = mask_t.unsqueeze(0)

        sample = {
            "rgb": rgb_t.float(),
            "depth": depth_t.float(),
            "mask": mask_t.float(),
            "label": int(label),
            "meta": {"rgb_path": rgb_p, "depth_path": depth_p, "mask_path": mask_p, "category": entry.get("category")},
        }
        if self.return_paths:
            sample["paths"] = (rgb_p, depth_p, mask_p)
        return sample


# -------------------------
# CLI quick smoke test
# -------------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--resize", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--require-depth", action="store_true")
    args = parser.parse_args()

    ds = RGBDDataset(root=args.root, category=args.category, split=args.split, resize=tuple(args.resize),
                     require_depth=args.require_depth, return_paths=True)
    print(f"Found {len(ds)} samples")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    for batch in dl:
        print("rgb:", batch["rgb"].shape, "depth:", batch["depth"].shape, "mask:", batch["mask"].shape, "label:", batch["label"])
        # basic sanity: depth min/max
        d = batch["depth"][0]
        print("depth stats min/max:", float(d.min()), float(d.max()))
        break

    print("Smoke test done.")
