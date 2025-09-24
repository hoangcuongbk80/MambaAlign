"""
MulSen-AD dataset loader compatible with the folder layout described in the MulSen-AD repo/README.

Directory layout (example):
MulSen_AD/
  └── capsule/
       ├── RGB/
       │    ├── train/
       │    │   └── 0.png ...
       │    ├── test/
       │    │   ├── hole/
       │    │   │   └── 0.png ...
       │    │   ├── crack/
       │    │   └── good/
       │    └── GT/
       │        ├── hole/
       │        │  └── 0.png ...
       │        └── good/
       ├── Infrared/
       │    ├── train/ ...
       │    ├── test/ ...
       │    └── GT/ ...
       └── Pointcloud/  (optional - may contain images or .ply/.pcd)
See README in the official repo for full dataset description. :contentReference[oaicite:2]{index=2}
"""

import os
import glob
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Any

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import csv


def _is_image_file(fname: str) -> bool:
    suf = os.path.splitext(fname)[1].lower()
    return suf in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


class MulsenAD(Dataset):
    """
    PyTorch Dataset for MulSen-AD (RGB + Infrared / Pointcloud-image fallback).

    Args:
        root: root path to `MulSen_AD` dataset folder.
        object_name: the object-category folder to load (e.g., "capsule"). If None, dataset will load images
                     across all object subfolders (concatenates).
        split: "train" or "test"
        rgb_folder_name: typically "RGB" (folder containing RGB/train, RGB/test, RGB/GT)
        aux_folder_name: e.g., "Infrared" or "Pointcloud" (folder containing aux modality)
        transforms: torchvision transforms applied jointly to RGB and aux+mask (expects PIL images)
                    If None, a default transform converts images to tensors and normalizes RGB.
        mask_transform: optional transform applied only to mask (e.g., resizing without color jitter)
        resize: optional tuple (H, W) to resize all images/masks to a fixed size (applied before transforms)
        require_aux: if True, samples missing aux modality will be skipped (default False: missing aux -> zeros)
    """

    def __init__(
        self,
        root: str,
        object_name: Optional[str] = None,
        split: str = "train",
        rgb_folder_name: str = "RGB",
        aux_folder_name: str = "Infrared",
        transforms: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        resize: Optional[Tuple[int, int]] = None,
        require_aux: bool = False,
        return_paths: bool = False,
    ):
        assert split in {"train", "test"}, "split must be 'train' or 'test'"
        self.root = Path(root)
        self.object_name = object_name
        self.split = split
        self.rgb_folder_name = rgb_folder_name
        self.aux_folder_name = aux_folder_name
        self.resize = resize
        self.require_aux = require_aux
        self.return_paths = return_paths

        # default transforms (if user didn't pass any)
        if transforms is None:
            # normalize RGB with ImageNet stats; aux modality is kept as single-channel [0,1]
            self.transforms = T.Compose(
                [
                    T.ToTensor(),  # converts PIL to [0,1] float tensor CxHxW
                ]
            )
        else:
            self.transforms = transforms

        # For masks we only convert to tensor (no normalization)
        if mask_transform is None:
            self.mask_transform = T.Compose([T.ToTensor()])
        else:
            self.mask_transform = mask_transform

        # Build list of sample entries (one per RGB image)
        self.samples = []  # each entry: dict with rgb_path, aux_path (maybe None), mask_path (maybe None), label, obj_name
        self._build_index()

    # -------------------------
    # Internal index construction
    # -------------------------
    def _list_objects(self) -> List[str]:
        """Return list of object subfolders under root if object_name is None."""
        entries = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir():
                entries.append(p.name)
        return entries

    def _build_index(self):
        """
        Walk dataset directories and populate self.samples.

        Rules:
         - If split == 'train', we read from <object>/<RGB>/train (and aux train) and label everything as normal (0).
         - If split == 'test', we read from <object>/<RGB>/test/<anomaly_type_or_good> and label 'good' as 0 else 1.
         - GT masks: try to find corresponding mask in <object>/<RGB>/GT/<anomaly_type>/<same_filename> (or same stem with .png).
            If mask file missing -> mask_path = None (dataset will return zero mask).
         - Aux modality: try <object>/<aux_folder>/test/... or train/... with the same subfolder and filename. If missing and require_aux -> skip.
        """
        objects = [self.object_name] if self.object_name else self._list_objects()
        if not objects:
            raise ValueError(f"No object folders found in dataset root: {self.root}")

        for obj in objects:
            obj_root = self.root / obj
            rgb_root = obj_root / self.rgb_folder_name
            aux_root = obj_root / self.aux_folder_name

            # train
            if self.split == "train":
                rgb_train_dir = rgb_root / "train"
                if not rgb_train_dir.exists():
                    continue
                # collect all image files in train (usually 'good' images)
                for p in sorted(rgb_train_dir.rglob("*")):
                    if p.is_file() and _is_image_file(str(p)):
                        rgb_path = p
                        # derive auxiliary path: mirror the relative path under aux_root if exists, else search aux_root/train/*
                        aux_path = None
                        candidate_aux = aux_root / "train" / p.relative_to(rgb_train_dir)
                        if candidate_aux.exists() and candidate_aux.is_file():
                            aux_path = candidate_aux
                        else:
                            # fallback: search aux train folder for same base filename
                            matches = list((aux_root / "train").rglob(p.name)) if (aux_root / "train").exists() else []
                            aux_path = matches[0] if matches else None

                        # masks: train likely has no masks; check RGB/GT/good/data.csv or GT/good/*
                        mask_path = None
                        gt_candidate = rgb_root / "GT" / "good" / p.name
                        if gt_candidate.exists():
                            mask_path = gt_candidate
                        # else look for any mask with same stem in GT/good
                        if mask_path is None and (rgb_root / "GT" / "good").exists():
                            matches = list((rgb_root / "GT" / "good").rglob(p.stem + ".*"))
                            mask_path = matches[0] if matches else None

                        if self.require_aux and aux_path is None:
                            # skip this sample
                            continue

                        entry = {
                            "obj": obj,
                            "rgb_path": str(rgb_path),
                            "aux_path": str(aux_path) if aux_path is not None else None,
                            "mask_path": str(mask_path) if mask_path is not None else None,
                            "label": 0,
                        }
                        self.samples.append(entry)

            # test
            else:
                rgb_test_dir = rgb_root / "test"
                if not rgb_test_dir.exists():
                    continue
                # iterate anomaly-type folders inside test/, including 'good'
                for anomaly_dir in sorted(rgb_test_dir.iterdir()):
                    if not anomaly_dir.is_dir():
                        continue
                    anomaly_name = anomaly_dir.name
                    label = 0 if anomaly_name.lower() in {"good", "normal", "ok"} else 1
                    for p in sorted(anomaly_dir.rglob("*")):
                        if p.is_file() and _is_image_file(str(p)):
                            rgb_path = p
                            # aux path mirror
                            aux_path = None
                            candidate_aux = aux_root / "test" / anomaly_name / p.name
                            if candidate_aux.exists():
                                aux_path = candidate_aux
                            else:
                                # fallback search
                                matches = list((aux_root / "test").rglob(p.name)) if (aux_root / "test").exists() else []
                                aux_path = matches[0] if matches else None

                            # mask path: try <rgb_root>/GT/<anomaly_name>/<same_name>
                            mask_path = None
                            gt_folder = rgb_root / "GT" / anomaly_name
                            if gt_folder.exists():
                                # prefer mask image with exact same filename
                                candidate_mask = gt_folder / p.name
                                if candidate_mask.exists():
                                    mask_path = candidate_mask
                                else:
                                    # try same stem with png
                                    candidate_mask = gt_folder / (p.stem + ".png")
                                    if candidate_mask.exists():
                                        mask_path = candidate_mask
                                    else:
                                        # try reading data.csv mapping if present
                                        csv_path = gt_folder / "data.csv"
                                        if csv_path.exists():
                                            # attempt to parse csv: assume it lists filenames in first column
                                            try:
                                                with open(csv_path, "r") as f:
                                                    reader = csv.reader(f)
                                                    for row in reader:
                                                        if len(row) == 0:
                                                            continue
                                                        # try matching first column to filename
                                                        if row[0].strip() == p.name or row[0].strip() == p.stem:
                                                            # If CSV provides mask file name in second column, use that
                                                            if len(row) > 1:
                                                                candidate = gt_folder / row[1].strip()
                                                                if candidate.exists():
                                                                    mask_path = candidate
                                                                    break
                                            except Exception:
                                                mask_path = None

                            if self.require_aux and aux_path is None:
                                # skip sample if paper requires aux
                                continue

                            entry = {
                                "obj": obj,
                                "rgb_path": str(rgb_path),
                                "aux_path": str(aux_path) if aux_path is not None else None,
                                "mask_path": str(mask_path) if mask_path is not None else None,
                                "label": int(label),
                            }
                            self.samples.append(entry)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split={self.split} under {self.root} (object={self.object_name})")

    # -------------------------
    # Dataset interface
    # -------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Optional[str]) -> Optional[Image.Image]:
        if path is None:
            return None
        path = str(path)
        if not os.path.exists(path):
            return None
        try:
            img = Image.open(path)
            img = img.convert("RGB")  # for RGB or three-channel aux
            return img
        except Exception:
            try:
                # if it's grayscale (Infrared) convert to 'L' then to RGB later if needed
                img = Image.open(path).convert("L")
                return img
            except Exception:
                return None

    def _load_mask(self, path: Optional[str]) -> Optional[Image.Image]:
        if path is None:
            return None
        if not os.path.exists(path):
            return None
        try:
            m = Image.open(path).convert("L")
            return m
        except Exception:
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.samples[idx]
        rgb_p = entry["rgb_path"]
        aux_p = entry["aux_path"]
        mask_p = entry["mask_path"]

        # load images
        rgb_img = self._load_image(rgb_p)
        aux_img = self._load_image(aux_p) if aux_p is not None else None
        mask_img = self._load_mask(mask_p) if mask_p is not None else None

        # resizing if requested (keep same interpolation for both images)
        if self.resize is not None:
            size = (self.resize[1], self.resize[0])  # PIL uses (W,H)
            if rgb_img is not None:
                rgb_img = rgb_img.resize(size, resample=Image.BILINEAR)
            if aux_img is not None:
                # preserve grayscale if aux loaded as L
                if aux_img.mode == "L":
                    aux_img = aux_img.resize(size, resample=Image.BILINEAR)
                else:
                    aux_img = aux_img.resize(size, resample=Image.BILINEAR)
            if mask_img is not None:
                mask_img = mask_img.resize(size, resample=Image.NEAREST)

        # If mask missing -> create zero mask with same size as rgb
        if mask_img is None and rgb_img is not None:
            # create single-channel zero mask PIL image
            mask_img = Image.fromarray(np.zeros((rgb_img.size[1], rgb_img.size[0]), dtype=np.uint8))

        # Transforms: we apply transforms separately to RGB and aux to allow different normalizations.
        # Default transforms provided during init simply convert to tensor.
        rgb_t = self.transforms(rgb_img) if rgb_img is not None else torch.zeros(3, *(self.resize if self.resize else (rgb_img.size[1], rgb_img.size[0])))
        # Aux modality: convert to single-channel tensor if original aux is grayscale, else keep 3-ch.
        if aux_img is None:
            # create zero aux of shape (1, H, W)
            if self.resize is not None:
                h, w = self.resize
            else:
                h, w = rgb_img.size[1], rgb_img.size[0]
            aux_t = torch.zeros(1, h, w, dtype=torch.float32)
        else:
            # If aux is single-channel ('L') convert to 1-channel tensor
            if aux_img.mode == "L":
                aux_t = T.ToTensor()(aux_img)  # (1,H,W)
            else:
                # if aux was converted to RGB, yield 3-channel
                aux_t = T.ToTensor()(aux_img)

        mask_t = self.mask_transform(mask_img)
        # mask transform may produce float in [0,1]. Convert to binary {0,1}
        mask_t = (mask_t > 0.5).float()

        sample = {
            "rgb": rgb_t.float(),
            "aux": aux_t.float(),
            "mask": mask_t.float(),
            "label": int(entry["label"]),
            "meta": {"rgb_path": rgb_p, "aux_path": aux_p, "mask_path": mask_p, "obj": entry["obj"]},
        }

        if self.return_paths:
            sample["paths"] = (rgb_p, aux_p, mask_p)

        return sample


# -------------------------
# Mini smoke test (not executed on import)
# -------------------------
if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Path to MulSen_AD dataset root (folder containing object subfolders)")
    parser.add_argument("--object", type=str, default=None, help="Object folder to load (default: all)")
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    ds = MulsenAD(root=args.root, object_name=args.object, split=args.split, resize=(256, 256))
    print(f"Found {len(ds)} samples")
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=2)
    for batch in dl:
        print("rgb:", batch["rgb"].shape, "aux:", batch["aux"].shape, "mask:", batch["mask"].shape, "label:", batch["label"])
        break
