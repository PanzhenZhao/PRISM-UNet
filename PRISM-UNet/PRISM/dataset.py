#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified MedicalDataset for medical image segmentation.

Supported datasets:
- Breast ultrasound: BUSI, BUSI-WHU
- Skin lesion: ISIC2017, ISIC2018
- Retinal vessel: CHASEDB1, FIVES

Expected directory structure:
    <root>/<split>/images/
    <root>/<split>/labels/
Optional ROI structure for retinal datasets:
    <root>/<split>/roi/
"""

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


DATASET_META = {
    # ---------------- Breast ultrasound ----------------
    "BUSI": {
        "img_ext": "png",
        "lbl_ext": "png",
        "lbl_suffix": "_mask",
        "task": "ultrasound",
        "has_roi": False,
    },
    "BUSI-WHU": {
        "img_ext": "bmp",
        "lbl_ext": "bmp",
        "lbl_suffix": "_anno",
        "task": "ultrasound",
        "has_roi": False,
    },

    # ---------------- Skin lesion ----------------
    "ISIC2017": {
        "img_ext": "jpg",
        "lbl_ext": "png",
        "lbl_suffix": "_segmentation",
        "task": "skin",
        "has_roi": False,
    },
    "ISIC2018": {
        "img_ext": "jpg",
        "lbl_ext": "png",
        "lbl_suffix": "_segmentation",
        "task": "skin",
        "has_roi": False,
    },

    # ---------------- Retinal vessel ----------------
    "CHASEDB1": {
        "img_ext": "jpg",
        "lbl_ext": "png",
        "lbl_suffix": "_1stHO",
        "task": "retinal",
        "has_roi": True,
        "roi_ext": "png",
        "roi_suffix": "_mask",
    },
    "FIVES": {
        "img_ext": "png",
        "lbl_ext": "png",
        "lbl_suffix": "",
        "task": "retinal",
        "has_roi": True,
        "roi_ext": "png",
        "roi_suffix": "",
    },
}


class MedicalDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        split: str = "train",
        img_size: int = 256,
        augment: bool = False,
        return_roi: bool = False,
        use_imagenet_norm: bool = True,
        mask_threshold: float = 0.0,
    ):
        if dataset_name not in DATASET_META:
            raise ValueError(
                f"Unsupported dataset_name: {dataset_name}. "
                f"Available: {list(DATASET_META.keys())}"
            )

        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.return_roi = return_roi

        meta = DATASET_META[dataset_name]
        self.meta = meta
        self.task = meta["task"]
        self.has_roi = bool(meta.get("has_roi", False))

        self.img_dir = self.root / split / "images"
        self.lbl_dir = self.root / split / "labels"
        self.roi_dir = self.root / split / "roi"

        self.img_ext = meta["img_ext"]
        self.lbl_ext = meta["lbl_ext"]
        self.lbl_suffix = meta["lbl_suffix"]

        self.roi_ext = meta.get("roi_ext", None)
        self.roi_suffix = meta.get("roi_suffix", "")

        self.mask_threshold = float(mask_threshold)

        if use_imagenet_norm:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.0, 0.0, 0.0]
            self.std = [1.0, 1.0, 1.0]

        self.img_paths = self._collect_files(self.img_dir, self.img_ext)
        if not self.img_paths:
            raise FileNotFoundError(f"No images found in: {self.img_dir}")

        missing_labels = [str(p) for p in self.img_paths if self._get_label_path(p) is None]
        if missing_labels:
            preview = "\n".join(missing_labels[:5])
            raise FileNotFoundError(f"Missing label files (showing at most 5):\n{preview}")

        if self.return_roi:
            if not self.has_roi:
                raise ValueError(
                    f"Dataset '{dataset_name}' does not define ROI masks, "
                    f"but return_roi=True was requested."
                )
            missing_rois = [str(p) for p in self.img_paths if self._get_roi_path(p) is None]
            if missing_rois:
                preview = "\n".join(missing_rois[:5])
                raise FileNotFoundError(f"Missing ROI files (showing at most 5):\n{preview}")

    def __len__(self):
        return len(self.img_paths)

    # ----------------------------------------------------
    # File matching helpers
    # ----------------------------------------------------
    def _collect_files(self, directory: Path, ext: str):
        files = sorted(directory.glob(f"*.{ext}"))
        if not files:
            files = sorted(directory.glob(f"*.{ext.upper()}"))
        if not files:
            files = sorted(directory.glob(f"*.{ext.lower()}"))
        return files

    def _find_existing_file(self, directory: Path, stem_with_suffix: str, ext: str) -> Optional[Path]:
        candidates = [
            directory / f"{stem_with_suffix}.{ext}",
            directory / f"{stem_with_suffix}.{ext.upper()}",
            directory / f"{stem_with_suffix}.{ext.lower()}",
        ]
        for p in candidates:
            if p.exists():
                return p
        return None

    def _get_label_path(self, img_path: Path) -> Optional[Path]:
        stem = img_path.stem
        return self._find_existing_file(self.lbl_dir, f"{stem}{self.lbl_suffix}", self.lbl_ext)

    def _get_roi_path(self, img_path: Path) -> Optional[Path]:
        if not self.has_roi:
            return None

        stem = img_path.stem

        if self.roi_ext is not None:
            p = self._find_existing_file(self.roi_dir, f"{stem}{self.roi_suffix}", self.roi_ext)
            if p is not None:
                return p

        if self.dataset_name in {"CHASEDB1", "FIVES"}:
            for suffix in [self.roi_suffix, "_mask", ""]:
                p = self._find_existing_file(self.roi_dir, f"{stem}{suffix}", self.roi_ext)
                if p is not None:
                    return p

        return None

    # ----------------------------------------------------
    # Reading
    # ----------------------------------------------------
    def _read_rgb(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _read_mask(self, path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    # ----------------------------------------------------
    # Augmentation
    # ----------------------------------------------------
    def _should_use_color_jitter(self) -> bool:
        return self.task == "skin"

    def _apply_resize_only(self, img, lbl, roi=None):
        img = TF.resize(img, (self.img_size, self.img_size))
        lbl = TF.resize(lbl, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)
        if roi is not None:
            roi = TF.resize(roi, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)
        return img, lbl, roi

    def _apply_augment(self, img, lbl, roi=None):
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
            )
            img = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size))
            lbl = TF.resized_crop(
                lbl, i, j, h, w, (self.img_size, self.img_size),
                interpolation=TF.InterpolationMode.NEAREST
            )
            if roi is not None:
                roi = TF.resized_crop(
                    roi, i, j, h, w, (self.img_size, self.img_size),
                    interpolation=TF.InterpolationMode.NEAREST
                )
        else:
            img, lbl, roi = self._apply_resize_only(img, lbl, roi)

        if random.random() > 0.5:
            img = TF.hflip(img)
            lbl = TF.hflip(lbl)
            if roi is not None:
                roi = TF.hflip(roi)

        if random.random() > 0.5:
            img = TF.vflip(img)
            lbl = TF.vflip(lbl)
            if roi is not None:
                roi = TF.vflip(roi)

        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img = TF.rotate(img, angle)
            lbl = TF.rotate(lbl, angle)
            if roi is not None:
                roi = TF.rotate(roi, angle)

        if self._should_use_color_jitter():
            if random.random() > 0.3:
                img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
                img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))

        return img, lbl, roi

    # ----------------------------------------------------
    # Tensor conversion
    # ----------------------------------------------------
    def _mask_to_tensor(self, mask_img: Image.Image) -> torch.Tensor:
        mask_np = np.array(mask_img, dtype=np.float32)
        mask_t = torch.from_numpy((mask_np > self.mask_threshold).astype(np.float32)).unsqueeze(0)
        return mask_t

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        lbl_path = self._get_label_path(img_path)
        if lbl_path is None:
            raise FileNotFoundError(f"Label not found for image: {img_path}")

        roi_path = None
        if self.return_roi:
            roi_path = self._get_roi_path(img_path)
            if roi_path is None:
                raise FileNotFoundError(f"ROI not found for image: {img_path}")

        img = self._read_rgb(img_path)
        lbl = self._read_mask(lbl_path)
        roi = self._read_mask(roi_path) if roi_path is not None else None

        if self.augment:
            img, lbl, roi = self._apply_augment(img, lbl, roi)
        else:
            img, lbl, roi = self._apply_resize_only(img, lbl, roi)

        img = TF.to_tensor(img)
        img = TF.normalize(img, self.mean, self.std)

        lbl_t = self._mask_to_tensor(lbl)

        if roi is not None:
            roi_t = self._mask_to_tensor(roi)
            return img, lbl_t, roi_t

        return img, lbl_t