#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PRISMUNet training entry.

Supported datasets:
- BUSI
- BUSI-WHU
- ISIC2017
- ISIC2018
- CHASEDB1
- FIVES
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from PRISM_UNet_Architecture import PRISMUNet
from dataset import MedicalDataset
import metrics as code_metrics

try:
    from scipy.ndimage import binary_erosion, distance_transform_edt
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


# =========================================================
# Paths / constants
# =========================================================
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DATASET_ROOT_DEFAULT = "DATASET"
RESULT_BASE = Path(__file__).resolve().parent / "result"
MODEL_NAME = "PRISM-UNet"
MODEL_SCALE_TO_WIDTH = {"S": 1.0, "M": 2.0, "L": 3.0}
MODEL_SCALE_TO_CHANNELS = {
    "S": [8, 16, 24, 32, 48, 64],
    "M": [16, 32, 48, 64, 96, 128],
    "L": [24, 48, 72, 96, 144, 192],
}

TOP_K = 20


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EMA:
    """Simple exponential moving average for evaluation."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = None

    def update(self, model: torch.nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: torch.nn.Module) -> None:
        self.backup = {}
        state = model.state_dict()
        for name, value in state.items():
            if name in self.shadow:
                self.backup[name] = value.detach().clone()
                value.copy_(self.shadow[name])

    def restore(self, model: torch.nn.Module) -> None:
        if not self.backup:
            return
        state = model.state_dict()
        for name, value in self.backup.items():
            if name in state:
                state[name].copy_(value)
        self.backup = None


def build_datasets(data_root: str) -> dict:
    root = Path(data_root)
    return {
        "BUSI": root / "Breast Ultrasound Dataset" / "BUSI",
        "BUSI-WHU": root / "Breast Ultrasound Dataset" / "BUSI-WHU",
        "ISIC2017": root / "Skin Lesion Dataset" / "ISIC 2017",
        "ISIC2018": root / "Skin Lesion Dataset" / "ISIC 2018",
        "CHASEDB1": root / "Retinal Vessel Dataset" / "CHASEDB1",
        "FIVES": root / "Retinal Vessel Dataset" / "FIVES",
    }


def resolve_data_root_path(data_root: str) -> Path:
    """Resolve dataset root using project-relative paths first."""
    p = Path(data_root)
    if p.is_absolute():
        return p

    candidates = [
        PROJECT_ROOT / p,  # preferred: relative to this project root
        Path.cwd() / p,    # fallback: relative to current working directory
        CODE_DIR / p,      # fallback: relative to PRISM directory
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def build_runtime_args(args, dataset_name):
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.strategy_name = "default"
    runtime_args.loss_mode = "bce_dice"
    runtime_args.reverse_aux = False
    runtime_args.use_early_stop = code_metrics.is_isic_dataset(dataset_name) and args.isic_early_stop_patience > 0
    runtime_args.early_stop_patience = args.isic_early_stop_patience if runtime_args.use_early_stop else 0

    if code_metrics.is_retinal_dataset(dataset_name):
        runtime_args.strategy_name = "retinal"
        runtime_args.loss_mode = "vessel_combo"
        runtime_args.aux_ds_weight = 1.0
        runtime_args.aux_ds_weight_end = 1.0
        runtime_args.aux_ds_decay_epochs = 0
        runtime_args.bce_weight = 0.5
        runtime_args.bce_weight_decay_epochs = 0
        runtime_args.use_adamw = False
        runtime_args.weight_decay = 1e-4
        runtime_args.scheduler = "cosine"
        # Keep CLI override ability: only switch to 512 when using the global default 256.
        if dataset_name == "FIVES" and args.img_size == 256:
            runtime_args.img_size = 512

    elif code_metrics.is_isic_dataset(dataset_name):
        runtime_args.strategy_name = "isic"
        runtime_args.loss_mode = "bce_dice"

    elif code_metrics.is_busi_dataset(dataset_name):
        runtime_args.strategy_name = "busi"
        runtime_args.loss_mode = "bce_dice"

    return runtime_args


def build_loader(root, dataset_name, split, args, shuffle=False, augment=False):
    use_roi = code_metrics.is_retinal_dataset(dataset_name) and split != "train"
    try:
        ds = MedicalDataset(
            str(root),
            dataset_name,
            split=split,
            img_size=args.img_size,
            augment=augment,
            return_roi=use_roi,
        )
    except (FileNotFoundError, ValueError) as e:
        if use_roi:
            print(f"[DataLoader] {dataset_name}/{split} ROI unavailable, fallback to return_roi=False: {e}")
            ds = MedicalDataset(
                str(root),
                dataset_name,
                split=split,
                img_size=args.img_size,
                augment=augment,
                return_roi=False,
            )
        else:
            raise
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=(split == "train"),
    )


def build_fives_external_chasedb1_loader(train_root, args):
    train_root = Path(train_root)
    if train_root.name != "FIVES":
        return None

    chasedb1_root = train_root.parent / "CHASEDB1"
    test_image_dir = chasedb1_root / "test" / "images"
    if not test_image_dir.exists():
        print(f"[FIVES External Eval] Skip CHASEDB1: test images not found -> {test_image_dir}")
        return None

    try:
        ds = MedicalDataset(
            str(chasedb1_root),
            "CHASEDB1",
            split="test",
            img_size=args.img_size,
            augment=False,
            return_roi=True,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"[FIVES External Eval] Skip CHASEDB1: {e}")
        return None

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


# =========================================================
# Losses
# =========================================================
def dice_loss(pred, target, smooth=1.0):
    pred_f = pred.contiguous().view(pred.size(0), -1)
    target_f = target.contiguous().view(target.size(0), -1)
    inter = (pred_f * target_f).sum(dim=1)
    dice = (2.0 * inter + smooth) / (pred_f.sum(dim=1) + target_f.sum(dim=1) + smooth)
    return (1.0 - dice).mean()


def tversky_loss(pred, target, alpha=0.7, beta=0.3, smooth=1e-7):
    pred_f = pred.contiguous().view(pred.size(0), -1)
    target_f = target.contiguous().view(target.size(0), -1)

    tp = (pred_f * target_f).sum(dim=1)
    fp = (pred_f * (1 - target_f)).sum(dim=1)
    fn = ((1 - pred_f) * target_f).sum(dim=1)

    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return (1.0 - tversky).mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * torch.pow(1 - pt, gamma)
    return (focal_weight * bce).mean()


def bce_dice_loss(logits, target, bce_weight=0.5):
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    dice = dice_loss(prob, target)
    return bce_weight * bce + dice


def focal_tversky_loss(logits, target, focal_weight=1.0, tversky_weight=1.0):
    prob = torch.sigmoid(logits)
    focal = focal_loss(prob, target)
    tversky = tversky_loss(prob, target)
    return focal_weight * focal + tversky_weight * tversky


def soft_erode(img, kernel_size=3):
    pad = kernel_size // 2
    return -F.max_pool2d(-img, kernel_size, stride=1, padding=pad)


def soft_dilate(img, kernel_size=3):
    pad = kernel_size // 2
    return F.max_pool2d(img, kernel_size, stride=1, padding=pad)


def soft_skeleton(img, iters=10):
    skel = F.relu(img - soft_dilate(soft_erode(img)))
    for _ in range(iters - 1):
        img = soft_erode(img)
        delta = F.relu(img - soft_dilate(soft_erode(img)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(pred, target, iters=10, smooth=1.0):
    skel_pred = soft_skeleton(pred, iters)
    skel_target = soft_skeleton(target, iters)
    tprec = ((skel_pred * target).sum(dim=(2, 3)) + smooth) / (skel_pred.sum(dim=(2, 3)) + smooth)
    tsens = ((skel_target * pred).sum(dim=(2, 3)) + smooth) / (skel_target.sum(dim=(2, 3)) + smooth)
    cl_dice = (2.0 * tprec * tsens + smooth) / (tprec + tsens + smooth)
    return (1.0 - cl_dice).mean()


def vessel_combo_loss(logits, target, bce_weight=0.5, cldice_weight=0.3):
    prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    dice = dice_loss(prob, target)
    cldice = soft_cldice_loss(prob, target, iters=10)
    return bce_weight * bce + dice + cldice_weight * cldice


def compute_single_loss(logits, target, loss_mode="bce_dice", bce_weight=0.5):
    if loss_mode == "focal_tversky":
        return focal_tversky_loss(logits, target)
    if loss_mode == "vessel_combo":
        return vessel_combo_loss(logits, target, bce_weight=bce_weight)
    return bce_dice_loss(logits, target, bce_weight=bce_weight)


def unpack_output(model_out):
    if isinstance(model_out, tuple) and len(model_out) == 2 and isinstance(model_out[0], (tuple, list)):
        aux_preds, main_pred = model_out
        return main_pred, list(aux_preds)
    return model_out, []


def total_seg_loss(main_pred, aux_preds, target, aux_weight=0.2, bce_weight=0.5, loss_mode="bce_dice", reverse_aux=False):
    main_loss = compute_single_loss(main_pred, target, loss_mode=loss_mode, bce_weight=bce_weight)
    if not aux_preds:
        return main_loss

    n_aux = len(aux_preds)
    if n_aux == 5:
        aux_weights = [0.30, 0.25, 0.20, 0.15, 0.10] if reverse_aux else [0.10, 0.15, 0.20, 0.25, 0.30]
    elif n_aux == 2:
        aux_weights = [0.6, 0.4]
    else:
        aux_weights = [(n_aux - i) / sum(range(1, n_aux + 1)) for i in range(n_aux)]

    aux_losses = []
    for w, p in zip(aux_weights, aux_preds):
        aux_losses.append(w * compute_single_loss(p, target, loss_mode=loss_mode, bce_weight=bce_weight))

    return main_loss + aux_weight * sum(aux_losses)


def get_aux_weight(args, epoch: int) -> float:
    if args.aux_ds_decay_epochs <= 0:
        return args.aux_ds_weight
    end_w = args.aux_ds_weight_end
    if end_w is None:
        end_w = max(0.0, args.aux_ds_weight * 0.25)
    progress = min(1.0, max(0.0, (epoch - 1) / float(args.aux_ds_decay_epochs)))
    return args.aux_ds_weight + (end_w - args.aux_ds_weight) * progress


def get_bce_weight(args, epoch: int) -> float:
    if args.bce_weight_decay_epochs <= 0:
        return args.bce_weight
    end_w = args.bce_weight_end
    if end_w is None:
        end_w = max(0.1, args.bce_weight * 0.5)
    progress = min(1.0, max(0.0, (epoch - 1) / float(args.bce_weight_decay_epochs)))
    return args.bce_weight + (end_w - args.bce_weight) * progress


# =========================================================
# Metrics / eval
# =========================================================
def _unpack_eval_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            return batch[0], batch[1], batch[2]
        if len(batch) == 2:
            return batch[0], batch[1], None
    raise ValueError(f"Unexpected evaluation batch format: {type(batch)}")


def _unpack_train_batch(batch):
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError(f"Unexpected training batch format: {type(batch)}")


def batch_metrics(pred, target, thresh=0.5):
    pred_prob = torch.sigmoid(pred.detach())
    pred_bin = (pred_prob >= thresh).float()
    p = pred_bin.view(pred_bin.size(0), -1)
    t = target.view(target.size(0), -1)
    inter = (p * t).sum(dim=1)
    union = (p + t - p * t).sum(dim=1)
    dice = ((2 * inter + 1e-7) / (p.sum(dim=1) + t.sum(dim=1) + 1e-7)).mean().item()
    iou = ((inter + 1e-7) / (union + 1e-7)).mean().item()
    return dice, iou


@torch.no_grad()
def _tta_forward(model, imgs):
    def _forward_prob(x):
        outputs = model(x)
        main_pred, _ = unpack_output(outputs)
        return torch.sigmoid(main_pred)

    probs = _forward_prob(imgs)
    probs = probs + torch.flip(_forward_prob(torch.flip(imgs, dims=[-1])), dims=[-1])
    probs = probs + torch.flip(_forward_prob(torch.flip(imgs, dims=[-2])), dims=[-2])
    probs = probs + torch.flip(_forward_prob(torch.flip(imgs, dims=[-2, -1])), dims=[-2, -1])
    return probs / 4.0


def hd95_score(pred_bin: np.ndarray, mask_bin: np.ndarray) -> float:
    if not SCIPY_AVAILABLE:
        return float("nan")

    pred_bin = pred_bin.astype(bool)
    mask_bin = mask_bin.astype(bool)
    if pred_bin.sum() == 0 and mask_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or mask_bin.sum() == 0:
        return float("nan")

    pred_surface = pred_bin ^ binary_erosion(pred_bin)
    mask_surface = mask_bin ^ binary_erosion(mask_bin)
    if pred_surface.sum() == 0 or mask_surface.sum() == 0:
        return float("nan")

    dt_pred = distance_transform_edt(~pred_surface)
    dt_mask = distance_transform_edt(~mask_surface)
    d1 = dt_mask[pred_surface]
    d2 = dt_pred[mask_surface]
    all_d = np.concatenate([d1, d2], axis=0)
    return float(np.percentile(all_d, 95))


@torch.no_grad()
def _collect_full_metric_rows(
    model,
    loader,
    device,
    dataset_name,
    threshold=0.5,
    aux_ds_weight=0.4,
    route_aux_weight=0.0,
    bce_weight=0.5,
    loss_mode="bce_dice",
    reverse_aux=False,
):
    model.eval()
    metric_fn = code_metrics.pick_metric_fn(dataset_name)
    rows = []
    total_loss, n = 0.0, 0

    for batch in loader:
        imgs, masks, roi = _unpack_eval_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if roi is not None:
            roi = roi.to(device, non_blocking=True)

        outputs = model(imgs)
        main_pred, aux_preds = unpack_output(outputs)
        loss = total_seg_loss(
            main_pred, aux_preds, masks,
            aux_weight=aux_ds_weight,
            bce_weight=bce_weight,
            loss_mode=loss_mode,
            reverse_aux=reverse_aux,
        )

        route_aux = getattr(model, "aux_loss_total", None)
        if route_aux is not None and route_aux_weight > 0:
            loss = loss + route_aux_weight * route_aux

        probs = torch.sigmoid(main_pred)
        if code_metrics.is_retinal_dataset(dataset_name) and roi is not None:
            metric_dict = metric_fn(probs, masks, thr=threshold, reduce="none", roi=roi, pred_type="prob")
        else:
            metric_dict = metric_fn(probs, masks, thr=threshold, reduce="none", pred_type="prob")

        for i in range(probs.shape[0]):
            rows.append({k: float(metric_dict[k][i]) for k in metric_dict})

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

    return (total_loss / n) if n else np.nan, rows


def _summarize_metric_rows(rows, dataset_name):
    metric_columns = code_metrics.get_metric_columns(dataset_name)
    if not rows:
        return {k: np.nan for k in metric_columns}
    return {k: float(np.nanmean([row[k] for row in rows])) for k in metric_columns if k in rows[0]}


@torch.no_grad()
def val_quick(model, loader, device, dataset_name, aux_ds_weight=0.4, route_aux_weight=0.0, bce_weight=0.5, loss_mode="bce_dice", reverse_aux=False):
    total_loss, rows = _collect_full_metric_rows(
        model, loader, device, dataset_name, threshold=0.5,
        aux_ds_weight=aux_ds_weight,
        route_aux_weight=route_aux_weight,
        bce_weight=bce_weight,
        loss_mode=loss_mode,
        reverse_aux=reverse_aux,
    )
    summary = _summarize_metric_rows(rows, dataset_name)
    return total_loss, summary.get("Dice", np.nan), summary.get("IoU", np.nan)


@torch.no_grad()
def val_quick_search_threshold(
    model,
    loader,
    device,
    dataset_name,
    thresholds,
    aux_ds_weight=0.4,
    route_aux_weight=0.0,
    bce_weight=0.5,
    loss_mode="bce_dice",
    reverse_aux=False,
):
    best_loss = np.nan
    best_dice = -np.inf
    best_iou = np.nan
    best_thr = 0.5

    for thr in thresholds:
        total_loss, rows = _collect_full_metric_rows(
            model, loader, device, dataset_name, threshold=float(thr),
            aux_ds_weight=aux_ds_weight,
            route_aux_weight=route_aux_weight,
            bce_weight=bce_weight,
            loss_mode=loss_mode,
            reverse_aux=reverse_aux,
        )
        summary = _summarize_metric_rows(rows, dataset_name)
        dice = summary.get("Dice", np.nan)
        if np.isnan(dice):
            continue
        if dice > best_dice:
            best_loss = total_loss
            best_dice = dice
            best_iou = summary.get("IoU", np.nan)
            best_thr = float(thr)

    if not np.isfinite(best_dice):
        return np.nan, np.nan, np.nan, 0.5
    return best_loss, float(best_dice), float(best_iou), best_thr


@torch.no_grad()
def val_full(model, loader, device, dataset_name, threshold: float = 0.5, use_tta: bool = False):
    if use_tta:
        return _val_full_tta(model, loader, device, dataset_name, threshold)
    total_loss, rows = _collect_full_metric_rows(
        model, loader, device, dataset_name, threshold=threshold,
        aux_ds_weight=0.0,
        route_aux_weight=0.0,
        bce_weight=0.5,
        loss_mode="bce_dice",
    )
    del total_loss
    return _summarize_metric_rows(rows, dataset_name)


@torch.no_grad()
def _val_full_tta(model, loader, device, dataset_name, threshold: float = 0.5):
    model.eval()
    metric_fn = code_metrics.pick_metric_fn(dataset_name)
    rows = []

    for batch in loader:
        imgs, masks, roi = _unpack_eval_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        if roi is not None:
            roi = roi.to(device, non_blocking=True)

        probs = _tta_forward(model, imgs)
        if code_metrics.is_retinal_dataset(dataset_name) and roi is not None:
            metric_dict = metric_fn(probs, masks, thr=threshold, reduce="none", roi=roi, pred_type="prob")
        else:
            metric_dict = metric_fn(probs, masks, thr=threshold, reduce="none", pred_type="prob")

        for i in range(probs.shape[0]):
            rows.append({k: float(metric_dict[k][i]) for k in metric_dict})

    return _summarize_metric_rows(rows, dataset_name)


# =========================================================
# Model
# =========================================================
def resolve_model_config(model_scale=None, width_multiplier=None):
    if model_scale is not None:
        scale = str(model_scale).upper()
        if scale not in MODEL_SCALE_TO_WIDTH:
            raise ValueError(f"Unsupported model_scale: {model_scale}. Choose from {list(MODEL_SCALE_TO_WIDTH.keys())}")
        return scale, MODEL_SCALE_TO_WIDTH[scale], list(MODEL_SCALE_TO_CHANNELS[scale])

    if width_multiplier is None:
        width_multiplier = 1.0

    width_multiplier = float(width_multiplier)
    if abs(width_multiplier - 1.0) < 1e-8:
        scale = "S"
    elif abs(width_multiplier - 2.0) < 1e-8:
        scale = "M"
    elif abs(width_multiplier - 3.0) < 1e-8:
        scale = "L"
    else:
        raise ValueError("width_multiplier only supports 1.0, 2.0, or 3.0. Use --model_scale S/M/L or one of these multipliers.")

    return scale, width_multiplier, list(MODEL_SCALE_TO_CHANNELS[scale])


def build_model(use_gt_ds=True, model_scale=None, width_multiplier=None, aux_alpha=1.0):
    scale, resolved_width, c_list = resolve_model_config(model_scale=model_scale, width_multiplier=width_multiplier)
    model = PRISMUNet(
        num_classes=1,
        input_channels=3,
        c_list=c_list,
        bridge=True,
        gt_ds=use_gt_ds,
        aux_alpha=aux_alpha,
    )
    model.model_scale = scale
    model.width_multiplier = resolved_width
    model.channel_list = c_list
    return model


# =========================================================
# Train
# =========================================================
def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    aux_ds_weight=0.2,
    route_aux_weight=0.0,
    bce_weight=0.5,
    debug_loss=False,
    scheduler=None,
    scheduler_per_batch=False,
    ema=None,
    loss_mode="bce_dice",
    reverse_aux=False,
):
    model.train()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch [{epoch}/{total_epochs}]", leave=False)

    for batch_idx, batch in pbar:
        imgs, masks = _unpack_train_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(imgs)
        main_pred, aux_preds = unpack_output(outputs)
        loss = total_seg_loss(
            main_pred, aux_preds, masks,
            aux_weight=aux_ds_weight,
            bce_weight=bce_weight,
            loss_mode=loss_mode,
            reverse_aux=reverse_aux,
        )

        if debug_loss and batch_idx == 0 and epoch == 1:
            with torch.no_grad():
                main_loss = compute_single_loss(main_pred, masks, loss_mode=loss_mode, bce_weight=bce_weight)
                print(f"\n[Debug] Loss breakdown (Epoch 1, Batch 1):")
                print(f"  Loss mode: {loss_mode}, reverse_aux: {reverse_aux}")
                print(f"  Main loss: {main_loss.item():.4f}")
                if aux_preds:
                    aux_names = (
                        ['decoder1(H/32)[w=0.30]', 'decoder2(H/16)[w=0.25]', 'decoder3(H/8)[w=0.20]', 'decoder4(H/4)[w=0.15]', 'decoder5(H/2)[w=0.10]']
                        if reverse_aux else
                        ['decoder1(H/32)[w=0.10]', 'decoder2(H/16)[w=0.15]', 'decoder3(H/8)[w=0.20]', 'decoder4(H/4)[w=0.25]', 'decoder5(H/2)[w=0.30]']
                    )
                    for i, aux_p in enumerate(aux_preds):
                        aux_l = compute_single_loss(aux_p, masks, loss_mode=loss_mode, bce_weight=bce_weight)
                        aux_name = aux_names[i] if i < len(aux_names) else f"aux{i+1}"
                        print(f"  Aux loss {i+1} ({aux_name}): {aux_l.item():.4f}")
                    print(f"  Total aux weight: {aux_ds_weight:.3f}")
                print(f"  Total loss: {loss.item():.4f}\n")

        route_aux = getattr(model, "aux_loss_total", None)
        if route_aux is not None and route_aux_weight > 0:
            loss = loss + route_aux_weight * route_aux

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if ema is not None:
            ema.update(model)
        if scheduler is not None and scheduler_per_batch:
            scheduler.step()

        d, iou = batch_metrics(main_pred, masks)
        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_dice += d * bs
        total_iou += iou * bs
        n += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    return total_loss / n, total_dice / n, total_iou / n


def train_dataset(dataset_name, root, device, args):
    runtime_args = build_runtime_args(args, dataset_name)
    resolved_scale, resolved_width, resolved_channels = resolve_model_config(
        model_scale=runtime_args.model_scale,
        width_multiplier=runtime_args.width_multiplier,
    )
    model_name = MODEL_NAME

    print(f"\nStart Training {MODEL_NAME}: [{dataset_name}]")
    print(
        f"[{runtime_args.strategy_name}] "
        f"lr={runtime_args.lr}, wd={runtime_args.weight_decay}, "
        f"scheduler={runtime_args.scheduler}, loss={runtime_args.loss_mode}, "
        f"img_size={runtime_args.img_size}, model_scale={resolved_scale}, width_multiplier={resolved_width}, channels={resolved_channels}"
    )
    if runtime_args.use_early_stop:
        print(f"[Early Stopping] Enabled for {dataset_name}, patience={runtime_args.early_stop_patience}.")

    train_loader = build_loader(root, dataset_name, "train", runtime_args, shuffle=True, augment=True)
    val_loader = build_loader(root, dataset_name, "val", runtime_args, shuffle=False, augment=False)

    test_root = root / "test" / "images"
    test_loader = None
    if test_root.exists():
        test_loader = build_loader(root, dataset_name, "test", runtime_args, shuffle=False, augment=False)

    model = build_model(
        use_gt_ds=runtime_args.use_gt_ds,
        model_scale=resolved_scale,
        width_multiplier=resolved_width,
        aux_alpha=runtime_args.aux_alpha,
    ).to(device)

    ema = EMA(model, decay=runtime_args.ema_decay) if runtime_args.use_ema else None

    if runtime_args.dry_run:
        print("\n[DRY RUN] One-batch data/model/loss sanity check...")
        model.eval()
        with torch.no_grad():
            for batch in train_loader:
                imgs, masks = _unpack_train_batch(batch)
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                outputs = model(imgs)
                main_pred, aux_preds = unpack_output(outputs)
                pred_prob = torch.sigmoid(main_pred)
                loss = total_seg_loss(
                    main_pred, aux_preds, masks,
                    aux_weight=runtime_args.aux_ds_weight,
                    bce_weight=runtime_args.bce_weight,
                    loss_mode=runtime_args.loss_mode,
                    reverse_aux=runtime_args.reverse_aux,
                )
                d, iou = batch_metrics(main_pred, masks)
                print(f"  Input range : [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
                print(f"  Target range: [{masks.min().item():.3f}, {masks.max().item():.3f}]")
                print(f"  Logits range: [{main_pred.min().item():.3f}, {main_pred.max().item():.3f}]")
                print(f"  Prob range  : [{pred_prob.min().item():.3f}, {pred_prob.max().item():.3f}]")
                print(f"  Loss={loss.item():.4f}, Dice={d:.4f}, IoU={iou:.4f}")
                break
        print("[DRY RUN] Done.")
        return

    if getattr(runtime_args, "use_adamw", False):
        optimizer = torch.optim.AdamW(model.parameters(), lr=runtime_args.lr, weight_decay=runtime_args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=runtime_args.lr, weight_decay=runtime_args.weight_decay)

    scheduler = None
    scheduler_mode = "none"
    if runtime_args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, verbose=True, min_lr=runtime_args.min_lr
        )
        scheduler_mode = "plateau"
    elif runtime_args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=runtime_args.epochs, eta_min=runtime_args.min_lr
        )
        scheduler_mode = "epoch"
    elif runtime_args.scheduler == "onecycle":
        max_lr = runtime_args.max_lr if runtime_args.max_lr is not None else runtime_args.lr * 2.0
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=runtime_args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=runtime_args.onecycle_pct_start,
            anneal_strategy="cos",
        )
        scheduler_mode = "batch"

    final_dir = RESULT_BASE / f"{model_name}_{dataset_name}"
    tmp_dir = final_dir / "_tmp"
    final_dir.mkdir(parents=True, exist_ok=True)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0
    best_epoch = -1
    best_thr = 0.5
    best_ckpt = final_dir / "best.pth"
    train_log = []
    epoch_candidates = []
    bad_epochs_count = 0

    thresholds = None
    if runtime_args.val_thresh_search:
        thresholds = np.arange(runtime_args.val_thresh_min, runtime_args.val_thresh_max + 1e-9, runtime_args.val_thresh_step).tolist()
        if 0.5 not in thresholds:
            thresholds.append(0.5)
        thresholds = sorted(set([round(t, 4) for t in thresholds]))

    for epoch in range(1, runtime_args.epochs + 1):
        cur_aux_weight = get_aux_weight(runtime_args, epoch)
        cur_bce_weight = get_bce_weight(runtime_args, epoch)

        tr_loss, tr_dice, tr_iou = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            runtime_args.epochs,
            aux_ds_weight=cur_aux_weight,
            route_aux_weight=runtime_args.route_aux_weight,
            bce_weight=cur_bce_weight,
            debug_loss=(runtime_args.use_gt_ds and epoch == 1),
            scheduler=scheduler,
            scheduler_per_batch=(scheduler_mode == "batch"),
            ema=ema,
            loss_mode=runtime_args.loss_mode,
            reverse_aux=runtime_args.reverse_aux,
        )

        if ema is not None:
            ema.apply_to(model)

        do_thresh_search = runtime_args.val_thresh_search and epoch > int(runtime_args.epochs * 0.7)
        if do_thresh_search:
            vl_loss, vl_dice, vl_iou, thr_now = val_quick_search_threshold(
                model,
                val_loader,
                device,
                dataset_name=dataset_name,
                thresholds=thresholds,
                aux_ds_weight=cur_aux_weight,
                route_aux_weight=runtime_args.route_aux_weight,
                bce_weight=cur_bce_weight,
                loss_mode=runtime_args.loss_mode,
                reverse_aux=runtime_args.reverse_aux,
            )
        else:
            vl_loss, vl_dice, vl_iou = val_quick(
                model,
                val_loader,
                device,
                dataset_name=dataset_name,
                aux_ds_weight=cur_aux_weight,
                route_aux_weight=runtime_args.route_aux_weight,
                bce_weight=cur_bce_weight,
                loss_mode=runtime_args.loss_mode,
                reverse_aux=runtime_args.reverse_aux,
            )
            thr_now = 0.5

        if scheduler_mode == "plateau":
            scheduler.step(vl_dice)
        elif scheduler_mode == "epoch":
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch:>3}/{runtime_args.epochs}] | "
            f"Aux_W: {cur_aux_weight:.3f} | BCE_W: {cur_bce_weight:.2f} | "
            f"LR: {lr_now:.2e} | "
            f"Tr_L: {tr_loss:.4f} Tr_D: {tr_dice:.4f} Tr_I: {tr_iou:.4f} | "
            f"Vl_L: {vl_loss:.4f} Vl_D: {vl_dice:.4f} Vl_I: {vl_iou:.4f} | Thr: {thr_now:.2f}"
        )

        weight_name = f"epoch_{epoch:03d}_dice_{vl_dice:.4f}.pth"
        ckpt_path = tmp_dir / weight_name
        torch.save(model.state_dict(), ckpt_path)

        log_row = {
            "Epoch": epoch,
            "Weight": weight_name,
            "Train_Loss": tr_loss,
            "Train_Dice": tr_dice,
            "Train_IoU": tr_iou,
            "Val_Loss": vl_loss,
            "Val_Dice": vl_dice,
            "Val_IoU": vl_iou,
            "Val_Thresh": thr_now,
        }
        train_log.append(log_row)
        epoch_candidates.append(
            {
                "epoch": epoch,
                "weight": weight_name,
                "dice": vl_dice,
                "thr": thr_now,
                "path": ckpt_path,
            }
        )

        if vl_dice > best_dice:
            best_dice = vl_dice
            best_epoch = epoch
            best_thr = thr_now
            bad_epochs_count = 0
        elif runtime_args.use_early_stop:
            bad_epochs_count += 1
            if bad_epochs_count >= runtime_args.early_stop_patience:
                print(f"[{dataset_name}] Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val dice: {best_dice:.4f}).")
                if ema is not None:
                    ema.restore(model)
                break

        if ema is not None:
            ema.restore(model)

    top_checkpoints = sorted(epoch_candidates, key=lambda x: x["dice"], reverse=True)[: min(TOP_K, len(epoch_candidates))]
    final_checkpoints = []

    for item in top_checkpoints:
        final_path = final_dir / item["weight"]
        shutil.copy2(item["path"], final_path)
        final_checkpoints.append({"epoch": item["epoch"], "final_path": final_path, "thr": item["thr"]})

    if final_checkpoints:
        best_item = top_checkpoints[0]
        shutil.copy2(best_item["path"], best_ckpt)
        best_dice = best_item["dice"]
        best_epoch = best_item["epoch"]
        best_thr = best_item["thr"]

    shutil.rmtree(tmp_dir, ignore_errors=True)
    pd.DataFrame(train_log).to_excel(final_dir / f"{MODEL_NAME}_{dataset_name}_train_log.xlsx", index=False)

    use_tta = code_metrics.is_retinal_dataset(dataset_name)
    test_rows = []
    if final_checkpoints and test_loader is not None:
        tta_str = " [TTA]" if use_tta else ""
        print(f"\nEvaluating Top {len(final_checkpoints)} checkpoints on Test Set{tta_str}...")
        for item in sorted(final_checkpoints, key=lambda x: x["epoch"]):
            model.load_state_dict(torch.load(item["final_path"], map_location=device))
            m = val_full(model, test_loader, device, dataset_name, threshold=item["thr"], use_tta=use_tta)
            metric_columns = code_metrics.get_metric_columns(dataset_name)
            test_rows.append(
                {
                    "Epoch": item["epoch"],
                    "Weight": item["final_path"].name,
                    **{k: round(m[k], 4) for k in metric_columns if k in m},
                }
            )
            print(f"[Test Ep {item['epoch']:>3}] {code_metrics.format_metric_summary(dataset_name, m)}")
    elif test_loader is None:
        print("No test split found. Still exporting empty test_results.xlsx.")

    pd.DataFrame(test_rows).to_excel(final_dir / f"{MODEL_NAME}_{dataset_name}_test_results.xlsx", index=False)

    if dataset_name == "FIVES":
        external_loader = build_fives_external_chasedb1_loader(root, runtime_args)
        if external_loader is not None and final_checkpoints:
            print(f"\nEvaluating Top {len(final_checkpoints)} checkpoints on External CHASEDB1 [TTA]...")
            external_rows = []
            for item in sorted(final_checkpoints, key=lambda x: x["epoch"]):
                model.load_state_dict(torch.load(item["final_path"], map_location=device))
                m = val_full(model, external_loader, device, "CHASEDB1", threshold=item["thr"], use_tta=True)
                metric_columns = code_metrics.get_metric_columns("CHASEDB1")
                external_rows.append(
                    {
                        "Epoch": item["epoch"],
                        "Weight": item["final_path"].name,
                        **{k: round(m[k], 4) for k in metric_columns if k in m},
                    }
                )
                print(f"[Ext CHASEDB1 Ep {item['epoch']:>3}] {code_metrics.format_metric_summary('CHASEDB1', m)}")
            pd.DataFrame(external_rows).to_excel(
                final_dir / f"{MODEL_NAME}_FIVES_to_CHASEDB1_test_results.xlsx",
                index=False,
            )

    print(f"[{dataset_name}] {model_name} Training & Evaluation Complete.")


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=None)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default=DATASET_ROOT_DEFAULT)

    parser.add_argument("--use_gt_ds", dest="use_gt_ds", action="store_true")
    parser.add_argument("--no_gt_ds", dest="use_gt_ds", action="store_false")
    parser.set_defaults(use_gt_ds=True)

    parser.add_argument("--aux_alpha", type=float, default=1.0)
    parser.add_argument("--aux_ds_weight", type=float, default=0.2)
    parser.add_argument("--aux_ds_weight_end", type=float, default=0.1)
    parser.add_argument("--aux_ds_decay_epochs", type=int, default=80)

    parser.add_argument("--route_aux_weight", type=float, default=0.02)
    parser.add_argument("--model_scale", type=str, default=None, choices=["S", "M", "L", "s", "m", "l"])
    parser.add_argument("--width_multiplier", type=float, default=None)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--bce_weight_end", type=float, default=None)
    parser.add_argument("--bce_weight_decay_epochs", type=int, default=0)

    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        choices=["plateau", "cosine", "onecycle"],
    )
    parser.add_argument("--onecycle_pct_start", type=float, default=0.2)

    parser.add_argument("--use_ema", dest="use_ema", action="store_true")
    parser.add_argument("--no_use_ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--val_thresh_search", dest="val_thresh_search", action="store_true")
    parser.add_argument("--no_val_thresh_search", dest="val_thresh_search", action="store_false")
    parser.set_defaults(val_thresh_search=True)
    parser.add_argument("--val_thresh_min", type=float, default=0.3)
    parser.add_argument("--val_thresh_max", type=float, default=0.7)
    parser.add_argument("--val_thresh_step", type=float, default=0.05)

    parser.add_argument("--isic_early_stop_patience", type=int, default=20)

    args = parser.parse_args()
    if args.model_scale is not None:
        args.model_scale = args.model_scale.upper()
    args.data_root = str(resolve_data_root_path(args.data_root))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = build_datasets(args.data_root)
    to_train = {d: datasets[d] for d in args.dataset if d in datasets} if args.dataset else datasets

    for ds_name, ds_root in to_train.items():
        if not (ds_root / "train").exists():
            print(f"Skip [{ds_name}]: train split not found -> {ds_root / 'train'}")
            continue
        train_dataset(ds_name, ds_root, device, args)


if __name__ == "__main__":
    main()