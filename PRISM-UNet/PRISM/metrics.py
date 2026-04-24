import numpy as np
import torch
from medpy import metric

BUSI_DATASETS = {"BUSI", "BUSI-WHU"}
ISIC_DATASETS = {"ISIC2017", "ISIC2018"}
RETINAL_DATASETS = {"CHASEDB1", "FIVES"}


# =========================================================
# Basic Utilities
# =========================================================
def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array-like input to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)

    pos = x >= 0
    neg = ~pos

    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _ensure_bhw(arr, name: str) -> np.ndarray:
    """
    Ensure input shape is (B, H, W).
    Supported:
        (B, 1, H, W)
        (B, H, W)
        (H, W)
    """
    arr = _to_numpy(arr)

    if arr.ndim == 4:
        if arr.shape[1] != 1:
            raise ValueError(
                f"{name} has shape {arr.shape}. Only single-channel (B,1,H,W) supported."
            )
        arr = arr[:, 0, :, :]
    elif arr.ndim == 3:
        pass
    elif arr.ndim == 2:
        arr = arr[None, ...]
    else:
        raise ValueError(f"{name} has unsupported ndim={arr.ndim}, shape={arr.shape}")

    return arr


def _prepare_pred_gt(pred, gt, pred_type="auto", gt_thr=0.5):
    """
    Prepare prediction and ground truth for binary segmentation evaluation.

    Args:
        pred: logits or probabilities
        gt: binary mask or soft mask
        pred_type: "auto", "logits", or "prob"
        gt_thr: threshold for binarizing gt
    """
    pred = _ensure_bhw(pred, "pred").astype(np.float32, copy=False)
    gt = _ensure_bhw(gt, "gt").astype(np.float32, copy=False)

    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")

    if pred_type not in ("auto", "logits", "prob"):
        raise ValueError("pred_type must be one of: 'auto', 'logits', 'prob'")

    if pred_type == "logits":
        pred = _sigmoid_np(pred)
    elif pred_type == "prob":
        pred = np.clip(pred, 0.0, 1.0)
    else:
        # auto mode only for convenience, not recommended for final reported experiments
        if pred.min() < 0.0 or pred.max() > 1.0:
            pred = _sigmoid_np(pred)
        else:
            pred = np.clip(pred, 0.0, 1.0)

    gt_bin = (gt >= gt_thr).astype(np.uint8)
    pred = pred.astype(np.float32, copy=False)
    return pred, gt_bin


def _prepare_roi(roi, target_shape):
    """Prepare ROI mask into binary (B,H,W)."""
    if roi is None:
        return None

    roi = _ensure_bhw(roi, "roi").astype(np.float32, copy=False)

    if tuple(roi.shape) != tuple(target_shape):
        raise ValueError(f"ROI shape {roi.shape} does not match target shape {target_shape}")

    roi_bin = (roi >= 0.5).astype(np.uint8)
    return roi_bin


def _safe_div(a, b, eps=1e-7):
    return float(a / (b + eps))


def _confusion_from_bin(pred_bin: np.ndarray, gt_bin: np.ndarray):
    """Return TP, TN, FP, FN."""
    p = pred_bin.astype(bool, copy=False)
    g = gt_bin.astype(bool, copy=False)

    tp = np.logical_and(p, g).sum()
    tn = np.logical_and(~p, ~g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    return tp, tn, fp, fn


def _safe_hd95(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing=(1.0, 1.0), empty_value=np.nan):
    """
    Safe HD95:
    - both empty -> 0
    - one empty -> empty_value
    - otherwise compute medpy hd95
    """
    pos_p = int(pred_bin.sum())
    pos_g = int(gt_bin.sum())

    if pos_p == 0 and pos_g == 0:
        return 0.0
    if pos_p == 0 or pos_g == 0:
        return float(empty_value)

    return float(
        metric.binary.hd95(
            pred_bin.astype(bool),
            gt_bin.astype(bool),
            voxelspacing=spacing
        )
    )


# =========================================================
# Metric Computation
# =========================================================
def _binary_metric_dict(tp, tn, fp, fn):
    """Compute common binary segmentation metrics from TP/TN/FP/FN."""
    dice = _safe_div(2 * tp, 2 * tp + fp + fn)
    iou = _safe_div(tp, tp + fp + fn)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)

    mcc_denom = np.sqrt(
        float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    ) + 1e-7
    mcc = float((float(tp) * float(tn) - float(fp) * float(fn)) / mcc_denom)

    # For binary segmentation, F1 is numerically identical to Dice
    f1 = dice

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "Accuracy": float(accuracy),
        "MCC": float(mcc),
        "F1": float(f1),
    }


def _handle_empty_case(empty_policy, pred_bin, gt_bin):
    """
    Handle empty-mask cases.

    empty_policy:
        - "perfect": if both empty, related overlap metrics = 1
        - "nan": if both empty, related metrics = nan
    """
    pos_p = int(pred_bin.sum())
    pos_g = int(gt_bin.sum())

    both_empty = (pos_p == 0 and pos_g == 0)
    one_empty = (pos_p == 0) ^ (pos_g == 0)

    return both_empty, one_empty


def _reduce_dict(list_of_dicts, reduce="mean"):
    """
    Reduce per-sample metric dicts.

    Supported:
        - "none": return list for each metric
        - "mean": return mean
        - "std": return std
        - "mean_std": return both mean and std
    """
    if not list_of_dicts:
        return {}

    keys = list(list_of_dicts[0].keys())

    if reduce == "none":
        return {k: [d[k] for d in list_of_dicts] for k in keys}

    out = {}
    for k in keys:
        vals = np.asarray([d[k] for d in list_of_dicts], dtype=np.float64)

        if reduce == "mean":
            out[k] = float(np.nanmean(vals))
        elif reduce == "std":
            out[k] = float(np.nanstd(vals))
        elif reduce == "mean_std":
            out[f"{k}_mean"] = float(np.nanmean(vals))
            out[f"{k}_std"] = float(np.nanstd(vals))
        else:
            raise ValueError("reduce must be one of: 'none', 'mean', 'std', 'mean_std'")

    return out


# =========================================================
# Generic Binary Segmentation Evaluator
# =========================================================
def calc_binary_seg_metrics(
    pred,
    gt,
    metrics=None,
    thr=0.5,
    pred_type="auto",
    gt_thr=0.5,
    reduce="mean",
    roi=None,
    spacing=(1.0, 1.0),
    include_hd95=False,
    empty_hd95_value=np.nan,
    empty_policy="perfect",
):
    """
    Generic evaluator for binary segmentation.

    Args:
        pred: prediction tensor/array, logits or probabilities
        gt: ground truth tensor/array
        metrics: list/tuple of metric names to keep
        thr: threshold for binarizing predictions
        pred_type: "auto", "logits", "prob"
        gt_thr: threshold for binarizing ground truth
        reduce: "none", "mean", "std", "mean_std"
        roi: optional ROI mask, same shape as gt
        spacing: voxel spacing for HD95
        include_hd95: whether to compute HD95
        empty_hd95_value: HD95 value when one mask is empty
        empty_policy: "perfect" or "nan"
    """
    pred_prob, gt_bin = _prepare_pred_gt(pred, gt, pred_type=pred_type, gt_thr=gt_thr)
    roi_bin = _prepare_roi(roi, gt_bin.shape)

    default_metrics = ["Dice", "IoU", "Sensitivity", "Specificity", "Precision", "Accuracy", "MCC", "F1"]
    if include_hd95:
        default_metrics.append("HD95")

    if metrics is None:
        metrics = default_metrics
    metrics = list(metrics)

    results = []

    for idx, (p, g) in enumerate(zip(pred_prob, gt_bin)):
        if roi_bin is not None:
            valid = roi_bin[idx].astype(bool)
            if valid.sum() == 0:
                raise ValueError(f"ROI for sample index {idx} is empty.")
            p = p[valid]
            g = g[valid]

        pb = (p >= thr).astype(np.uint8)

        both_empty, one_empty = _handle_empty_case(empty_policy, pb, g)

        if both_empty:
            if empty_policy == "perfect":
                metric_dict = {
                    "Dice": 1.0,
                    "IoU": 1.0,
                    "Sensitivity": 1.0,
                    "Specificity": 1.0,
                    "Precision": 1.0,
                    "Accuracy": 1.0,
                    "MCC": 1.0,
                    "F1": 1.0,
                }
                if include_hd95:
                    metric_dict["HD95"] = 0.0
            elif empty_policy == "nan":
                metric_dict = {
                    "Dice": np.nan,
                    "IoU": np.nan,
                    "Sensitivity": np.nan,
                    "Specificity": np.nan,
                    "Precision": np.nan,
                    "Accuracy": np.nan,
                    "MCC": np.nan,
                    "F1": np.nan,
                }
                if include_hd95:
                    metric_dict["HD95"] = np.nan
            else:
                raise ValueError("empty_policy must be 'perfect' or 'nan'")

            results.append({k: metric_dict[k] for k in metrics if k in metric_dict})
            continue

        tp, tn, fp, fn = _confusion_from_bin(pb, g)
        metric_dict = _binary_metric_dict(tp, tn, fp, fn)

        # Optional stricter handling for one-empty cases on some metrics
        if one_empty:
            if int(g.sum()) == 0:
                metric_dict["Sensitivity"] = 0.0 if empty_policy == "perfect" else np.nan
            if int(pb.sum()) == 0:
                metric_dict["Precision"] = 0.0 if empty_policy == "perfect" else np.nan

        if include_hd95:
            metric_dict["HD95"] = _safe_hd95(
                pb, g, spacing=spacing, empty_value=empty_hd95_value
            )

        results.append({k: metric_dict[k] for k in metrics if k in metric_dict})

    return _reduce_dict(results, reduce=reduce)


# =========================================================
# Dataset-Specific Wrappers
# =========================================================
def calc_isic_metrics(
    pred,
    gt,
    thr=0.5,
    pred_type="auto",
    reduce="mean",
    gt_thr=0.5,
    empty_policy="perfect",
):
    """
    Skin lesion segmentation metrics.
    Commonly reported: Dice, IoU, Sensitivity, Specificity, Accuracy
    """
    return calc_binary_seg_metrics(
        pred=pred,
        gt=gt,
        metrics=["Dice", "IoU", "Sensitivity", "Specificity", "Accuracy"],
        thr=thr,
        pred_type=pred_type,
        gt_thr=gt_thr,
        reduce=reduce,
        roi=None,
        include_hd95=False,
        empty_policy=empty_policy,
    )


def calc_busi_metrics(
    pred,
    gt,
    spacing=(1.0, 1.0),
    thr=0.5,
    pred_type="auto",
    reduce="mean",
    gt_thr=0.5,
    empty_hd95_value=np.nan,
    empty_policy="perfect",
):
    """
    Breast ultrasound segmentation metrics.
    Commonly reported: Dice, IoU, Sensitivity, Precision, HD95
    """
    return calc_binary_seg_metrics(
        pred=pred,
        gt=gt,
        metrics=["Dice", "IoU", "Sensitivity", "Precision", "HD95"],
        thr=thr,
        pred_type=pred_type,
        gt_thr=gt_thr,
        reduce=reduce,
        roi=None,
        spacing=spacing,
        include_hd95=True,
        empty_hd95_value=empty_hd95_value,
        empty_policy=empty_policy,
    )


def calc_retinal_metrics(
    pred,
    gt,
    thr=0.5,
    pred_type="auto",
    reduce="mean",
    roi=None,
    gt_thr=0.5,
    empty_policy="perfect",
):
    """
    Retinal vessel segmentation metrics.
    Commonly reported: Dice, IoU, Sensitivity, Specificity, MCC
    ROI/FOV masking is supported.
    """
    return calc_binary_seg_metrics(
        pred=pred,
        gt=gt,
        metrics=["Dice", "IoU", "Sensitivity", "Specificity", "MCC"],
        thr=thr,
        pred_type=pred_type,
        gt_thr=gt_thr,
        reduce=reduce,
        roi=roi,
        include_hd95=False,
        empty_policy=empty_policy,
    )


# =========================================================
# Dataset-aware metric helpers
# =========================================================
def is_busi_dataset(dataset_name: str) -> bool:
    return dataset_name in BUSI_DATASETS


def is_isic_dataset(dataset_name: str) -> bool:
    return dataset_name in ISIC_DATASETS


def is_retinal_dataset(dataset_name: str) -> bool:
    return dataset_name in RETINAL_DATASETS


def pick_metric_fn(dataset_name: str):
    if is_busi_dataset(dataset_name):
        return calc_busi_metrics
    if is_retinal_dataset(dataset_name):
        return calc_retinal_metrics
    if is_isic_dataset(dataset_name):
        return calc_isic_metrics
    raise ValueError(
        f"Unsupported dataset_name: {dataset_name}. "
        f"Available: {sorted(BUSI_DATASETS | ISIC_DATASETS | RETINAL_DATASETS)}"
    )


def get_metric_columns(dataset_name: str):
    if is_busi_dataset(dataset_name):
        return ["Dice", "IoU", "Sensitivity", "Precision", "HD95"]
    if is_retinal_dataset(dataset_name):
        return ["Dice", "IoU", "Sensitivity", "Specificity", "MCC"]
    if is_isic_dataset(dataset_name):
        return ["Dice", "IoU", "Sensitivity", "Specificity", "Accuracy"]
    raise ValueError(
        f"Unsupported dataset_name: {dataset_name}. "
        f"Available: {sorted(BUSI_DATASETS | ISIC_DATASETS | RETINAL_DATASETS)}"
    )


def format_metric_summary(dataset_name: str, metrics_dict: dict) -> str:
    metric_columns = get_metric_columns(dataset_name)
    return " | ".join(
        f"{name}: {metrics_dict[name]:.4f}"
        for name in metric_columns
        if name in metrics_dict
    )