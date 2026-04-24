from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader


def build_external_retinal_test_loaders(
    train_dataset_name,
    train_root,
    medical_dataset_cls,
    img_size,
    batch_size,
    num_workers,
    pin_memory=False,
):
    if train_dataset_name != "FIVES":
        return {}

    base_root = Path(train_root).parent
    loaders = {}
    for dataset_name in ("CHASEDB1", "DRIVE"):
        dataset_root = base_root / dataset_name
        test_image_dir = dataset_root / "test" / "images"
        if not test_image_dir.exists():
            continue
        try:
            dataset = medical_dataset_cls(
                str(dataset_root),
                dataset_name,
                split="test",
                img_size=img_size,
                augment=False,
                return_roi=True,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"[retinal_external_eval] 跳过 {dataset_name}: {e}")
            continue
        loaders[dataset_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    return loaders


def export_external_retinal_results(
    *,
    final_dir,
    filename_prefix,
    checkpoint_items,
    external_test_loaders,
    load_checkpoint_fn,
    eval_fn,
    metric_columns_fn=None,
):
    if not external_test_loaders:
        return

    for dataset_name, loader in external_test_loaders.items():
        rows = []
        for item in checkpoint_items:
            load_checkpoint_fn(item)
            metrics = eval_fn(loader, dataset_name, item)
            row = {
                "Epoch": item["epoch"],
                "Weight": Path(item["final_path"]).name,
            }
            metric_columns = metric_columns_fn(dataset_name) if metric_columns_fn is not None else list(metrics.keys())
            row.update({k: round(metrics[k], 4) for k in metric_columns if k in metrics})
            rows.append(row)
        pd.DataFrame(rows).to_excel(final_dir / f"{filename_prefix}_{dataset_name}_test_results.xlsx", index=False)


def run_external_retinal_tests(
    *,
    train_dataset_name,
    train_root,
    medical_dataset_cls,
    img_size,
    batch_size,
    num_workers,
    pin_memory,
    final_dir,
    filename_prefix,
    checkpoint_items,
    load_checkpoint_fn,
    eval_fn,
    metric_columns_fn=None,
):
    external_test_loaders = build_external_retinal_test_loaders(
        train_dataset_name=train_dataset_name,
        train_root=train_root,
        medical_dataset_cls=medical_dataset_cls,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if not checkpoint_items or not external_test_loaders:
        return

    export_external_retinal_results(
        final_dir=final_dir,
        filename_prefix=filename_prefix,
        checkpoint_items=checkpoint_items,
        external_test_loaders=external_test_loaders,
        load_checkpoint_fn=load_checkpoint_fn,
        eval_fn=eval_fn,
        metric_columns_fn=metric_columns_fn,
    )
