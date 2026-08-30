import argparse
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils.config import load_training_config
from training.lightning import DoppelgangersDataModule, DoppelgangersLitModule
from training.precision import resolve_mixed_precision
from training.utils import set_random_seed


TEST_DATASETS = ("doppelgangers", "visymscenes")


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the XDG classifier with PyTorch Lightning."
    )
    parser.add_argument(
        "--config",
        default='./configs/training_configs/training.yaml',
        type=str,
        help="Path to a training config YAML that references the model config.",
    )
    parser.add_argument("--log_dir", type=str, default="val_logs")
    parser.add_argument("--devices", type=str, default="auto", help="Devices to use.")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        help="Mixed precision policy. auto uses bf16 when supported, otherwise fp16.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path to load for testing.",
    )
    parser.add_argument(
        "--dataset",
        choices=TEST_DATASETS,
        default="doppelgangers",
        help="Dataset to evaluate (default: doppelgangers).",
    )
    parser.add_argument(
        "--allow_random_init",
        action="store_true",
        help="Run validation with randomly initialized classifier weights.",
    )
    parser.add_argument(
        "--allow_partial_load",
        action="store_true",
        help="Allow checkpoints with missing or unexpected model keys.",
    )
    parser.add_argument("--max_vis", type=int, default=4)
    return parser.parse_args()


def devices_arg(devices: str):
    if devices == "auto":
        return "auto"
    if devices.isdigit():
        return int(devices)
    if "," in devices:
        return [int(d) for d in devices.split(",")]
    return devices


def build_logger(log_dir, cfg_name):
    run_time = time.strftime("%Y-%b-%d-%H-%M-%S")
    name = f"{cfg_name}_val_{run_time}"
    os.makedirs(log_dir, exist_ok=True)
    return TensorBoardLogger(save_dir=log_dir, name=name)


def set_matmul_precision(cfg):
    precision = getattr(cfg.trainer, "matmul_precision", "high")
    if precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(precision))


def select_test_dataset(cfg, dataset_name):
    test_sets = getattr(cfg.data, "test_sets", None)
    dataset_cfg = getattr(test_sets, dataset_name, None)
    if dataset_cfg is None:
        raise ValueError(
            f"Dataset '{dataset_name}' is not defined under data.test_sets."
        )

    for field in ("image_dir", "pair_path"):
        if not hasattr(dataset_cfg, field):
            raise ValueError(
                f"data.test_sets.{dataset_name} must define '{field}'."
            )
        setattr(cfg.data.test, field, getattr(dataset_cfg, field))

    print(f"Using test dataset: {dataset_name}")


def _match_state_dict_prefix(state_dict, module_state_dict):
    candidates = [
        ("as-is", state_dict),
        (
            "add model.",
            {f"model.{key}": value for key, value in state_dict.items()},
        ),
        (
            "strip model.",
            {
                key[len("model.") :]: value
                for key, value in state_dict.items()
                if key.startswith("model.")
            },
        ),
        (
            "strip module.",
            {
                key[len("module.") :]: value
                for key, value in state_dict.items()
                if key.startswith("module.")
            },
        ),
        (
            "strip model.module.",
            {
                key[len("model.module.") :]: value
                for key, value in state_dict.items()
                if key.startswith("model.module.")
            },
        ),
    ]

    best_name = "as-is"
    best_state_dict = state_dict
    best_overlap = -1
    target_keys = set(module_state_dict.keys())
    for name, candidate in candidates:
        if not candidate:
            continue
        overlap = len(target_keys.intersection(candidate.keys()))
        if overlap > best_overlap:
            best_name = name
            best_state_dict = candidate
            best_overlap = overlap
    return best_name, best_state_dict, best_overlap


def load_weights_for_validation(lit_module, ckpt_path, allow_partial_load=False):
    if ckpt_path is None:
        return None

    # Older checkpoints in this repo may store argparse.Namespace in Lightning
    # hyperparameters. PyTorch 2.6 blocks that type by default with weights_only=True.
    torch.serialization.add_safe_globals([argparse.Namespace])

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type at {ckpt_path}: {type(checkpoint)!r}")

    is_lightning_checkpoint = (
        "state_dict" in checkpoint and "pytorch-lightning_version" in checkpoint
    )
    raw_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    if not isinstance(raw_state_dict, dict):
        raise TypeError(
            f"Checkpoint state_dict at {ckpt_path} must be a dict, got {type(raw_state_dict)!r}"
        )

    prefix_mode, state_dict, overlap = _match_state_dict_prefix(
        raw_state_dict, lit_module.state_dict()
    )
    if overlap <= 0:
        raise ValueError(
            f"Checkpoint at {ckpt_path} does not match the current model state_dict keys."
        )
    missing, unexpected = lit_module.load_state_dict(state_dict, strict=False)

    ckpt_kind = "Lightning checkpoint" if is_lightning_checkpoint else "weights-only checkpoint"
    print(f"Loaded {ckpt_kind} from {ckpt_path} using key mapping: {prefix_mode}")
    print(f"Matched checkpoint keys: {overlap}/{len(lit_module.state_dict())}")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
    if not allow_partial_load and (missing or unexpected):
        raise ValueError(
            "Checkpoint does not exactly match the current model. "
            "This usually means the config and checkpoint architectures differ. "
            "Use the matching checkpoint, or pass --allow_partial_load only for debugging."
        )
    if allow_partial_load and (missing or unexpected):
        print("Warning: checkpoint was only partially loaded.")
    return None


def test_main():
    args = get_args()
    if args.ckpt is None and not args.allow_random_init:
        raise ValueError(
            "--ckpt is required for testing a trained model. "
            "Pass --allow_random_init only when intentionally measuring an untrained baseline."
        )

    cfg = load_training_config(args.config)
    select_test_dataset(cfg, args.dataset)

    cfg_name = os.path.splitext(os.path.basename(args.config))[0]
    logger = build_logger(args.log_dir, cfg_name)

    set_matmul_precision(cfg)

    seed = int(getattr(cfg.trainer, "seed", 100))
    set_random_seed(seed)
    pl.seed_everything(seed, workers=True)

    data_module = DoppelgangersDataModule(cfg)
    lit_module = DoppelgangersLitModule(
        cfg,
        max_vis=args.max_vis,
        load_backbone_pretrained=False,
    )
    lit_module._measure_val_inference_time = True
    lit_module._compute_val_loss = False

    precision = resolve_mixed_precision(args.precision)
    print(f"Using precision: {precision}")

    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        accelerator=args.accelerator,
        devices=devices_arg(args.devices),
        precision=precision,
        logger=logger,
        log_every_n_steps=10,
    )

    ckpt_path = load_weights_for_validation(
        lit_module,
        args.ckpt,
        allow_partial_load=args.allow_partial_load,
    )
    results = trainer.validate(lit_module, datamodule=data_module, ckpt_path=ckpt_path)
    _print_validation_summary(results)


def _as_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _print_validation_summary(results):
    if not results:
        return
    metrics = results[0]
    metric_order = [
        ("AP↑", "val_ap"),
        ("ROC AUC↑", "val_roc_auc"),
        ("Prec@Recall=0.85↑", "val_prec_at_recall_0_85"),
        ("Recall@Prec=0.99↑", "val_recall_at_prec_0_99"),
        ("Inference / pair↓ (ms)", "val_inference_time_per_pair_ms"),
    ]

    print("Validation metrics:")
    for label, key in metric_order:
        value = metrics.get(key)
        if value is None:
            print(f"{label}: n/a")
            continue
        value = _as_float(value)
        print(f"{label}: {value:.6f}")

    total_time = metrics.get("val_inference_time_total_s")
    num_pairs = metrics.get("val_num_pairs")
    if total_time is not None and num_pairs is not None:
        print(f"Total inference time (s): {_as_float(total_time):.6f}")
        print(f"Evaluated pairs: {int(round(_as_float(num_pairs)))}")


if __name__ == '__main__':
    test_main()
