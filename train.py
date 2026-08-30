import argparse

import pytorch_lightning as pl
import torch

from src.utils.config import load_training_config
from training.lightning import DoppelgangersDataModule, DoppelgangersLitModule
from training.precision import resolve_mixed_precision
from training.utils import set_random_seed


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the XDG classifier with PyTorch Lightning."
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to a training config YAML that references the model config.",
    )
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Devices to use (e.g. auto, 1, 0,1).",
    )
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        '--precision',
        type=str,
        default='auto',
        help='Mixed precision policy. auto uses bf16 when supported, otherwise fp16.',
    )
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--max_vis", type=int, default=4)
    return parser.parse_args()


def devices_arg(devices: str):
    if devices == 'auto':
        return 'auto'
    if devices.isdigit():
        return int(devices)
    if ',' in devices:
        return [int(d) for d in devices.split(',')]
    return devices


def set_matmul_precision(cfg):
    precision = getattr(cfg.trainer, "matmul_precision", "high")
    if precision is not None and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(str(precision))


def build_trainer(cfg, args):
    precision = resolve_mixed_precision(args.precision)
    print(f"Using precision: {precision}")
    gradient_clip_val = float(getattr(cfg.trainer, "gradient_clip_val", 0.0) or 0.0)
    gradient_clip_algorithm = getattr(cfg.trainer, "gradient_clip_algorithm", "norm")
    ckpt_cb = pl.callbacks.ModelCheckpoint(
        filename="doppelgangers-{epoch:02d}-{val_ap:.4f}",
        save_top_k=3,
        monitor="val_ap",
        mode="max",
        save_last=True,
    )
    return pl.Trainer(
        default_root_dir=args.log_dir,
        max_epochs=cfg.trainer.epochs,
        accelerator=args.accelerator,
        devices=devices_arg(args.devices),
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm=gradient_clip_algorithm,
        callbacks=[ckpt_cb],
        log_every_n_steps=10,
    )


def train_main():
    args = get_args()

    # Older checkpoints in this repo may store argparse.Namespace in Lightning
    # hyperparameters. PyTorch 2.6 blocks that type by default with weights_only=True.
    torch.serialization.add_safe_globals([argparse.Namespace])

    cfg = load_training_config(args.config)

    if args.max_epochs is not None:
        cfg.trainer.epochs = args.max_epochs

    set_matmul_precision(cfg)

    seed = int(getattr(cfg.trainer, "seed", 42))
    set_random_seed(seed)
    pl.seed_everything(seed, workers=True)

    data_module = DoppelgangersDataModule(cfg)
    lit_module = DoppelgangersLitModule(
        cfg,
        max_vis=args.max_vis,
        load_backbone_pretrained=True,
    )
    trainer = build_trainer(cfg, args)
    trainer.fit(lit_module, datamodule=data_module, ckpt_path=args.resume_from)


if __name__ == "__main__":
    train_main()
