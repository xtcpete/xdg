import importlib
import math
import os
import random
import time
from argparse import Namespace

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageDraw
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from torch.utils.data._utils.collate import default_collate

from src.datasets.pairwise_disambiguation_dataset import get_datasets
from src.utils.dataset import normalize_target_size
from src.utils.prediction import vote_ensemble_logits, vote_symmetrical_logits
from training.utils import (
    FocalLoss,
    compute_ap,
    compute_validation_metrics,
    plot_pr_curve,
)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


class MultiResolutionBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, resolutions, seed=0):
        super().__init__(sampler, batch_size, drop_last)
        self.resolutions = [normalize_target_size(resolution) for resolution in resolutions]
        if not self.resolutions:
            raise ValueError("MultiResolutionBatchSampler requires at least one resolution.")
        self.seed = int(seed)
        self._iter_count = 0

    def __iter__(self):
        rng = random.Random(self.seed + self._iter_count)
        self._iter_count += 1
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                resolution = rng.choice(self.resolutions)
                yield [(sample_idx, resolution) for sample_idx in batch]
                batch = []
        if batch and not self.drop_last:
            resolution = rng.choice(self.resolutions)
            yield [(sample_idx, resolution) for sample_idx in batch]


def stereo_pair_collate(batch):
    collated = default_collate(batch)
    target_size = collated.get("target_size")
    if target_size is not None and target_size.numel() > 0:
        collated["batch_resolution"] = target_size[0]
    return collated


class DoppelgangersDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.seed = int(getattr(getattr(cfg, "trainer", None), "seed", 100))
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        for split in ("train", "test"):
            split_cfg = getattr(self.cfg.data, split, None)
            if split_cfg is not None and not hasattr(split_cfg, "seed"):
                setattr(split_cfg, "seed", self.seed)
        self.train_dataset, self.val_dataset = get_datasets(self.cfg.data)

    def _get_resolution_candidates(self):
        candidates = getattr(self.cfg.data.train, "resolutions", None)
        if candidates is None:
            candidates = getattr(self.cfg.data.train, "img_sizes", None)
        if candidates is None:
            return None
        return [normalize_target_size(candidate) for candidate in candidates]

    def _make_generator(self, offset=0):
        generator = torch.Generator()
        generator.manual_seed(self.seed + int(offset))
        return generator

    def _build_sampler(self, dataset, shuffle, generator=None):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return DistributedSampler(dataset, shuffle=shuffle, seed=self.seed)
        if shuffle:
            return RandomSampler(dataset, generator=generator)
        return SequentialSampler(dataset)

    def train_dataloader(self):
        sampler = self._build_sampler(
            self.train_dataset,
            shuffle=True,
            generator=self._make_generator(0),
        )
        resolutions = self._get_resolution_candidates()
        batch_size = self.cfg.data.train.batch_size
        common_kwargs = dict(
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            collate_fn=stereo_pair_collate,
            worker_init_fn=seed_worker,
            generator=self._make_generator(1),
        )
        if resolutions:
            batch_sampler = MultiResolutionBatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=False,
                resolutions=resolutions,
                seed=self.seed + 2,
            )
            return DataLoader(self.train_dataset, batch_sampler=batch_sampler, **common_kwargs)
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            **common_kwargs,
        )

    def val_dataloader(self):
        sampler = self._build_sampler(self.val_dataset, shuffle=False)
        batch_size = self.cfg.data.test.batch_size
        if getattr(self.val_dataset, "mode", None) == "dust3r_like":
            batch_size = 1
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            collate_fn=stereo_pair_collate,
            worker_init_fn=seed_worker,
            generator=self._make_generator(100),
        )


class DoppelgangersLitModule(pl.LightningModule):
    def __init__(self, cfg, max_vis=4, load_backbone_pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters({"trainer": self._namespace_to_dict(cfg.trainer)})

        decoder_lib = importlib.import_module(cfg.model.type)
        self.model = decoder_lib.decoder(
            cfg.model,
            load_backbone_pretrained=load_backbone_pretrained,
        )
        self.criterion = FocalLoss()
        self.max_vis = max_vis
        self._viz_batch = None
        self._last_train_viz_step = None
        self._last_val_viz_step = None
        self._val_indices = []
        self._val_gt = []
        self._val_scores = []
        self._measure_val_inference_time = False
        self._compute_val_loss = True
        self._val_inference_time_total_s = 0.0
        self._val_inference_pair_count = 0
        self._optimizer_base_lrs = None
        self._warmup_epochs = float(getattr(cfg.trainer.opt, "warmup_epoch", 0) or 0)
        self._scheduler_type = getattr(cfg.trainer.opt, "scheduler", None)
        self._scheduler_start_epoch = int(getattr(cfg.trainer.opt, "start_epoch", 0) or 0)
        self._scheduler_step_size = int(
            getattr(
                cfg.trainer.opt,
                "step_size",
                getattr(cfg.trainer.opt, "step_epoch", 1),
            )
            or 1
        )
        self._scheduler_gamma = float(getattr(cfg.trainer.opt, "gamma", 0.1))
        self._scheduler_min_lr = float(getattr(cfg.trainer.opt, "min_lr", 1e-8))
        self._warmup_steps = 0

    @staticmethod
    def _namespace_to_dict(value):
        if isinstance(value, Namespace):
            return {
                key: DoppelgangersLitModule._namespace_to_dict(val)
                for key, val in vars(value).items()
            }
        if isinstance(value, dict):
            return {
                key: DoppelgangersLitModule._namespace_to_dict(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [DoppelgangersLitModule._namespace_to_dict(val) for val in value]
        if isinstance(value, tuple):
            return tuple(DoppelgangersLitModule._namespace_to_dict(val) for val in value)
        return value

    def forward(self, images, masks=None):
        return self.model(images, masks=masks)

    def _uses_symmetrical_logits(self):
        return bool(getattr(self.model, "return_symmetrical_logits", False))

    def _collapse_model_logits(self, logits):
        if self._uses_symmetrical_logits():
            return vote_symmetrical_logits(logits)
        return logits

    def _targets_for_logits(self, logits, gt):
        if self._uses_symmetrical_logits():
            expected = int(gt.shape[0]) * 2
            if int(logits.shape[0]) != expected:
                raise ValueError(
                    f"Expected {expected} symmetric logits for {gt.shape[0]} labels, "
                    f"got {logits.shape[0]}."
                )
            return gt.repeat(2)
        if int(logits.shape[0]) != int(gt.shape[0]):
            raise ValueError(
                f"Logit/label batch mismatch: {logits.shape[0]} logits for {gt.shape[0]} labels."
            )
        return gt

    def _classification_loss(self, logits, gt):
        return self.criterion(logits, self._targets_for_logits(logits, gt))

    def training_step(self, batch, batch_idx):
        images = batch["images"].float()
        gt = batch["gt"].long()
        masks = batch.get("masks")
        raw_logits = self(images, masks=masks)
        loss = self._classification_loss(raw_logits, gt)
        logits = self._collapse_model_logits(raw_logits)

        if 'images_resized' in batch:
            images_resized = batch['images_resized']
            raw_logits_resized = self(images_resized, masks=masks)
            logits_resized = self._collapse_model_logits(raw_logits_resized)
            loss = 0.5 * (loss + self._classification_loss(raw_logits_resized, gt))

            logits = torch.cat([logits, logits_resized], dim=0)
            gt = torch.cat([gt, gt], dim=0)

        ap = compute_ap(gt, logits.detach())
        current_lr = self._get_current_lr()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ap", ap, on_step=True, on_epoch=True, prog_bar=True)
        if current_lr is not None:
            self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        self._log_train_visualization(images, gt, logits)
        return loss

    def on_after_backward(self):
        if getattr(self, "_reported_unused_params", False):
            return
        unused = [
            name
            for name, param in self.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        if unused:
            self.print(f"Unused parameters (first batch): {unused}")
        self._reported_unused_params = True

    def on_validation_epoch_start(self):
        self._viz_batch = None
        self._val_indices = []
        self._val_gt = []
        self._val_scores = []
        self._last_val_viz_step = None
        self._val_inference_time_total_s = 0.0
        self._val_inference_pair_count = 0

    def validation_step(self, batch, batch_idx):
        images = batch["images"].float()
        gt = batch["gt"].long()
        pair_idx = batch["pair_idx"].long()
        masks = batch.get("masks")
        should_compute_loss = bool(getattr(self, "_compute_val_loss", True))
        should_time = bool(getattr(self, "_measure_val_inference_time", False))
        if should_time:
            self._synchronize_device(images.device)
            start_time = time.perf_counter()

        raw_logits = self(images, masks=masks)
        logits = self._collapse_model_logits(raw_logits)
        eval_logits = logits
        loss = self._classification_loss(raw_logits, gt) if should_compute_loss else None

        if "images_resized" in batch:
            images_resized = batch["images_resized"].float()
            raw_logits_resized = self(images_resized, masks=masks)
            logits_resized = self._collapse_model_logits(raw_logits_resized)
            eval_logits = vote_ensemble_logits(logits, logits_resized)
            if should_compute_loss:
                loss = 0.5 * (loss + self._classification_loss(raw_logits_resized, gt))

        if should_time:
            self._synchronize_device(images.device)
            elapsed = time.perf_counter() - start_time
            self._val_inference_time_total_s += float(elapsed)
            self._val_inference_pair_count += int(gt.shape[0])

        probs = torch.softmax(eval_logits, dim=1)
        if loss is not None:
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        pred = probs.argmax(dim=1)
        pred_prob = probs.gather(1, pred.view(-1, 1)).squeeze(1)
        self._viz_batch = {
            "images": images.detach().cpu(),
            "gt": gt.detach().cpu(),
            "pred": pred.detach().cpu(),
            "prob": pred_prob.detach().cpu(),
        }
        self._log_val_visualization(step=batch_idx)
        self._val_indices.append(pair_idx.detach())
        self._val_gt.append(gt.detach())
        self._val_scores.append(eval_logits.detach().float())
        if loss is not None:
            return {"val_loss": loss}
        return {}

    def on_validation_epoch_end(self):
        outputs = self._consolidate_validation_outputs()
        if outputs is not None:
            _, y_true, y_scores = outputs
            self._val_gt = [y_true]
            self._val_scores = [y_scores]
            metrics = compute_validation_metrics(y_true, y_scores)
            self.log("val_ap", metrics["ap"], on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "val_roc_auc",
                metrics["roc_auc"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val_prec_at_recall_0_85",
                metrics["prec_at_recall_0_85"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val_recall_at_prec_0_99",
                metrics["recall_at_prec_0_99"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            timing_metrics = self._consolidate_validation_timing()
            if timing_metrics is not None:
                self.log(
                    "val_inference_time_total_s",
                    timing_metrics["total_s"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "val_inference_time_per_pair_ms",
                    timing_metrics["per_pair_ms"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
                self.log(
                    "val_num_pairs",
                    timing_metrics["num_pairs"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
        self._log_pr_curve()

    def _consolidate_validation_outputs(self):
        if not self._val_indices or not self._val_gt or not self._val_scores:
            return None

        pair_idx = torch.cat(self._val_indices, dim=0)
        y_true = torch.cat(self._val_gt, dim=0)
        y_scores = torch.cat(self._val_scores, dim=0)

        if getattr(self.trainer, "world_size", 1) > 1:
            pair_idx = self.all_gather(pair_idx).reshape(-1)
            y_true = self.all_gather(y_true).reshape(-1)
            y_scores = self.all_gather(y_scores).reshape(-1, y_scores.shape[-1])

        pair_idx = pair_idx.cpu()
        y_true = y_true.cpu()
        y_scores = y_scores.float().cpu()

        sort_order = torch.argsort(pair_idx)
        pair_idx = pair_idx[sort_order]
        y_true = y_true[sort_order]
        y_scores = y_scores[sort_order]

        if pair_idx.numel() > 1:
            keep = torch.ones(pair_idx.shape[0], dtype=torch.bool)
            keep[1:] = pair_idx[1:] != pair_idx[:-1]
            pair_idx = pair_idx[keep]
            y_true = y_true[keep]
            y_scores = y_scores[keep]

        return pair_idx, y_true, y_scores

    def _consolidate_validation_timing(self):
        if not getattr(self, "_measure_val_inference_time", False):
            return None
        if self._val_inference_pair_count <= 0:
            return None

        total_time = torch.tensor(
            self._val_inference_time_total_s,
            device=self.device,
            dtype=torch.float64,
        )
        num_pairs = torch.tensor(
            self._val_inference_pair_count,
            device=self.device,
            dtype=torch.long,
        )

        if getattr(self.trainer, "world_size", 1) > 1:
            total_time = self.all_gather(total_time).sum()
            num_pairs = self.all_gather(num_pairs).sum()

        total_time_value = float(total_time.item())
        num_pairs_value = int(num_pairs.item())
        if num_pairs_value <= 0:
            return None

        return {
            "total_s": total_time_value,
            "per_pair_ms": (total_time_value * 1000.0) / float(num_pairs_value),
            "num_pairs": float(num_pairs_value),
        }

    @staticmethod
    def _synchronize_device(device):
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def on_train_start(self):
        if not self._optimizer_base_lrs:
            return
        self._warmup_steps = self._compute_warmup_steps()
        self._apply_warmup_lr(self.global_step)

    def on_train_batch_start(self, batch, batch_idx):
        self._apply_warmup_lr(self.global_step)

    def _get_viz_interval(self, primary, default=0, fallback=None):
        viz_cfg = getattr(self.cfg, "viz", None)
        if viz_cfg is None:
            return default
        if hasattr(viz_cfg, primary):
            return int(getattr(viz_cfg, primary))
        if fallback and hasattr(viz_cfg, fallback):
            return int(getattr(viz_cfg, fallback))
        return default

    def _log_train_visualization(self, images, gt, logits):
        if self.logger is None or not self.trainer.is_global_zero:
            return
        interval = self._get_viz_interval("log_freq", default=0)
        if interval <= 0:
            return
        step = int(self.global_step)
        if self._last_train_viz_step == step or (step % interval != 0):
            return
        probs = torch.softmax(logits.detach(), dim=1)
        pred = probs.argmax(dim=1)
        pred_prob = probs.gather(1, pred.view(-1, 1)).squeeze(1)
        viz_batch = {
            "images": images.detach().cpu(),
            "gt": gt.detach().cpu(),
            "pred": pred.detach().cpu(),
            "prob": pred_prob.detach().cpu(),
        }
        image = self._make_side_by_side(viz_batch)
        if image is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if hasattr(experiment, "add_image"):
            experiment.add_image("train_examples", image, step, dataformats="HWC")
            self._last_train_viz_step = step

    def _log_val_visualization(self, step=None):
        if self._viz_batch is None or self.logger is None:
            return
        interval = self._get_viz_interval("val_plot_freq", default=0, fallback="val_freq")
        if interval <= 0:
            return
        step = int(0 if step is None else step)
        if self._last_val_viz_step == step or (step % interval != 0):
            return
        image = self._make_side_by_side(self._viz_batch)
        if image is None:
            return
        experiment = getattr(self.logger, "experiment", None)
        if hasattr(experiment, "add_image"):
            experiment.add_image("val_examples", image, step, dataformats="HWC")
            self._last_val_viz_step = step

    def _log_pr_curve(self):
        if self.logger is None or not self.trainer.is_global_zero:
            return
        if not self._val_gt or not self._val_scores:
            return
        experiment = getattr(self.logger, "experiment", None)
        if not hasattr(experiment, "add_image"):
            return
        y_true = torch.cat(self._val_gt, dim=0).cpu().numpy()
        y_scores = torch.cat(self._val_scores, dim=0).float().cpu().numpy()
        plot_pr_curve(y_true, y_scores, experiment, epoch=self.current_epoch, name="pr_curve")

    def _make_side_by_side(self, batch):
        images = batch["images"][: self.max_vis]
        gt = batch["gt"][: self.max_vis]
        pred = batch["pred"][: self.max_vis]
        prob = batch["prob"][: self.max_vis]

        rows = []
        for i in range(images.shape[0]):
            img0 = self._to_uint8(images[i, 0])
            img1 = self._to_uint8(images[i, 1])
            if img0 is None or img1 is None:
                continue
            pair = np.concatenate([img0, img1], axis=1)
            text = f"gt={int(gt[i])} pred={int(pred[i])} p={float(prob[i]):.3f}"
            pair = self._overlay_text(pair, text)
            rows.append(pair)

        if not rows:
            return None
        return np.concatenate(rows, axis=0)

    def _to_uint8(self, image):
        if image.dim() != 3:
            return None
        image = image.detach().cpu().clamp(0, 1)
        image = (image * 255.0).byte().permute(1, 2, 0).numpy()
        return image

    def _overlay_text(self, image, text):
        pil_img = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_img)
        draw.rectangle([(0, 0), (pil_img.size[0], 18)], fill=(0, 0, 0))
        draw.text((4, 2), text, fill=(255, 255, 255))
        return np.array(pil_img)

    def _scheduler_lambda(self, index, base_lr):
        scheduler_type = self._scheduler_type
        if scheduler_type is not None:
            scheduler_type = scheduler_type.lower()
        min_scale = 0.0
        if base_lr > 0.0:
            min_scale = min(1.0, self._scheduler_min_lr / float(base_lr))
        if scheduler_type == "step":
            return max(min_scale, self._lr_scale_for_step(index))
        return max(min_scale, self._lr_scale_for_epoch(index))

    def _lr_scale_for_epoch(self, epoch):
        total_epochs = max(1, int(getattr(self.cfg.trainer, "epochs", 1)))
        epoch = max(0, int(epoch))
        scheduler_type = self._scheduler_type
        if scheduler_type is not None:
            scheduler_type = scheduler_type.lower()

        if scheduler_type is None:
            return 1.0
        if scheduler_type == "linear":
            start_epoch = max(self._warmup_epochs, self._scheduler_start_epoch)
            if epoch < start_epoch:
                return 1.0
            decay_epochs = max(1, total_epochs - start_epoch)
            decay_progress = max(
                0.0, float((epoch - start_epoch) + 1) / float(decay_epochs)
            )
            return max(0.0, 1.0 - decay_progress)
        if scheduler_type == "step":
            if epoch < self._scheduler_start_epoch:
                return 1.0
            decay_count = ((epoch - self._scheduler_start_epoch) // self._scheduler_step_size) + 1
            return self._scheduler_gamma ** decay_count
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def _lr_scale_for_step(self, step):
        step = max(0, int(step))
        if self.trainer is None:
            return 1.0

        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        total_epochs = max(1, int(getattr(self.cfg.trainer, "epochs", 1)))
        if total_steps is None:
            steps_per_epoch = 1
        else:
            steps_per_epoch = max(1, int(math.ceil(float(total_steps) / float(total_epochs))))

        start_step = max(self._warmup_steps, self._scheduler_start_epoch * steps_per_epoch)
        if step < start_step:
            return 1.0

        decay_interval = max(1, int(self._scheduler_step_size * steps_per_epoch))
        decay_count = ((step - start_step) // decay_interval) + 1
        return self._scheduler_gamma ** decay_count

    def _compute_warmup_steps(self):
        if self.trainer is None or self._warmup_epochs <= 0:
            return 0
        total_steps = getattr(self.trainer, "estimated_stepping_batches", None)
        total_epochs = max(1, int(getattr(self.cfg.trainer, "epochs", 1)))
        if total_steps is None:
            return 0
        steps_per_epoch = max(1, int(math.ceil(float(total_steps) / float(total_epochs))))
        return max(1, int(math.ceil(self._warmup_epochs * steps_per_epoch)))

    def _apply_warmup_lr(self, step):
        if self._optimizer_base_lrs is None or self.trainer is None:
            return
        if self._warmup_steps <= 0 or step >= self._warmup_steps:
            return
        scale = float(step + 1) / float(self._warmup_steps)
        for optimizer in self.trainer.optimizers:
            for param_group, base_lr in zip(optimizer.param_groups, self._optimizer_base_lrs):
                param_group["lr"] = base_lr * scale

    def _get_current_lr(self):
        if self.trainer is None or not self.trainer.optimizers:
            return None
        return float(self.trainer.optimizers[0].param_groups[0]["lr"])

    def configure_optimizers(self):
        opt_cfg = self.cfg.trainer.opt
        lr = float(getattr(opt_cfg, "lr", 1e-4))
        weight_decay = float(getattr(opt_cfg, "weight_decay", 0.0))
        opt_type = getattr(opt_cfg, "type", "adamw").lower()
        backbone_lr = float(getattr(opt_cfg, "backbone_lr", 0.05 * lr))
        backbone_module = getattr(self.model, "backbone", None)
        backbone_trainable = backbone_module is not None and any(
            param.requires_grad for param in backbone_module.parameters()
        )

        if backbone_trainable:
            backbone_param_ids = {id(param) for param in backbone_module.parameters() if param.requires_grad}
            backbone_params = []
            other_params = []
            for param in self.parameters():
                if not param.requires_grad:
                    continue
                if id(param) in backbone_param_ids:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            optimizer_params = [{"params": other_params, "lr": lr}]
            if backbone_params:
                optimizer_params.append({"params": backbone_params, "lr": backbone_lr})
        else:
            optimizer_params = [param for param in self.parameters() if param.requires_grad]

        if opt_type == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_params,
                lr=lr,
                betas=(
                    float(getattr(opt_cfg, "beta1", 0.9)),
                    float(getattr(opt_cfg, "beta2", 0.999)),
                ),
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {opt_type}")

        self._optimizer_base_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler_type = getattr(opt_cfg, "scheduler", None)
        if scheduler_type is None and self._warmup_epochs <= 0:
            return optimizer

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=[
                (lambda index, base_lr=base_lr: self._scheduler_lambda(index, base_lr))
                for base_lr in self._optimizer_base_lrs
            ],
        )
        scheduler_interval = "step" if str(scheduler_type).lower() == "step" else "epoch"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": scheduler_interval},
        }
