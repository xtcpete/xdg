import os
import sys

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}.")
        self.linear = linear
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()
        self.lora_a = nn.Linear(linear.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, linear.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)
        for param in self.linear.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def _as_mapping(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _get_lora_attr(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _replace_lora_linears(module, target_modules, rank, alpha, dropout):
    replaced = 0
    target_modules = set(target_modules)
    for child_name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and child_name in target_modules:
            setattr(
                module,
                child_name,
                LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
            )
            replaced += 1
            continue
        replaced += _replace_lora_linears(child, target_modules, rank, alpha, dropout)
    return replaced


def _append_local_dependency_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(current_dir))
    dependency_path = os.path.join(repo_root, "Depth-Anything-3", "src")
    if os.path.isdir(dependency_path) and dependency_path not in sys.path:
        sys.path.append(dependency_path)


def _load_depth_anything3():
    _append_local_dependency_paths()
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "Depth Anything 3 backend is unavailable. Install `depth_anything_3` "
            "or use the vendored `Depth-Anything-3/src` checkout."
        ) from exc
    return DepthAnything3


def _load_da3_backbone_only(
    model_path=None,
    model_name="da3-base",
    load_pretrained=False,
):
    DepthAnything3 = _load_depth_anything3()
    if load_pretrained and model_path is not None:
        da3_model = DepthAnything3.from_pretrained(model_path)
    else:
        da3_model = DepthAnything3(model_name=model_name)

    feature_net = da3_model.model.backbone

    # The classifier only consumes encoder features, so drop the unused DA3
    # heads before the XDG checkpoint is loaded.
    for attr in ("head", "cam_dec", "cam_enc", "gs_head", "gs_adapter"):
        if hasattr(da3_model.model, attr):
            delattr(da3_model.model, attr)
    for attr in ("input_processor", "output_processor"):
        if hasattr(da3_model, attr):
            delattr(da3_model, attr)

    return feature_net


class Backbone(nn.Module):
    """DA3 feature extractor used by the classifier."""

    def __init__(
        self,
        model_path=None,
        model_name="da3-base",
        freeze=True,
        img_size=504,
        lora=None,
        train_camera_token=False,
        load_pretrained=False,
    ):
        super().__init__()
        lora_cfg = _as_mapping(lora)
        self.lora_enabled = bool(_get_lora_attr(lora_cfg, "enabled", False))
        if self.lora_enabled:
            train_camera_token = _get_lora_attr(lora_cfg, "train_camera_token", True)
        self.train_camera_token = bool(train_camera_token)
        self.freeze_requested = bool(freeze)
        self.freeze = self.freeze_requested and not self.lora_enabled and not self.train_camera_token
        self.eval_frozen_backbone = self.freeze_requested and not self.lora_enabled
        self.img_size = img_size
        self.feature_net = _load_da3_backbone_only(
            model_path=model_path,
            model_name=model_name,
            load_pretrained=load_pretrained,
        )
        self.patch_size = int(
            getattr(getattr(self.feature_net, "pretrained", None), "patch_size", 14)
        )

        if self.lora_enabled:
            for param in self.feature_net.parameters():
                param.requires_grad_(False)
            target_modules = _get_lora_attr(
                lora_cfg,
                "target_modules",
                ["qkv", "proj", "fc1", "fc2", "w12", "w3"],
            )
            replaced = _replace_lora_linears(
                self.feature_net,
                target_modules=target_modules,
                rank=int(_get_lora_attr(lora_cfg, "rank", 8)),
                alpha=float(_get_lora_attr(lora_cfg, "alpha", 16.0)),
                dropout=float(_get_lora_attr(lora_cfg, "dropout", 0.0)),
            )
            if replaced <= 0:
                raise ValueError(
                    f"LoRA enabled, but no target modules were replaced: {target_modules}."
                )
            self._set_camera_token_requires_grad(self.train_camera_token)
            self.feature_net.train(self.training)
        elif self.freeze_requested:
            self.feature_net.eval()
            for param in self.feature_net.parameters():
                param.requires_grad_(False)
            self._set_camera_token_requires_grad(self.train_camera_token)
        else:
            self.feature_net.train(self.training)
            for param in self.feature_net.parameters():
                param.requires_grad_(True)

    def _set_camera_token_requires_grad(self, requires_grad):
        camera_token = getattr(
            getattr(self.feature_net, "pretrained", None),
            "camera_token",
            None,
        )
        if camera_token is not None:
            camera_token.requires_grad_(bool(requires_grad))

    def train(self, mode=True):
        super().train(mode)
        if self.eval_frozen_backbone:
            self.feature_net.eval()
        else:
            self.feature_net.train(mode)
        return self

    def forward(self, x, masks=None):
        if x.dim() != 5:
            raise ValueError(f"Expected x as (B,N,3,H,W), got {tuple(x.shape)}")
        if x.shape[2] != 3:
            raise ValueError(f"Expected RGB inputs with C=3, got {x.shape[2]}")
        if self.freeze:
            with torch.no_grad():
                feats, _ = self.feature_net(x, masks=masks)
        else:
            feats, _ = self.feature_net(x, masks=masks)
        return feats
