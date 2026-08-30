import os
from types import SimpleNamespace

os.environ["DA3_LOG_LEVEL"] = "ERROR"

import torch
import torch.nn as nn
import torchvision.transforms as T

from .backbone import Backbone


DEFAULT_INPUT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_INPUT_STD = [0.229, 0.224, 0.225]


def _ensure_namespace(value):
    if value is None:
        return SimpleNamespace()
    if isinstance(value, dict):
        return SimpleNamespace(**value)
    return value


class CameraTokenClassifier(nn.Module):
    def __init__(
        self,
        *,
        enc_dim=1536,
        dec_dim=768,
        hidden_dim=768,
        dropout=0.1,
        num_stages=4,
        num_classes=2,
        pool="mean",
    ):
        super().__init__()
        self.num_stages = num_stages
        self.pool = pool
        self.stage_projs = nn.ModuleList(
            [nn.Linear(enc_dim, dec_dim) for _ in range(num_stages)]
        )
        self.stage_norms = nn.ModuleList(
            [nn.LayerNorm(dec_dim) for _ in range(num_stages)]
        )

        hidden_dim = int(hidden_dim or 0)
        if hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.LayerNorm(dec_dim),
                nn.Linear(dec_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(dec_dim),
                nn.Linear(dec_dim, num_classes),
            )

    @staticmethod
    def _extract_camera_token(feature):
        if isinstance(feature, (tuple, list)) and len(feature) >= 2:
            return feature[1]
        raise ValueError(
            "CameraTokenClassifier requires DA3 features with camera tokens."
        )

    def _build_tokens(self, raw_features):
        if raw_features is None or len(raw_features) < self.num_stages:
            count = 0 if raw_features is None else len(raw_features)
            raise ValueError(
                f"Expected at least {self.num_stages} feature stages, got {count}."
            )

        stage_tokens = []
        for index, feature in enumerate(raw_features[: self.num_stages]):
            camera_token = self._extract_camera_token(feature)
            if camera_token.dim() != 3:
                raise ValueError(
                    "Expected camera tokens with shape [batch, views, channels], "
                    f"got {tuple(camera_token.shape)}."
                )
            camera_token = self.stage_projs[index](camera_token)
            stage_tokens.append(self.stage_norms[index](camera_token))
        return torch.cat(stage_tokens, dim=1)

    def forward(self, raw_feats):
        tokens = self._build_tokens(raw_feats)
        if self.pool == "mean":
            pooled = tokens.mean(dim=1)
        elif self.pool == "max":
            pooled = tokens.max(dim=1).values
        else:
            raise ValueError(f"Unsupported camera token pool mode '{self.pool}'.")
        return self.proj(pooled)


class decoder(nn.Module):
    """Released XDG classifier built from DA3 camera-token features."""

    def __init__(self, cfg=None, load_backbone_pretrained=False):
        super().__init__()
        cfg = _ensure_namespace(cfg)
        backbone_cfg = _ensure_namespace(getattr(cfg, "backbone", None))
        classifier_cfg = _ensure_namespace(getattr(cfg, "classifier", None))
        normalize_cfg = _ensure_namespace(getattr(cfg, "input_normalize", None))

        classifier_type = str(getattr(classifier_cfg, "type", "camera_token")).lower()
        if classifier_type not in {"camera_token", "camera_token_classifier"}:
            raise ValueError(
                "The release model only supports the camera_token classifier, "
                f"got '{classifier_type}'."
            )

        self.symmetrical = bool(getattr(cfg, "symmetrical", True))
        self.return_symmetrical_logits = bool(
            getattr(cfg, "return_symmetrical_logits", False)
        )
        if self.return_symmetrical_logits and not self.symmetrical:
            raise ValueError("return_symmetrical_logits requires symmetrical=True.")

        mean = getattr(normalize_cfg, "mean", DEFAULT_INPUT_MEAN)
        std = getattr(normalize_cfg, "std", DEFAULT_INPUT_STD)
        self.input_mean = mean
        self.input_std = std
        self.input_normalize = T.Normalize(mean=mean, std=std)

        self.backbone = Backbone(
            model_path=getattr(backbone_cfg, "model_path", None),
            model_name=getattr(backbone_cfg, "model_name", "da3-base"),
            freeze=getattr(backbone_cfg, "freeze", True),
            img_size=getattr(backbone_cfg, "img_size", 560),
            lora=getattr(backbone_cfg, "lora", None),
            train_camera_token=getattr(backbone_cfg, "train_camera_token", False),
            load_pretrained=load_backbone_pretrained,
        )

        num_classes = int(getattr(cfg, "num_classes", 2))
        enc_dim = int(getattr(classifier_cfg, "enc_dim", 1536))
        dec_dim = int(getattr(classifier_cfg, "dec_dim", 768))
        num_stages = int(getattr(classifier_cfg, "num_stages", 4))
        dropout = float(
            getattr(
                classifier_cfg,
                "dec_dropout",
                getattr(classifier_cfg, "dropout", 0.1),
            )
        )

        self.raw_agg_mlp = None
        if self.symmetrical and not self.return_symmetrical_logits:
            hidden_ratio = float(
                getattr(classifier_cfg, "sym_agg_hidden_ratio", 1.0)
            )
            aggregation_dim = max(1, int(round(enc_dim * hidden_ratio)))
            self.raw_agg_mlp = nn.Sequential(
                nn.Linear(enc_dim * 2, aggregation_dim),
                nn.GELU(),
                nn.Linear(aggregation_dim, enc_dim),
            )

        self.num_stages = num_stages
        self.classifier = CameraTokenClassifier(
            enc_dim=enc_dim,
            dec_dim=dec_dim,
            hidden_dim=getattr(classifier_cfg, "hidden_dim", dec_dim),
            dropout=dropout,
            num_stages=num_stages,
            num_classes=num_classes,
            pool=getattr(classifier_cfg, "pool", "mean"),
        )

    def _aggregate_symmetrical_camera_features(self, features):
        if features is None:
            return None
        aggregated = []
        for feature in features:
            if not isinstance(feature, (tuple, list)) or len(feature) < 2:
                raise ValueError(
                    "Symmetrical aggregation requires DA3 features with camera tokens."
                )
            camera_token = feature[1]
            if camera_token.shape[0] % 2 != 0:
                raise ValueError(
                    "Symmetrical aggregation expects an even batch size, "
                    f"got {camera_token.shape[0]}."
                )
            half = camera_token.shape[0] // 2
            forward = camera_token[:half]
            backward = torch.flip(camera_token[half:], dims=[1])
            camera_token = self.raw_agg_mlp(
                torch.cat([forward, backward], dim=-1)
            )
            aggregated.append((feature[0], camera_token))
        return aggregated

    def forward(self, images, masks=None):
        images = self.input_normalize(images)
        if self.symmetrical:
            images = torch.cat([images, images.flip(dims=[1])], dim=0)
            if masks is not None:
                masks = torch.cat([masks, masks.flip(dims=[1])], dim=0)

        features = self.backbone(images, masks=masks)
        if self.symmetrical and not self.return_symmetrical_logits:
            features = self._aggregate_symmetrical_camera_features(features)
        return self.classifier(raw_feats=features[: self.num_stages])
