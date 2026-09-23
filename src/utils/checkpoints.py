"""Resolve local checkpoints or the released XDG weights on Hugging Face."""

from pathlib import Path
from typing import Optional, Union

from huggingface_hub import hf_hub_download


DEFAULT_CHECKPOINT = Path("weights/xdg.pth")
HF_REPO_ID = "xtcpete/xdg"
# Pin the release matching configs/model_configs/xdg.yaml.
HF_REVISION = "0d7703d9da5afb6b7a9d0c7178f4a5ee8e83b388"
CHECKPOINT_HELP = (
    "Local checkpoint path. If omitted, use weights/xdg.pth when present, "
    "otherwise download the released weights from xtcpete/xdg on Hugging Face."
)


def resolve_checkpoint(checkpoint: Optional[Union[str, Path]] = None) -> str:
    """Prefer local weights; cache the Hub release when no path was supplied.

    Explicit paths must exist so a typo cannot silently select different weights.
    Hugging Face handles caching and honors HF_HOME and HF_HUB_OFFLINE.
    """
    path = Path(checkpoint).expanduser() if checkpoint is not None else DEFAULT_CHECKPOINT
    if path.is_file():
        return str(path)
    if checkpoint is not None or path.exists():
        raise FileNotFoundError(f"Checkpoint is not a file: {path}")

    print(f"Loading released XDG weights from {HF_REPO_ID} (downloaded if not cached).")
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="xdg.pth",
        revision=HF_REVISION,
    )
