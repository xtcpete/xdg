import torch


_BF16_MIXED = "bf16-mixed"
_FP16_MIXED = "16-mixed"
_FP32_TRUE = "32-true"


def _cuda_device_index(device=None):
    if device is None:
        return torch.cuda.current_device()

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda":
        return None
    if cuda_device.index is None:
        return torch.cuda.current_device()
    return cuda_device.index


def cuda_bf16_supported(device=None) -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        device_index = _cuda_device_index(device)
    except RuntimeError:
        return False
    if device_index is None:
        return False

    try:
        major, _minor = torch.cuda.get_device_capability(device_index)
    except (AssertionError, RuntimeError):
        return False
    return major >= 8


def resolve_mixed_precision(precision="auto", device=None):
    if precision is None:
        precision = "auto"
    value = str(precision).lower()

    if value == "auto":
        return _BF16_MIXED if cuda_bf16_supported(device=device) else _FP16_MIXED
    if value in {"bf16", "bfloat16", _BF16_MIXED}:
        return _BF16_MIXED if cuda_bf16_supported(device=device) else _FP16_MIXED
    if value in {"fp16", "float16", "16", "mixed", _FP16_MIXED}:
        return _FP16_MIXED
    if value in {"fp32", "float32", "32", _FP32_TRUE, "32-true"}:
        return _FP32_TRUE
    return precision


def resolve_autocast_dtype(precision="auto", device=None):
    if device is not None and torch.device(device).type != "cuda":
        return None
    if precision is None:
        precision = "auto"
    value = str(precision).lower()

    if value in {"fp32", "float32", "32", _FP32_TRUE, "32-true"}:
        return None
    if value in {"auto", "bf16", "bfloat16", _BF16_MIXED} and cuda_bf16_supported(device=device):
        return torch.bfloat16
    return torch.float16
