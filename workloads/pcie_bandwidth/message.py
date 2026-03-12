from typing import Any, Dict


def variant_message(ctx: Dict[str, Any], variant: Dict[str, Any], settings: Dict[str, Any]) -> str | None:
    custom_ms = variant.get("custom_ms")
    if not isinstance(custom_ms, (int, float)) or custom_ms <= 0:
        return None

    nbytes = int(settings.get("nbytes", 0))
    transfer_iters = int(settings.get("transfer_iters", settings.get("kernel_iters", 0)))
    if nbytes <= 0 or transfer_iters <= 0:
        return None

    bytes_moved = float(nbytes) * float(transfer_iters)
    gbps = bytes_moved / (float(custom_ms) / 1e3) / 1e9
    return f"{gbps:.2f} GB/s"
