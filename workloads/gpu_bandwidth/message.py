from typing import Any, Dict


def variant_message(ctx: Dict[str, Any], variant: Dict[str, Any], settings: Dict[str, Any]) -> str | None:
    custom_ms = variant.get("custom_ms")
    if not isinstance(custom_ms, (int, float)) or custom_ms <= 0:
        return None

    n = int(settings.get("n", 0))
    copy_iters = int(settings.get("copy_iters", settings.get("kernel_iters", 0)))
    if n <= 0 or copy_iters <= 0:
        return None

    # Copy path: read + write float32 per element, repeated copy_iters times.
    bytes_moved = float(n) * 4.0 * 2.0 * float(copy_iters)
    gbps = bytes_moved / (float(custom_ms) / 1e3) / 1e9
    return f"{gbps:.2f} GB/s"
