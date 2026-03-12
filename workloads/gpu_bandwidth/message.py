from typing import Any, Dict


def variant_message(ctx: Dict[str, Any], variant: Dict[str, Any], settings: Dict[str, Any]) -> str | None:
    custom_ms = variant.get("custom_ms")
    if not isinstance(custom_ms, (int, float)) or custom_ms <= 0:
        return None

    n = int(settings.get("n", 0))
    kernel_iters = int(settings.get("kernel_iters", 0))
    if n <= 0 or kernel_iters <= 0:
        return None

    # Copy path: read + write float32 per element, repeated kernel_iters times.
    bytes_moved = float(n) * 4.0 * 2.0 * float(kernel_iters)
    gbps = bytes_moved / (float(custom_ms) / 1e3) / 1e9
    return f"{gbps:.2f} GB/s"
