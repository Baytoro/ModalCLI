from typing import Any, Dict


def variant_message(ctx: Dict[str, Any], variant: Dict[str, Any], settings: Dict[str, Any]) -> str | None:
    custom_ms = variant.get("custom_ms")
    if not isinstance(custom_ms, (int, float)) or custom_ms <= 0:
        return None

    kernel_iters = int(settings.get("kernel_iters", 0))
    unroll = int(settings.get("unroll", 0))
    num_threads = int(settings.get("num_threads", 0))
    if kernel_iters <= 0 or unroll <= 0 or num_threads <= 0:
        return None

    # Each loop body does:
    # - 8 * fmaf => 8 * 2 FLOPs
    # - 8 * (a +=) + 8 * (b -=) => 16 FLOPs
    # Total = 32 FLOPs per unroll step.
    flops_per_step = (8 * 2) + 16
    flops_per_thread = kernel_iters * unroll * flops_per_step
    total_flops = float(flops_per_thread) * float(num_threads)
    seconds = float(custom_ms) / 1e3
    tflops = total_flops / seconds / 1e12
    return f"tflops={tflops:.2f}"
