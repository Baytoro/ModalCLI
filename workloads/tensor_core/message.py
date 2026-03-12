from typing import Any, Dict


def variant_message(ctx: Dict[str, Any], variant: Dict[str, Any], settings: Dict[str, Any]) -> str | None:
    custom_ms = variant.get("custom_ms")
    if not isinstance(custom_ms, (int, float)) or custom_ms <= 0:
        return None

    kernel_iters = int(settings.get("kernel_iters", 0))
    repeat = int(settings.get("repeat", 0))
    total_warps = int(settings.get("total_warps", 0))
    accumulators = int(settings.get("accumulators", 0))
    if kernel_iters <= 0 or repeat <= 0 or total_warps <= 0 or accumulators <= 0:
        return None

    # m16n16k16: 16 * 16 * 16 FMAs per mma_sync, 1 FMA = 2 FLOPs.
    flops_per_mma = 16 * 16 * 16 * 2
    total_mma = kernel_iters * repeat * accumulators * total_warps
    total_flops = float(total_mma) * float(flops_per_mma)
    seconds = float(custom_ms) / 1e3
    tflops = total_flops / seconds / 1e12
    return f"tflops={tflops:.2f}"
