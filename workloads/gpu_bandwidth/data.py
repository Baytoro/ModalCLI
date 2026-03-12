from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    cfg = test_cfg.get("gpu_bandwidth", test_cfg) if isinstance(test_cfg, dict) else {}

    n = int(cfg.get("n", 67_108_864))
    copy_iters = int(cfg.get("copy_iters", cfg.get("kernel_iters", 20)))
    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iters", 20))

    if n <= 0:
        raise ValueError("n must be positive")
    if copy_iters <= 0:
        raise ValueError("copy_iters must be positive")

    src = torch.randn((n,), dtype=torch.float32, device="cuda").contiguous()

    return {
        "inputs": [src, copy_iters],
        "n": n,
        "copy_iters": copy_iters,
        # Keep legacy field for backward-compatible message/workflow parsing.
        "kernel_iters": copy_iters,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
        # Keep default runner L2 flush behavior for a streaming memory benchmark.
        "flush_l2": True,
    }
