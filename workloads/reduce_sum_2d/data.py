from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    cfg = test_cfg.get("reduce_sum_2d", test_cfg) if isinstance(test_cfg, dict) else {}

    rows = int(cfg.get("rows", 4096))
    cols = int(cfg.get("cols", 4096))
    block_size = int(cfg.get("block_size", 256))
    warmup = int(cfg.get("warmup", 10))
    iters = int(cfg.get("iters", 50))

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if block_size <= 0 or block_size > 1024:
        raise ValueError("block_size must be in (0, 1024]")

    x = torch.ones((rows, cols), dtype=torch.float32, device="cuda").contiguous()

    return {
        "inputs": [x, block_size],
        "rows": rows,
        "cols": cols,
        "n": rows * cols,
        "block_size": block_size,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
        "flush_l2": True,
    }
