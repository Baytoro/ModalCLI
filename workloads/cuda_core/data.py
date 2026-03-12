from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    cfg = test_cfg.get("cuda_core", test_cfg) if isinstance(test_cfg, dict) else {}

    block_size = int(cfg.get("block_size", 256))
    num_blocks = int(cfg.get("num_blocks", 4096))
    kernel_iters = int(cfg.get("kernel_iters", 20000))
    unroll = int(cfg.get("unroll", 16))
    warmup = int(cfg.get("warmup", 10))
    iters = int(cfg.get("iters", 10))

    if block_size <= 0 or num_blocks <= 0:
        raise ValueError("block_size and num_blocks must be positive")
    if kernel_iters <= 0:
        raise ValueError("kernel_iters must be positive")
    if unroll <= 0:
        raise ValueError("unroll must be positive")

    device = torch.device("cuda")
    num_threads = block_size * num_blocks
    out = torch.empty((num_threads,), dtype=torch.float32, device=device).contiguous()

    # inputs are forwarded to extension function: cuda_core(out, kernel_iters, unroll)
    return {
        "inputs": [out, kernel_iters, unroll],
        "block_size": block_size,
        "num_blocks": num_blocks,
        "kernel_iters": kernel_iters,
        "unroll": unroll,
        "num_threads": num_threads,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
        # this workload is compute-only; skip L2 flush between timed iters.
        "flush_l2": False,
    }
