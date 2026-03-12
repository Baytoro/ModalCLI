from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    cfg = test_cfg.get("tensor_core", test_cfg) if isinstance(test_cfg, dict) else {}

    warps_per_block = int(cfg.get("warps_per_block", 8))
    kernel_iters = int(cfg.get("kernel_iters", 2000))
    repeat = int(cfg.get("repeat", 16))
    accumulators = int(cfg.get("accumulators", 4))
    warmup = int(cfg.get("warmup", 3))
    iters = int(cfg.get("iters", 8))

    if warps_per_block <= 0:
        raise ValueError("warps_per_block must be positive")
    if kernel_iters <= 0:
        raise ValueError("kernel_iters must be positive")
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if accumulators <= 0:
        raise ValueError("accumulators must be positive")

    block_size = warps_per_block * 32
    if block_size > 1024:
        raise ValueError("warps_per_block is too large for CUDA block size")

    props = torch.cuda.get_device_properties(0)
    default_blocks = int(props.multi_processor_count) * 8
    num_blocks = int(cfg.get("num_blocks", default_blocks))
    if num_blocks <= 0:
        num_blocks = default_blocks
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")

    total_warps = num_blocks * warps_per_block
    out = torch.empty((total_warps,), dtype=torch.float32, device="cuda").contiguous()

    return {
        "inputs": [out, num_blocks, block_size, kernel_iters, repeat, accumulators],
        "warps_per_block": warps_per_block,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "kernel_iters": kernel_iters,
        "repeat": repeat,
        "accumulators": accumulators,
        "total_warps": total_warps,
        "warmup": warmup,
        "iters": iters,
        "dtype": "fp16_input_fp32_accum",
        # Compute-focused benchmark; avoid forcing L2 flush between iterations.
        "flush_l2": False,
    }
