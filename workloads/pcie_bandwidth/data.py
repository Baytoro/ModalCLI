from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    cfg = test_cfg.get("pcie_bandwidth", test_cfg) if isinstance(test_cfg, dict) else {}

    bytes_mb = int(cfg.get("bytes_mb", 256))
    transfer_iters = int(cfg.get("transfer_iters", cfg.get("kernel_iters", 20)))
    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iters", 20))

    if bytes_mb <= 0:
        raise ValueError("bytes_mb must be positive")
    if transfer_iters <= 0:
        raise ValueError("transfer_iters must be positive")

    nbytes = bytes_mb * 1024 * 1024
    n = nbytes // 4
    if n <= 0:
        raise ValueError("bytes_mb too small")

    host_buf = torch.empty((n,), dtype=torch.float32, pin_memory=True).contiguous()
    host_buf.uniform_(-1.0, 1.0)
    device_buf = torch.empty((n,), dtype=torch.float32, device="cuda").contiguous()
    device_buf.uniform_(-1.0, 1.0)

    return {
        "inputs": [host_buf, device_buf, transfer_iters, nbytes],
        "bytes_mb": bytes_mb,
        "nbytes": nbytes,
        "transfer_iters": transfer_iters,
        # Keep legacy field for backward-compatible message/workflow parsing.
        "kernel_iters": transfer_iters,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
        "flush_l2": False,
    }
