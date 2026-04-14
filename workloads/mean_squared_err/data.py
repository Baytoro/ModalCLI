from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    mse_cfg = test_cfg.get("mean_squared_err", test_cfg) if isinstance(test_cfg, dict) else {}
    n = int(mse_cfg.get("n", 16_777_216))
    warmup = int(mse_cfg.get("warmup", 20))
    iters = int(mse_cfg.get("iters", 100))

    dtype_name = str(mse_cfg.get("dtype", "float32")).lower()
    if dtype_name != "float32":
        raise ValueError("mean_squared_err currently supports only float32")

    device = torch.device("cuda")
    predictions = torch.randn(n, dtype=torch.float32, device=device).contiguous()
    targets = torch.randn(n, dtype=torch.float32, device=device).contiguous()

    return {
        "inputs": [predictions, targets],
        "n": n,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
    }
