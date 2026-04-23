from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    hist_cfg = test_cfg.get("histogram", test_cfg) if isinstance(test_cfg, dict) else {}

    n = int(hist_cfg.get("n", 8_388_608))
    num_bins = int(hist_cfg.get("num_bins", 256))
    warmup = int(hist_cfg.get("warmup", 10))
    iters = int(hist_cfg.get("iters", 50))

    if n <= 0:
        raise ValueError("n must be positive")
    if num_bins <= 0 or num_bins > 1024:
        raise ValueError("num_bins must be in (1, 1024]")

    device = torch.device("cuda")
    x = torch.randint(0, num_bins, (n,), dtype=torch.int32, device=device).contiguous()

    return {
        "inputs": [x, num_bins],
        "n": n,
        "num_bins": num_bins,
        "warmup": warmup,
        "iters": iters,
        "dtype": "int32",
    }