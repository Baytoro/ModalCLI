from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    softmax_cfg = test_cfg.get("softmax", test_cfg) if isinstance(test_cfg, dict) else {}
    n = int(softmax_cfg.get("n", 8_388_608))
    warmup = int(softmax_cfg.get("warmup", 10))
    iters = int(softmax_cfg.get("iters", 50))

    device = torch.device("cuda")
    x = torch.randn(n, dtype=torch.float32, device=device).contiguous()

    return {
        "inputs": [x],
        "n": n,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
    }