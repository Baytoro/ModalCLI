from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    vector_cfg = test_cfg.get("vector_add", test_cfg) if isinstance(test_cfg, dict) else {}
    n = int(vector_cfg.get("n", 16_777_216))
    warmup = int(vector_cfg.get("warmup", 20))
    iters = int(vector_cfg.get("iters", 100))

    dtype_name = str(vector_cfg.get("dtype", "float32")).lower()
    if dtype_name != "float32":
        raise ValueError("vector_add currently supports only float32")

    device = torch.device("cuda")
    a = torch.randn(n, dtype=torch.float32, device=device).contiguous()
    b = torch.randn(n, dtype=torch.float32, device=device).contiguous()

    return {
        "inputs": [a, b],
        "n": n,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
    }
