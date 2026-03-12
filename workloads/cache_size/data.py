from typing import Any, Dict

import torch


def data(ctx: Dict[str, Any]) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    test_cfg = ctx.get("test_config", {})
    cfg = test_cfg.get("cache_size", test_cfg) if isinstance(test_cfg, dict) else {}

    max_ws_mb = int(cfg.get("max_ws_mb", 128))
    probe_iters = int(cfg.get("probe_iters", 2048))
    repeat = int(cfg.get("repeat", 8))
    warmup = int(cfg.get("warmup", 0))
    iters = int(cfg.get("iters", 1))

    if max_ws_mb <= 0:
        raise ValueError("max_ws_mb must be positive")
    if probe_iters <= 0 or repeat <= 0:
        raise ValueError("probe_iters and repeat must be positive")

    max_ws_bytes = max_ws_mb * 1024 * 1024
    max_elems = max_ws_bytes // 4
    if max_elems < 1024:
        raise ValueError("max_ws_mb is too small")

    # Sweep powers of two: 4 KiB ... max_ws_bytes
    sizes = []
    ws = 4 * 1024
    while ws <= max_ws_bytes:
        sizes.append(ws)
        ws *= 2
    sizes_bytes = torch.tensor(sizes, dtype=torch.int64)

    buf = torch.randn((max_elems,), dtype=torch.float32, device="cuda").contiguous()
    report_path = "/tmp/modalcli_cache_size_report.json"

    return {
        "inputs": [buf, sizes_bytes, probe_iters, repeat, report_path],
        "max_ws_mb": max_ws_mb,
        "probe_iters": probe_iters,
        "repeat": repeat,
        "sizes_count": len(sizes),
        "report_path": report_path,
        "warmup": warmup,
        "iters": iters,
        "dtype": "float32",
        # This benchmark intentionally sweeps locality; do not force cache flush from runner.
        "flush_l2": False,
    }
