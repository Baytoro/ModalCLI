import time

import torch


def run(ctx):
    try:
        if not torch.cuda.is_available():
            return {"error": "CUDA not available", "ctx": ctx}

        device = torch.device("cuda")
        n = 50_000_000
        a = torch.randn(n, device=device)
        b = torch.randn(n, device=device)

        # Warm-up
        _ = a + b
        torch.cuda.synchronize()

        start = time.perf_counter()
        c = a + b
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        bytes_moved = a.element_size() * a.numel() * 3
        bandwidth_gb_s = bytes_moved / elapsed / 1e9

        return {
            "gpu": torch.cuda.get_device_name(0),
            "n": n,
            "elapsed_s": elapsed,
            "bandwidth_gb_s": bandwidth_gb_s,
            "c_mean": float(c.mean().item()),
        }
    except Exception as exc:
        return {"error": str(exc), "ctx": ctx}
