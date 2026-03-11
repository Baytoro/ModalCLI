import hashlib
import json
import os
import subprocess
import time
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, Tuple

import torch
import torch.utils.cpp_extension as cpp_ext
from torch.utils.cpp_extension import load_inline


def _load_module(module_path: str, module_name: str):
    loader = SourceFileLoader(module_name, module_path)
    return loader.load_module()


def _read_source(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_cmd_output(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as exc:
        return f"<failed to run {' '.join(cmd)}: {exc}>"
    return (completed.stdout or "").strip() or "<no output>"


def _collect_build_diagnostics() -> dict:
    info = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "<unset>"),
        "nvcc_version": _read_cmd_output(["nvcc", "--version"]),
        "ninja_version": _read_cmd_output(["ninja", "--version"]),
        "cxx_version": _read_cmd_output(["c++", "--version"]),
    }
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_capability"] = f"{cap[0]}.{cap[1]}"
    return info


def _capture_ninja_output(module_name: str) -> dict:
    try:
        build_dir = cpp_ext._get_build_directory(module_name, verbose=False)
    except Exception as exc:
        return {"build_dir_error": str(exc)}

    try:
        completed = subprocess.run(
            ["ninja", "-v"],
            cwd=build_dir,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = (completed.stdout or "").strip() or "<no output>"
        code = completed.returncode
    except Exception as exc:
        return {"build_dir": build_dir, "ninja_run_error": str(exc)}

    return {"build_dir": build_dir, "ninja_exit_code": code, "ninja_output": output}


def _time_kernel(fn, warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    if iters <= 0:
        return 0.0

    total_ms = 0.0
    l2_flush_bytes = 128 * 1024 * 1024
    l2_flush_buf = torch.empty(l2_flush_bytes, dtype=torch.uint8, device="cuda")

    # Flush once before measured iters to reduce warmup cache carry-over.
    l2_flush_buf.add_(1)
    torch.cuda.synchronize()

    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        total_ms += float(start.elapsed_time(end))

        # Clear L2 cache between measured iterations.
        l2_flush_buf.add_(1)
        torch.cuda.synchronize()

    return total_ms / iters


def _extract_output(payload: Any) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    if isinstance(payload, dict) and "output" in payload:
        return payload["output"], payload.get("metrics", {}), payload.get("meta", {})
    return payload, {}, {}


def _compare_outputs(custom_out: Any, ref_out: Any) -> Dict[str, Any]:
    if isinstance(custom_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):
        if custom_out.shape != ref_out.shape:
            return {
                "allclose": False,
                "reason": f"shape mismatch: {tuple(custom_out.shape)} vs {tuple(ref_out.shape)}",
            }
        atol = 1e-6
        rtol = 1e-5
        max_abs_err = float((custom_out - ref_out).abs().max().item())
        return {
            "allclose": bool(torch.allclose(custom_out, ref_out, atol=atol, rtol=rtol)),
            "max_abs_err": max_abs_err,
            "atol": atol,
            "rtol": rtol,
        }
    return {"allclose": bool(custom_out == ref_out), "max_abs_err": None}


def _normalize_mode(test_config: Dict[str, Any]) -> str:
    raw = None
    if isinstance(test_config, dict):
        raw = test_config.get("mode")
    if raw is None or str(raw).strip() == "" or str(raw).lower() == "all":
        return "all"
    mode = str(raw).lower()
    if mode not in {"accuracy", "benchmark"}:
        raise ValueError("config.json field 'mode' must be one of: accuracy, benchmark, all")
    return mode


def _build_extension(test_dir: str, ext_cfg: Dict[str, Any]):
    cpp_name = str(ext_cfg["cpp"])
    cu_name = str(ext_cfg["cu"])
    cpp_path = os.path.join(test_dir, cpp_name)
    cu_path = os.path.join(test_dir, cu_name)
    cpp_src = _read_source(cpp_path)
    cu_src = _read_source(cu_path)

    base_name = str(ext_cfg.get("name", "custom_ext"))
    source_fingerprint = hashlib.sha1((cpp_src + cu_src).encode("utf-8")).hexdigest()[:8]
    module_name = f"{base_name}_{source_fingerprint}"

    extra_cflags = list(ext_cfg.get("extra_cflags", ["-O3"]))
    extra_cuda_cflags = list(ext_cfg.get("extra_cuda_cflags", ["-O3", "--use_fast_math"]))

    build_start = time.perf_counter()
    try:
        ext = load_inline(
            name=module_name,
            cpp_sources=[cpp_src],
            cuda_sources=[cu_src],
            functions=None,
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda_cflags,
            with_cuda=True,
            verbose=bool(ext_cfg.get("verbose_build", True)),
        )
    except Exception as exc:
        details = {
            "module_name": module_name,
            "diagnostics": _collect_build_diagnostics(),
            "ninja_replay": _capture_ninja_output(module_name),
            "original_error": str(exc),
        }
        raise RuntimeError(
            "Failed to build extension: " + json.dumps(details, ensure_ascii=False)
        ) from exc

    build_s = time.perf_counter() - build_start
    return ext, module_name, build_s


def run(ctx: Dict[str, Any]) -> Dict[str, Any]:
    test_dir = ctx["test_dir"]
    test_config = ctx.get("test_config", {})
    mode = _normalize_mode(test_config if isinstance(test_config, dict) else {})
    need_accuracy = mode in {"accuracy", "all"}
    need_benchmark = mode in {"benchmark", "all"}

    data_path = os.path.join(test_dir, "data.py")
    ref_path = os.path.join(test_dir, "ref.py")
    required_paths = [data_path]
    if need_accuracy:
        required_paths.append(ref_path)
    for required_path in required_paths:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"missing required file: {required_path}")

    data_mod = _load_module(data_path, "modalcli_data")
    ref_mod = _load_module(ref_path, "modalcli_ref") if need_accuracy else None
    if not hasattr(data_mod, "data"):
        raise AttributeError("data.py must define data(ctx)")
    if need_accuracy and not hasattr(ref_mod, "run"):
        raise AttributeError("ref.py must define run(ctx, data)")

    cfg = ctx.get("config", {})
    ext_cfgs = cfg.get("custom_extensions")
    if not isinstance(ext_cfgs, list) or not ext_cfgs:
        raise ValueError("config.custom_extensions is required and must be non-empty")

    data_payload = data_mod.data(ctx)
    if not isinstance(data_payload, dict):
        raise TypeError("data(ctx) must return a dict")
    inputs = data_payload.get("inputs")
    if not isinstance(inputs, (list, tuple)):
        raise ValueError("data(ctx) must provide 'inputs' as list/tuple")
    warmup = int(data_payload.get("warmup", 20))
    iters = int(data_payload.get("iters", 100))

    ref_out = None
    ref_metrics: Dict[str, Any] = {}
    ref_meta: Dict[str, Any] = {}
    if need_accuracy:
        ref_payload = ref_mod.run(ctx, data_payload)
        ref_out, ref_metrics, ref_meta = _extract_output(ref_payload)

    variants: list[Dict[str, Any]] = []
    for ext_cfg in ext_cfgs:
        if not isinstance(ext_cfg, dict):
            raise TypeError("each custom_extensions item must be an object")
        func_name = str(ext_cfg.get("function", "vector_add"))
        variant = str(ext_cfg.get("variant", os.path.basename(str(ext_cfg.get("cu", "")))))

        ext, module_name, build_s = _build_extension(test_dir, ext_cfg)
        if not hasattr(ext, func_name):
            raise AttributeError(f"compiled extension has no function '{func_name}'")
        custom_fn = getattr(ext, func_name)

        row: Dict[str, Any] = {
            "variant": variant,
            "cu": ext_cfg.get("cu"),
            "function": func_name,
            "module_name": module_name,
            "build_s": build_s,
        }

        custom_out = None
        if need_accuracy:
            custom_out = custom_fn(*inputs)
            accuracy = _compare_outputs(custom_out, ref_out)
            row["allclose"] = accuracy.get("allclose")
            row["max_abs_err"] = accuracy.get("max_abs_err")

        if need_benchmark:
            custom_ms = _time_kernel(lambda: custom_fn(*inputs), warmup=warmup, iters=iters)
            row["custom_ms"] = custom_ms

        variants.append(row)

    passed = [v for v in variants if bool(v.get("allclose"))] if need_accuracy else []
    best_variant = None
    best_custom_ms = None
    if need_benchmark and variants:
        bench_variants = [v for v in variants if "custom_ms" in v]
        if bench_variants:
            best = min(bench_variants, key=lambda v: float(v.get("custom_ms", float("inf"))))
            best_variant = best.get("variant")
            best_custom_ms = best.get("custom_ms")

    result: Dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "mode": mode,
        "summary": {
            "total": len(variants),
            "passed": len(passed) if need_accuracy else None,
            "failed": (len(variants) - len(passed)) if need_accuracy else None,
            "best_variant": best_variant,
            "best_custom_ms": best_custom_ms,
        },
        "variants": variants,
        "ref": {
            "metrics": ref_metrics if isinstance(ref_metrics, dict) else {},
            "meta": ref_meta if isinstance(ref_meta, dict) else {},
        },
    }
    if isinstance(data_payload, dict):
        result["data"] = {
            k: v for k, v in data_payload.items() if isinstance(v, (int, float, str, bool))
        }
    return result
