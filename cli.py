import argparse
import json
import os
import re
import sys
from glob import glob
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, Tuple

REPO_ROOT = os.path.dirname(__file__)
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "scripts", "settings.json")


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _resolve_entry(test_dir: str, repo_root: str, entry_spec: str | None) -> Tuple[str, str]:
    if not entry_spec:
        return os.path.join(test_dir, "run.py"), "run"

    if ":" not in entry_spec:
        raise ValueError("entry must be in the form 'file.py:function'")
    file_part, func_name = entry_spec.split(":", 1)
    if not file_part.endswith(".py"):
        raise ValueError("entry file must be a .py file")

    if os.path.isabs(file_part):
        return file_part, func_name

    test_candidate = os.path.join(test_dir, file_part)
    repo_candidate = os.path.join(repo_root, file_part)
    if os.path.exists(test_candidate):
        return test_candidate, func_name
    if os.path.exists(repo_candidate):
        return repo_candidate, func_name
    return test_candidate, func_name


def _to_remote_path(local_path: str, test_dir: str, repo_root: str) -> str:
    local_norm = os.path.normcase(os.path.abspath(local_path))
    test_norm = os.path.normcase(os.path.abspath(test_dir))
    repo_norm = os.path.normcase(os.path.abspath(repo_root))

    if local_norm.startswith(test_norm + os.sep) or local_norm == test_norm:
        rel = os.path.relpath(local_path, test_dir)
        return os.path.join("/root/test", rel).replace("\\", "/")

    if local_norm.startswith(repo_norm + os.sep) or local_norm == repo_norm:
        rel = os.path.relpath(local_path, repo_root)
        return os.path.join("/root/project", rel).replace("\\", "/")

    raise ValueError(
        f"entry file must be under test dir or repo root: {local_path}"
    )


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _sanitize_identifier(name: str) -> str:
    sanitized = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not sanitized:
        return "kernel"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _derive_custom_extension_defaults(test_dir: str) -> Dict[str, Any]:
    workload_name = os.path.basename(os.path.normpath(test_dir))
    symbol_base = _sanitize_identifier(workload_name)
    cpp_name = f"{workload_name}.cpp"
    pattern = os.path.join(test_dir, f"{workload_name}_*.cu")
    cu_files = sorted(os.path.basename(path) for path in glob(pattern))
    if not cu_files:
        raise FileNotFoundError(
            f"No CUDA sources found by pattern: {workload_name}_*.cu in {test_dir}"
        )

    extensions = []
    for cu_name in cu_files:
        suffix = cu_name[len(workload_name) + 1 : -3]  # remove "<workload>_" and ".cu"
        variant = _sanitize_identifier(suffix or "default")
        extensions.append(
            {
                "variant": variant,
                "name": f"{symbol_base}_{variant}_ext",
                "cpp": cpp_name,
                "cu": cu_name,
                "function": symbol_base,
                "extra_cflags": ["-O3"],
                "extra_cuda_cflags": ["-O3", "--use_fast_math"],
                "verbose_build": True,
            }
        )

    return {
        "custom_extensions": extensions,
    }


def _finalize_custom_extension_config(config: Dict[str, Any], test_dir: str) -> None:
    derived = _derive_custom_extension_defaults(test_dir)
    config["custom_extensions"] = derived["custom_extensions"]


def _validate_custom_extension_files(test_dir: str, ext_cfgs: list[Dict[str, Any]]) -> None:
    missing = []
    for ext_cfg in ext_cfgs:
        cpp_path = os.path.join(test_dir, str(ext_cfg["cpp"]))
        cu_path = os.path.join(test_dir, str(ext_cfg["cu"]))
        for candidate in (cpp_path, cu_path):
            if not os.path.exists(candidate):
                missing.append(candidate)
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(
            f"Auto-derived extension files not found: {missing_list}"
        )


def _select_custom_extensions(
    ext_cfgs: list[Dict[str, Any]], test_config: Dict[str, Any]
) -> list[Dict[str, Any]]:
    raw_variants = test_config.get("variants") if isinstance(test_config, dict) else None
    if raw_variants is None:
        return ext_cfgs
    if not isinstance(raw_variants, list):
        raise ValueError("config.json field 'variants' must be a list when provided")

    requested = [str(v).strip() for v in raw_variants if str(v).strip()]
    if not requested:
        return ext_cfgs

    available_variants = {str(ext.get("variant", "")) for ext in ext_cfgs}
    available_cu = {str(ext.get("cu", "")) for ext in ext_cfgs}
    requested_set = set(requested)

    filtered = [
        ext
        for ext in ext_cfgs
        if str(ext.get("variant", "")) in requested_set or str(ext.get("cu", "")) in requested_set
    ]
    missing = [
        name
        for name in requested
        if name not in available_variants and name not in available_cu
    ]
    if missing:
        available = ", ".join(sorted(available_variants))
        raise ValueError(
            f"Unknown variants in config.json: {missing}. Available variants: {available}"
        )
    return filtered


def _resolve_gpu(default_gpu: Any, test_config: Dict[str, Any]) -> Any:
    if not isinstance(test_config, dict):
        return default_gpu
    override_gpu = test_config.get("gpu")
    if override_gpu is None:
        return default_gpu
    override_gpu_str = str(override_gpu).strip()
    return override_gpu_str if override_gpu_str else default_gpu


def _resolve_test_dir(path_arg: str | None, target_arg: str | None) -> str:
    if path_arg and target_arg:
        raise ValueError("Use either positional workload or -p/--path, not both")
    raw = path_arg or target_arg
    if not raw:
        raise ValueError("missing test target; use e.g. `python cli.py vector_add`")

    raw = str(raw).strip()
    if not raw:
        raise ValueError("test target is empty")

    is_simple_name = (
        not os.path.isabs(raw)
        and "/" not in raw
        and "\\" not in raw
    )

    candidates = []
    if is_simple_name:
        candidates.append(os.path.join(REPO_ROOT, "workloads", raw))
        candidates.append(os.path.join(REPO_ROOT, "worloads", raw))  # backward compatibility
    candidates.append(raw)

    for candidate in candidates:
        if os.path.isdir(candidate):
            return os.path.abspath(candidate)

    preferred = candidates[0] if candidates else raw
    return os.path.abspath(preferred)


def build_image(base_image: str, image_cfg: Dict[str, Any] | None = None):
    import modal

    cfg = image_cfg or {}
    image = modal.Image.from_registry(base_image)

    env = cfg.get("env", {})
    if env:
        image = image.env(env)

    run_commands = cfg.get("run_commands", [])
    for command in run_commands:
        image = image.run_commands(command)

    apt = cfg.get("apt", [])
    pip = cfg.get("pip", [])

    if apt:
        image = image.apt_install(*apt)
    if pip:
        image = image.pip_install(*pip)
    return image


def remote_runner(
    config_payload: Dict[str, Any],
    test_config_payload: Dict[str, Any],
    entry_file_path: str,
    func_name: str,
):
    return _invoke_entry(
        config_payload=config_payload,
        test_config_payload=test_config_payload,
        entry_file_path=entry_file_path,
        func_name=func_name,
        test_root="/root/test",
        shared_scripts_root="/root/project/scripts",
        tag="remote",
    )


def _invoke_entry(
    config_payload: Dict[str, Any],
    test_config_payload: Dict[str, Any],
    entry_file_path: str,
    func_name: str,
    test_root: str,
    shared_scripts_root: str,
    tag: str,
):
    sys.path.insert(0, test_root)
    if os.path.isdir(shared_scripts_root):
        sys.path.insert(0, shared_scripts_root)
    print(f"[{tag}] cwd={os.getcwd()}")
    print(f"[{tag}] test_root={test_root}")
    print(f"[{tag}] shared_scripts_root={shared_scripts_root}")
    print(f"[{tag}] entry={entry_file_path}:{func_name}")
    print(f"[{tag}] python={sys.version.split()[0]}")

    loader = SourceFileLoader("modalcli_user_module", entry_file_path)
    module = loader.load_module()
    if not hasattr(module, func_name):
        raise AttributeError(f"function '{func_name}' not found in {entry_file_path}")

    func = getattr(module, func_name)
    ctx = {
        "config": config_payload,
        "test_config": test_config_payload,
        "test_dir": test_root,
        "shared_scripts_dir": shared_scripts_root,
        "cwd": os.getcwd(),
        "run_mode": tag,
    }
    print(f"[{tag}] running user function...")
    return func(ctx)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[modalcli] {message}")


def _fmt_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _format_variant_message(raw: Any) -> str:
    text = str(raw).strip()
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.+)$", text)
    if match:
        key = match.group(1).strip()
        value = match.group(2).strip()
        if value:
            return f"{value} {key}"
    return text


def _print_result(result: Dict[str, Any]) -> None:
    if isinstance(result.get("variants"), list):
        variants = result["variants"]
        summary = result.get("summary", {})
        data = result.get("data", {})
        mode = str(result.get("mode", "all"))
        show_accuracy = mode in {"accuracy", "all"}
        show_benchmark = mode in {"benchmark", "all"}
        print("=== ModalCLI Result ===")
        print(f"GPU          : {result.get('gpu', 'unknown')}")
        print(f"Mode         : {mode}")
        if show_accuracy:
            print(
                f"Variants     : {summary.get('passed', 0)}/{summary.get('total', len(variants))} passed"
            )
        else:
            print(f"Variants     : {summary.get('total', len(variants))}")
        if show_benchmark and summary.get("best_variant"):
            print(
                f"Best Variant : {summary['best_variant']} "
                f"({_fmt_float(summary.get('best_custom_ms'), 6)} ms)"
            )
        print("--- Variants ---")
        # Print header
        header_parts = ["Variant".ljust(10)]
        if show_accuracy:
            header_parts.append("Accuracy".ljust(8))
        if show_benchmark:
            header_parts.append("Time (ms)".ljust(10))
        if any(item.get("message") for item in variants):
            header_parts.append("Message")
        print("  ".join(header_parts))
        print("-" * (len("  ".join(header_parts))))
        
        # Print each variant
        for item in variants:
            name = item.get("variant", "unknown")
            row_parts = [str(name).ljust(10)]
            if show_accuracy:
                ok = str(bool(item.get("allclose")))
                row_parts.append(("PASS" if ok == "True" else "FAIL").ljust(8))
            if show_benchmark:
                custom_ms = _fmt_float(item.get("custom_ms"), 6)
                row_parts.append(f"{custom_ms}".rjust(10))
            if item.get("message"):
                row_parts.append(_format_variant_message(item.get("message")))
            print("  ".join(row_parts))
        if isinstance(data, dict) and data:
            print("--- Settings ---")
            for key, value in data.items():
                print(f"{key:<12}: {value}")
        return

    gpu = result.get("gpu", "unknown")
    acc = result.get("accuracy", {})
    metrics = result.get("metrics", {})
    custom_metrics = metrics.get("custom", {}) if isinstance(metrics, dict) else {}
    perf = custom_metrics.get("performance", {}) if isinstance(custom_metrics, dict) else {}
    meta = result.get("meta", {})
    custom_meta = meta.get("custom", {}) if isinstance(meta, dict) else {}
    data = result.get("data", {})

    print("=== ModalCLI Result ===")
    print(f"GPU          : {gpu}")
    print(f"Allclose     : {acc.get('allclose')}")
    if "max_abs_err" in acc:
        print(f"Max Abs Err  : {_fmt_float(acc.get('max_abs_err'), 6)}")
    if "build_s" in custom_metrics:
        print(f"Build Time   : {_fmt_float(custom_metrics.get('build_s'), 3)} s")
    if "custom_ms" in perf:
        print(f"Kernel Time  : {_fmt_float(perf.get('custom_ms'), 6)} ms")
    if "ref_ms" in perf:
        print(f"Ref Time     : {_fmt_float(perf.get('ref_ms'), 6)} ms")
    if "speedup_vs_ref" in perf:
        print(f"Speedup      : {_fmt_float(perf.get('speedup_vs_ref'), 4)}x")

    if custom_meta:
        print("--- Runtime ---")
        for key in ("module_name", "function", "warmup", "iters"):
            if key in custom_meta:
                print(f"{key:<12}: {custom_meta[key]}")

    if isinstance(data, dict) and data:
        print("--- Settings ---")
        for key, value in data.items():
            print(f"{key:<12}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Modal CUDA test runner")
    parser.add_argument("target", nargs="?", help="Workload name (e.g. vector_add) or test folder path")
    parser.add_argument("-p", "--path", help="Path to test folder (legacy option)")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of on Modal cloud",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed runner logs",
    )
    args = parser.parse_args()
    verbose = args.verbose

    try:
        test_dir = _resolve_test_dir(args.path, args.target)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    test_config_path = os.path.join(test_dir, "config.json")
    _log(verbose, f"Test directory: {test_dir}")

    config = _load_config(DEFAULT_CONFIG_PATH)
    _log(verbose, "Loaded scripts/settings.json")
    test_config: Dict[str, Any] = {}
    if os.path.exists(test_config_path):
        test_config = _load_config(test_config_path)
        _log(verbose, f"Loaded test config: {test_config_path}")
    else:
        _log(verbose, f"Test config not found: {test_config_path}; using data.py defaults")
    _finalize_custom_extension_config(config, test_dir)
    ext_cfgs = _select_custom_extensions(config["custom_extensions"], test_config)
    config["custom_extensions"] = ext_cfgs
    _validate_custom_extension_files(test_dir, ext_cfgs)
    _log(verbose, f"Discovered CUDA variants: {len(ext_cfgs)}")
    for ext_cfg in ext_cfgs:
        _log(
            verbose,
            f"Variant={ext_cfg['variant']} cpp={ext_cfg['cpp']} cu={ext_cfg['cu']} function={ext_cfg['function']}",
        )
    entry_spec = config.get("entry")
    entry_file, entry_func = _resolve_entry(test_dir, REPO_ROOT, entry_spec)
    _log(verbose, f"Entry: {entry_file}:{entry_func}")
    shared_scripts_local = os.path.join(REPO_ROOT, "scripts")

    if args.local:
        _log(verbose, "Run mode: local")
        _log(verbose, "Executing entry on local machine")
        try:
            result = _invoke_entry(
                config_payload=config,
                test_config_payload=test_config,
                entry_file_path=entry_file,
                func_name=entry_func,
                test_root=test_dir,
                shared_scripts_root=shared_scripts_local,
                tag="local",
            )
            _print_result(result)
            _log(verbose, "Local run completed")
            return 0
        except Exception as exc:
            message = str(exc)
            if "No module named 'torch'" in message:
                print(
                    "Local run failed because PyTorch is not installed in this environment. "
                    "Install dependencies locally or use Modal cloud mode (without --local).",
                    file=sys.stderr,
                )
                print(f"Details: {exc}", file=sys.stderr)
                return 4
            if "CUDA not available" in message:
                print(
                    "Local run failed because CUDA is unavailable on this machine. "
                    "Use Modal cloud mode (without --local) or run on a CUDA-capable host.",
                    file=sys.stderr,
                )
                print(f"Details: {exc}", file=sys.stderr)
                return 3
            raise

    try:
        import modal
    except Exception as exc:  # pragma: no cover - import error only
        print("Modal is not installed. Run: pip install modal", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return 1

    app_name = config["name"]
    app = modal.App(app_name)

    image_cfg = config["image"]
    base_image = image_cfg["base"]
    image = build_image(base_image, image_cfg)
    gpu = _resolve_gpu(config["gpu"], test_config)
    cpu = config.get("cpu", 2)
    timeout = int(config["timeout"])
    _log(verbose, f"App name: {app_name}")
    _log(verbose, f"GPU: {gpu}")
    _log(verbose, f"CPU: {cpu}")
    _log(verbose, f"Timeout: {timeout}s")
    _log(verbose, f"Base image: {base_image}")
    _log(verbose, f"Image env vars: {len(image_cfg.get('env', {}))}")
    _log(verbose, f"Image run_commands: {len(image_cfg.get('run_commands', []))}")
    _log(verbose, f"Image apt packages: {len(image_cfg.get('apt', []))}")
    _log(verbose, f"Image pip packages: {len(image_cfg.get('pip', []))}")

    image = image.add_local_dir(test_dir, "/root/test")
    _log(verbose, f"Synced local test dir -> /root/test")
    if os.path.isdir(shared_scripts_local):
        image = image.add_local_dir(shared_scripts_local, "/root/project/scripts")
        _log(verbose, "Synced shared scripts dir -> /root/project/scripts")
    entry_file_remote = _to_remote_path(entry_file, test_dir, REPO_ROOT)
    _log(verbose, f"Remote entry path: {entry_file_remote}")

    remote_runner_modal = app.function(
        image=image,
        gpu=gpu,
        cpu=cpu,
        timeout=timeout,
    )(remote_runner)

    _log(verbose, "Starting Modal app run (image build may happen here)...")
    try:
        with app.run():
            _log(verbose, "Invoking remote function...")
            result = remote_runner_modal.remote(
                config,
                test_config,
                entry_file_remote,
                entry_func,
            )
            _print_result(result)
    except Exception as exc:
        if exc.__class__.__name__ == "ConnectionError":
            print(
                "Failed to connect to Modal server. "
                "Please check network/proxy/firewall and Modal authentication.",
                file=sys.stderr,
            )
            print(f"Details: {exc}", file=sys.stderr)
            return 2
        raise
    _log(verbose, "Run completed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
