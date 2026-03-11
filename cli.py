import argparse
import json
import os
import sys
from importlib.machinery import SourceFileLoader
from typing import Any, Dict, Tuple

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")


def _load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_entry(test_dir: str, entry_spec: str | None) -> Tuple[str, str]:
    if not entry_spec:
        return os.path.join(test_dir, "run.py"), "run"

    if ":" not in entry_spec:
        raise ValueError("entry must be in the form 'file.py:function'")
    file_part, func_name = entry_spec.split(":", 1)
    if not file_part.endswith(".py"):
        raise ValueError("entry file must be a .py file")
    return os.path.join(test_dir, file_part), func_name


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


def remote_runner(config_payload: Dict[str, Any], entry_file_path: str, func_name: str):
    test_root = "/root/test"
    sys.path.insert(0, test_root)
    print(f"[remote] cwd={os.getcwd()}")
    print(f"[remote] test_root={test_root}")
    print(f"[remote] entry={entry_file_path}:{func_name}")
    print(f"[remote] python={sys.version.split()[0]}")

    loader = SourceFileLoader("modalcli_user_module", entry_file_path)
    module = loader.load_module()
    if not hasattr(module, func_name):
        raise AttributeError(f"function '{func_name}' not found in {entry_file_path}")

    func = getattr(module, func_name)
    ctx = {
        "config": config_payload,
        "test_dir": test_root,
        "cwd": os.getcwd(),
    }
    print("[remote] running user function...")
    return func(ctx)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(f"[modalcli] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Modal CUDA test runner")
    parser.add_argument("-p", "--path", required=True, help="Path to test folder")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print final result and errors",
    )
    args = parser.parse_args()
    verbose = not args.quiet

    test_dir = os.path.abspath(args.path)
    config_path = os.path.join(test_dir, "config.json")
    _log(verbose, f"Test directory: {test_dir}")
    _log(verbose, f"Config path: {config_path}")

    config = _load_config(DEFAULT_CONFIG_PATH)
    _log(verbose, "Loaded default_config.json")
    if os.path.exists(config_path):
        user_config = _load_config(config_path)
        config = _merge_config(config, user_config)
        _log(verbose, "Loaded and merged config.json")
    else:
        print(
            f"config.json not found in {test_dir}; using {DEFAULT_CONFIG_PATH}",
            file=sys.stderr,
        )
    entry_spec = config.get("entry")
    entry_file, entry_func = _resolve_entry(test_dir, entry_spec)
    _log(verbose, f"Entry: {entry_file}:{entry_func}")

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
    gpu = config["gpu"]
    timeout = int(config["timeout"])
    _log(verbose, f"App name: {app_name}")
    _log(verbose, f"GPU: {gpu}")
    _log(verbose, f"Timeout: {timeout}s")
    _log(verbose, f"Base image: {base_image}")
    _log(verbose, f"Image env vars: {len(image_cfg.get('env', {}))}")
    _log(verbose, f"Image run_commands: {len(image_cfg.get('run_commands', []))}")
    _log(verbose, f"Image apt packages: {len(image_cfg.get('apt', []))}")
    _log(verbose, f"Image pip packages: {len(image_cfg.get('pip', []))}")

    image = image.add_local_dir(test_dir, "/root/test")
    _log(verbose, f"Synced local test dir -> /root/test")
    rel_entry = os.path.relpath(entry_file, test_dir)
    entry_file_remote = os.path.join("/root/test", rel_entry).replace("\\", "/")
    _log(verbose, f"Remote entry path: {entry_file_remote}")

    remote_runner_modal = app.function(
        image=image,
        gpu=gpu,
        timeout=timeout,
    )(remote_runner)

    _log(verbose, "Starting Modal app run (image build may happen here)...")
    with app.run():
        _log(verbose, "Invoking remote function...")
        result = remote_runner_modal.remote(config, entry_file_remote, entry_func)
        print("Result:", result)
    _log(verbose, "Run completed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
