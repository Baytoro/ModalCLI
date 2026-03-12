import json
import os
from typing import Any, Dict


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "unknown"
    kib = 1024
    mib = 1024 * 1024
    if n >= mib:
        return f"{n / mib:.2f} MiB"
    return f"{n / kib:.0f} KiB"


def variant_message(ctx: Dict[str, Any], variant: Dict[str, Any], settings: Dict[str, Any]) -> str | None:
    report_path = str(settings.get("report_path", "")).strip()
    if not report_path or not os.path.exists(report_path):
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except Exception:
        return None

    l1_bytes = int(report.get("estimated_l1_bytes", 0))
    l2_bytes = int(report.get("estimated_l2_bytes", 0))
    if l1_bytes <= 0 and l2_bytes <= 0:
        return None
    return f"L1~{_fmt_bytes(l1_bytes)} L2~{_fmt_bytes(l2_bytes)}"
