from typing import Any, Dict


def run(ctx: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    a, b = data["inputs"]
    return {
        "output": a + b,
        "meta": {
            "impl": "torch_add",
        },
    }
