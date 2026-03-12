from typing import Any, Dict


def run(ctx: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    x = data["inputs"][0]
    out = x.sum(dim=1)
    return {
        "output": out,
        "meta": {
            "impl": "torch_sum_dim1",
        },
    }
