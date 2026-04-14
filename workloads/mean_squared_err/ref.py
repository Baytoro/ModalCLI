from typing import Any, Dict


def run(ctx: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    predictions, targets = data["inputs"]
    sq_errors = (predictions - targets) ** 2
    mse = sq_errors.mean()
    return {
        "output": mse,
        "meta": {
            "impl": "torch_mse",
        },
    }
