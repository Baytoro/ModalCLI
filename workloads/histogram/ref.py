from typing import Any, Dict

import torch


def run(ctx: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    x, num_bins = data["inputs"]
    result = torch.bincount(x, minlength=num_bins).to(torch.int32)
    return {
        "output": result,
        "meta": {
            "impl": "torch_bincount",
        },
    }