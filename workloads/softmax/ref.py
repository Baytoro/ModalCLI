from typing import Any, Dict

import torch


def run(ctx: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    x = data["inputs"][0]
    x_max = x.max()
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum()
    result = x_exp / x_sum
    return {
        "output": result,
        "meta": {
            "impl": "torch_softmax",
        },
    }