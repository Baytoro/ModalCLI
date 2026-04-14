# ModalCLI

CUDA 原型验证工具，通过 Modal 在远程 GPU 上运行测试，无需本地 GPU。

## 快速开始

```bash
# 安装依赖
pip install modal

# 运行测试
python cli.py <workload_name>
python cli.py -p /path/to/test/folder

# 详细日志
python cli.py --verbose vector_add
```

## 目录结构

```
ModalCLI/
├── cli.py              # 主入口
├── scripts/
│   ├── settings.json   # 全局配置 (GPU, image, timeout)
│   └── run.py          # 远程执行器 (编译、测试、对比)
├── workloads/          # 测试 workload
│   ├── vector_add/
│   ├── reduce_sum/
│   ├── reduce_sum_2d/
│   └── mean_squared_err/
└── update/             # 变更记录
```

## Workload 结构

每个 workload 目录包含：

| 文件 | 说明 | 必需 |
|------|------|------|
| `config.json` | 测试参数 (n, warmup, iters, mode, variants) | 可选 |
| `data.py` | `data(ctx)` → inputs, warmup, iters | 是 |
| `ref.py` | `run(ctx, data)` → 参考实现 | mode=accuracy/all 时需 |
| `message.py` | `variant_message(ctx, variant, settings)` → 额外输出 | 可选 |
| `notes.md` | 问题定义、优化日志、实验记录 | 建议 |
| `<name>.cpp` | PyBind11 绑定 | 是 |
| `<name>_*.cu` | CUDA kernel 变体 | 是 |

## 配置说明

### 全局配置 (scripts/settings.json)

```json
{
  "name": "modalcli-app",
  "entry": "scripts/run.py:run",
  "gpu": "H100",
  "cpu": 8,
  "timeout": 600,
  "image": {
    "base": "pytorch/pytorch:2.9.1-cuda13.0-cudnn9-devel",
    "apt": ["build-essential"],
    "pip": ["ninja"]
  }
}
```

### Workload 配置 (config.json)

```json
{
  "n": 8388608,
  "warmup": 10,
  "iters": 50,
  "gpu": "A100",
  "mode": "all",
  "variants": []
}
```

- `mode`: `accuracy` (只检查正确性), `benchmark` (只测性能), `all` (两者都测)
- `variants`: 空列表测试全部变体，或指定变体名/文件名

## 当前 workload

| 名称 | 说明 |
|------|------|
| [vector_add]() | 向量加法 (base, float4 变体) |
| [reduce_sum]() | 1D 归约求和 (base, cub 变体) |
| [reduce_sum_2d]() | 2D 按行归约求和 (base, cub 变体) |
| [mean_squared_err](https://leetgpu.com/challenges/mean-squared-error) | 均方误差计算 |

## 提交规范

- Commit message 格式：`[TYPE] <简要描述>`
- TYPE: `FEAT`, `FIX`, `CHORE`, `REFACTOR`, `DOCS`, `TEST`
- 提交前需更新 `update/YYYY-MM-DD.md`

## 依赖

- Python 3.10+
- Modal
- PyTorch (远程容器内)
