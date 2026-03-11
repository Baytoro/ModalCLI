## 项目目标
- 这是一个用于快速验证 CUDA 原型想法的工具，通过 Modal 在远程 GPU 上运行测试，避免本地必须有 GPU。
- 典型用途包括：带宽测试、简单 kernel 的性能验证等。

## 期望形态
- 统一入口：`python cli.py -p /path/to/test/folder`
- `/path/to/test/folder` 内包含：
  - 测试代码（入口默认为 `run.py:run`）
  - 可选 `config.json`（用于覆盖默认配置）

## 配置行为
- 仓库根目录提供 `default_config.json` 作为默认配置来源。
- 运行时先读取 `default_config.json`，再用测试目录下的 `config.json` 覆盖（深度合并）。
- 配置项至少包含：
  - `name`
  - `entry`
  - `gpu`
  - `timeout`
  - `image`（如 `base`、`env`、`run_commands`、`apt`、`pip`）

## 示例约束
- 仓库内的 `examples/*` 可以不提供 `config.json`，此时自动使用 `default_config.json`。

## 提交规范
- Git commit message 采用：`[TYPE] Change`
- `TYPE` 建议使用：`FEAT`、`FIX`、`CHORE`、`REFACTOR`、`DOCS`、`TEST`
- 示例：`[CHORE] Change`

## Modal 参考
- 开发相关的 Modal App 时需要参考官方文档：
  - `https://modal.com/docs/guide`
  - `https://modal.com/docs/reference`
  - `https://modal.com/docs/examples`
- 上述链接的子目录及其子目录可以无限查阅。
