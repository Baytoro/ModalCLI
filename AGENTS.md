## 项目目标
- 这是一个用于快速验证 CUDA 原型想法的工具，通过 Modal 在远程 GPU 上运行测试，避免本地必须有 GPU。
- 典型用途包括：带宽测试、简单 kernel 的性能验证等。

## 期望形态
- 统一入口：`python cli.py -p /path/to/test/folder`
- `/path/to/test/folder` 内包含：
  - 测试代码（统一入口为 `scripts/run.py:run`）
  - `data.py`（提供 `data(ctx)`）
  - `ref.py`（提供 `run(ctx, data)`，`mode=accuracy/all` 时需要）
  - 可选 `message.py`（提供 `variant_message(ctx, variant, settings)`，用于每个 variant 的额外输出）
  - `notes.md`（问题定义、优化日志、实验记录）
  - 自动推导命名的 CUDA 源码文件：
    - `<workload>.cpp`
    - `<workload>_*.cu`（同一 workload 可有多个变体，全部会被测试）

## 配置行为
- 仓库使用 `scripts/settings.json` 作为默认配置来源。
- 测试目录下 `config.json` 主要用于测试参数（如 `n`、`warmup`、`iters`、`dtype`）。
- `config.json` 可选字段 `gpu` 可覆盖 `scripts/settings.json` 里的默认 GPU，其它运行配置不覆盖。
- 配置项至少包含：
  - `name`
  - `entry`
  - `gpu`
  - `timeout`
  - `image`（如 `base`、`env`、`run_commands`、`apt`、`pip`）

## 示例约束
- 仓库内的 `workloads/*` 可以有可选 `config.json`，用于测试参数输入。
- `config.json` 可选字段 `variants`：
  - 不写或 `[]`：测试全部 `<workload>_*.cu`
  - 写具体列表：仅测试指定变体（可写 `variant` 名或 `.cu` 文件名）
- `config.json` 可选字段 `mode`：
  - `accuracy`：只检查正确性（需要 `ref.py`）
  - `benchmark`：只测性能（不需要 `ref.py`）
  - 不写或 `all`：正确性和性能都测试
- 若不存在 `message.py`，`Variants` 中不打印额外信息。

## 提交规范
- Git commit message 采用：`[TYPE] <简要描述>`
- `TYPE` 建议使用：`FEAT`、`FIX`、`CHORE`、`REFACTOR`、`DOCS`、`TEST`
- 示例：`[CHORE] Sync update notes`
- **每次提交前必须先整理当天 `update/YYYY-MM-DD.md`，未整理不得提交**。
- 整理标准（精简 + 颗粒度）：
  - 仅保留本次提交相关改动，删除重复和过程性描述。
  - 每条记录必须写清：改了什么（文件/功能）、为什么改、结果（是否验证通过）。
  - 优先使用短句条目，避免大段叙述。
- 新增 workload 时，必须同步更新 `README.md` 的「当前 workload」表格：
  - 使用超链接格式：`[workload_name](link)`
  - 没有链接时留空：`[workload_name]()`
- **每次提交前必须先征得用户同意，不得擅自提交**。
- **每次提交前必须运行 `clang-format` 格式化所有 C++/CUDA 代码**。

## Modal 参考
- 开发相关的 Modal App 时需要参考官方文档：
  - `https://modal.com/docs/guide`
  - `https://modal.com/docs/reference`
  - `https://modal.com/docs/examples`
- 上述链接的子目录及其子目录可以无限查阅。

## 更新记录规则
- 仓库根目录下使用 `update/` 保存变更记录。
- 记录文件按日期命名：`YYYY-MM-DD.md`（例如：`2026-03-11.md`）。
- 同一天的更新追加到同一个文件中。
- 每条记录保持简洁，建议包含：
  - 改了什么（文件/功能）
  - 为什么改（问题或目标）
  - 结果（是否验证通过）
