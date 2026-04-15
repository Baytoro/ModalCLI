# Reduction (Parallel Sum)

## 问题定义

对 32 位浮点数组进行并行归约，计算所有元素的和：

```
output = Σ input[i]
```

## 约束条件

- 1 ≤ N ≤ 100,000,000
- -1000.0 ≤ input[i] ≤ 1000.0
- 最终和值保证在 32 位浮点范围内
- 性能测试：N = 4,194,304
- 不允许使用外部库
- 必须保持 `solve` 函数签名不变
- 最终结果存入 `output` 变量

## 示例

**Example 1:**
- Input: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
- Output: 36.0

**Example 2:**
- Input: [-2.5, 1.5, -1.0, 2.0]
- Output: 0.0

## 实现思路

1. 每个 block 处理一个数据块
2. Block 内使用 shared memory 归约（树形结构）
3. Warp 内使用 shuffle 指令加速归约
4. 每个 block 的结果通过 atomicAdd 写入全局 output

## 优化日志

暂无

## 实验记录

暂无