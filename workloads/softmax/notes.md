# Softmax

## 问题定义

对长度为 N 的浮点数组计算 softmax 函数：

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

## 约束条件

- 1 ≤ N ≤ 500,000
- 性能测试：N = 500,000
- 不允许使用外部库
- 必须保持 `solve` 函数签名不变
- 最终结果存入 `output` 数组

## 实现要点

1. **Overflow 处理 (Max Trick)**
   - 减去数组最大值避免 exp 溢出
   - `softmax(x_i) = exp(x_i - max) / Σ exp(x_j - max)`

2. **两步计算**
   - 第一步：求最大值并归约
   - 第二步：计算 exp 并归一化

3. **数值稳定性**
   - 直接计算 `exp(x_i)` 会溢出
   - 使用 max trick 保证数值稳定

## 示例

**Example 1:**
- Input: [1.0, 2.0, 3.0], N = 3
- Output: [0.090, 0.244, 0.665] (approximately)

**Example 2:**
- Input: [-10.0, -5.0, 0.0, 5.0, 10.0], N = 5
- Output: [2.047e-09, 3.038e-07, 4.509e-05, 6.693e-03, 9.933e-01] (approximately)

## 优化日志

暂无

## 实验记录

暂无