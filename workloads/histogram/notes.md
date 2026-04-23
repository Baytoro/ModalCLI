# Histogramming

## 问题定义

计算输入数组的直方图，统计每个值在 [0, num_bins) 范围内的出现次数：

```
histogram[value] = count of occurrences of value in input
```

## 约束条件

- 1 ≤ N ≤ 100,000,000
- 0 ≤ input[i] < num_bins
- 1 ≤ num_bins ≤ 1024
- 性能测试：N = 50,000,000, num_bins = 256
- 不允许使用外部库
- 必须保持 `solve` 函数签名不变
- 最终结果存入 `histogram` 数组

## 实现要点

1. **Atomic Add**: 多个线程可能同时写入同一个 bin，需要 atomicAdd
2. **Memory Coalescing**: 合理安排全局内存访问模式
3. **Bank Conflict**: Shared memory 访问需注意 bank conflict 优化

## 示例

**Example 1:**
- Input: input = [0, 1, 2, 1, 0], N = 5, num_bins = 3
- Output: [2, 2, 1]

**Example 2:**
- Input: input = [3, 3, 3, 3], N = 4, num_bins = 5
- Output: [0, 0, 0, 4, 0]

## 优化日志

暂无

## 实验记录

暂无