# Prefix Sum (Scan)

## 问题定义

计算数组的前缀和（累积和）：

```
output[i] = Σ input[j] for j in [0, i]
```

即：[a, b, c, d] → [a, a+b, a+b+c, a+b+c+d]

## 约束条件

- 1 ≤ N ≤ 100,000,000
- -1000.0 ≤ input[i] ≤ 1000.0
- 最终结果保证在 32 位浮点范围内
- 性能测试：N = 250,000
- 不允许使用外部库
- 必须保持 `solve` 函数签名不变
- 最终结果存入 `output` 数组

## 实现要点

1. **Two-Phase Algorithm**: 
   - Phase 1: Block-level reduction (每个 block 计算本地 prefix sum)
   - Phase 2: 将 block 间依赖累加到后续 block

2. **Work-Efficient**: 使用 Hillis-Steele 或 Blelloch 扫描算法
3. **Shared Memory**: 用于 block 内暂存数据
4. **跨 Block 依赖**: 需要将前一个 block 的结果累加到当前 block

## 示例

**Example 1:**
- Input: [1.0, 2.0, 3.0, 4.0]
- Output: [1.0, 3.0, 6.0, 10.0]

**Example 2:**
- Input: [5.0, -2.0, 3.0, 1.0, -4.0]
- Output: [5.0, 3.0, 6.0, 7.0, 3.0]

## 优化日志

暂无

## 实验记录

暂无