# Mean Squared Error (MSE)

## 问题定义

计算预测值 (predictions) 和目标值 (targets) 之间的均方误差：

```
MSE = (1/N) * Σ (predictions[i] - targets[i])²
```

其中 N 是数组长度。

## 约束条件

- 1 ≤ N ≤ 100,000,000
- -1000.0 ≤ predictions[i], targets[i] ≤ 1000.0
- 性能测试：N = 50,000,000
- 不允许使用外部库
- 必须保持 solve 函数签名不变
- 最终结果存入 mse 变量

## 示例

**Example 1:**
- Input: predictions = [1.0, 2.0, 3.0, 4.0], targets = [1.5, 2.5, 3.5, 4.5]
- Output: mse = 0.25

**Example 2:**
- Input: predictions = [10.0, 20.0, 30.0], targets = [12.0, 18.0, 33.0]
- Output: mse = 5.67

## 应用场景

- 机器学习损失函数
- 数值精度验证
- 模型输出对比

## 优化日志

暂无

## 实验记录

暂无
