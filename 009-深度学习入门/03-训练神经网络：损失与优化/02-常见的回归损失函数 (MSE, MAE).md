# 常见的回归损失函数 (MSE, MAE)

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/regression-loss-functions)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练神经网络 (neural network)需要尽量减少其预测值与实际目标值之间的差异。对于回归问题，其目标是预测连续数值（例如预测房价或股票价值），为此任务设计的特定损失函数 (loss function)至关重要。两种主要的函数是：均方误差 (MSE) 和平均绝对误差 (MAE)。

### 均方误差 (MSE)

均方误差，通常称为 MSE 或 L2 损失，可能是回归中最广泛使用的损失函数 (loss function)。它衡量预测值与真实值之间误差平方的平均值。

数学上，对于 $n$ 个数据点，其中 $y_i$ 是第 $i$ 个数据点的真实目标值，$\hat{y}_i$ 是预测值，MSE 定义为：


$$
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$


**为什么要平方差值？**

1. **非负性：** 平方确保误差项始终为非负数。完美的预测会产生 0 误差。
2. **惩罚大误差：** 平方运算对大预测误差的惩罚远大于小预测误差。例如，误差为 4 会使总和增加 16，而误差为 2 则只增加 4。这一特点使 MSE 对异常值敏感，因为单个大误差会大幅增加总损失。
3. **数学上的便利性：** MSE 函数是连续可微的，这会产生平滑的梯度。这对于梯度下降 (gradient descent)等基于梯度的优化算法有利，因为它为参数 (parameter)更新提供了明确的方向。梯度指向损失增加最快的方向，因此向相反方向移动有助于减小损失。

MSE 的单位是目标变量单位的平方（例如，如果预测价格，单位是美元的平方）。尽管这可能看起来不太直观，但均方根误差 (RMSE)，即 $\sqrt{MSE}$，由于与目标变量具有相同的单位，因此常用于解释。然而，由于其有利的数学特性，MSE 通常在训练期间被最小化。

### 平均绝对误差 (MAE)

平均绝对误差，也称为 MAE 或 L1 损失，为衡量回归误差提供了另一种视角。MAE 不对差值进行平方，而是计算预测值与真实值之间绝对差值的平均值。

MAE 的公式是：


$$
L_{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$


**MAE 的特点：**

1. **对异常值的鲁棒性：** 因为 MAE 使用绝对差值，它不像 MSE 那样不成比例地惩罚大误差。误差为 4 会使总和增加 4，误差为 2 则增加 2。这种线性惩罚使得 MAE 对数据集中的异常值不如 MSE 敏感。如果您的数据包含您不希望主导损失计算的显著异常值，MAE 可能是一个更好的选择。
2. **可解释性：** 生成的 MAE 值与目标变量单位相同（例如，如果预测价格，单位是美元），因此可以直接将其解释为与真实值的平均绝对偏差。
3. **梯度行为：** MAE 函数对于非零误差具有恒定的梯度大小（$+1$ 或 $-1$）。尽管这可以防止大误差的梯度消失（与 MSE 不同，MSE 的梯度在误差接近零时趋于零），但它可能导致在最小值附近收敛性较差，因为梯度不会缩小。此外，当误差恰好为零时，导数未定义，尽管这在实践中通常通过使用次梯度或在误差恰好为零时跳过更新来处理。

### 比较 MSE 和 MAE

MSE 和 MAE 之间的选择取决于具体问题和数据特点。

- **对异常值的敏感度：** MSE 会严厉惩罚异常值；MAE 则更直接。
- **梯度：** MSE 具有平滑梯度，随误差减小而减小，可能导致在最小值附近收敛更稳定。MAE 具有恒定梯度大小，这对于大误差可能有利，但可能需要在训练进行过程中调整学习率。
- **可解释性：** MAE 的单位与目标变量匹配，使其更易于直接解释。

以下是 MSE 和 MAE 如何惩罚误差的视觉比较：



[交互图表：MSE 与 MAE 损失惩罚](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-3-training-loss-optimization/regression-loss-functions#plot-1tprlcu)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "MSE 与 MAE 损失惩罚",
    "xaxis": {
      "title": "预测误差 (y - y_hat)"
    },
    "yaxis": {
      "title": "损失值"
    },
    "legend": {
      "yanchor": "top",
      "y": 0.99,
      "xanchor": "left",
      "x": 0.01
    }
  },
  "data": [
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        25,
        16,
        9,
        4,
        1,
        0,
        1,
        4,
        9,
        16,
        25
      ],
      "mode": "lines",
      "name": "MSE",
      "line": {
        "color": "#4263eb"
      }
    },
    {
      "x": [
        -5,
        -4,
        -3,
        -2,
        -1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        5,
        4,
        3,
        2,
        1,
        0,
        1,
        2,
        3,
        4,
        5
      ],
      "mode": "lines",
      "name": "MAE",
      "line": {
        "color": "#12b886"
      }
    }
  ]
}
```

</details>



> 基于预测误差的 MSE（蓝色）和 MAE（绿色）损失值比较。请注意 MSE 的二次方增长与 MAE 的线性增长。

实践中，由于其数学特性与梯度下降 (gradient descent)高度契合，MSE 通常是默认选择。然而，如果异常值是一个主要问题，尝试使用 MAE（或 Huber 损失等其他鲁棒损失函数 (loss function)，它结合了两者的特点）是值得的。

这是一个使用 PyTorch 计算两种损失的快速示例：

```python
import torch
import torch.nn as nn

# 示例预测和目标
predictions = torch.tensor([2.5, 0.0, 1.8, 9.1])
targets = torch.tensor([3.0, -0.5, 2.0, 8.0])

# 计算 MSE 损失
mse_loss_fn = nn.MSELoss()
mse_loss = mse_loss_fn(predictions, targets)

# 计算 MAE 损失（PyTorch 中的 L1Loss）
mae_loss_fn = nn.L1Loss()
mae_loss = mae_loss_fn(predictions, targets)

print(f"预测值: {predictions.numpy()}")
print(f"目标值:     {targets.numpy()}")
print(f"误差:      {(targets - predictions).numpy()}")
print(f"平方误差: {((targets - predictions)**2).numpy()}")
print(f"绝对误差: {torch.abs(targets - predictions).numpy()}")
print("-" * 30)
print(f"MSE 损失: {mse_loss.item():.4f}") # 平均值：[0.25, 0.25, 0.04, 1.21] = 1.75 / 4 = 0.4375
print(f"MAE 损失: {mae_loss.item():.4f}") # 平均值：[0.5, 0.5, 0.2, 1.1] = 2.3 / 4 = 0.5750
```

这种简单的计算表明，与 MAE 相比，MSE 中的平方运算如何赋予较大误差（1.1）更大的权重 (weight)。了解这些差异对于选择合适的损失函数以引导模型训练过程、从而做出准确预测非常重要。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教材，阐释了各种损失函数、它们的特性及其在训练深度神经网络中的作用。
- [Supervised Learning (CS229 Lecture Notes)](https://cs229.stanford.edu/notes2023fall/cs229-notes.pdf) — Andrew Ng, Tengyu Ma (2023)
  Publisher: Stanford University
  清晰地解释了线性回归和MSE等基本损失函数在更广泛的机器学习背景中的应用。
- [MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) — PyTorch Contributors (2023)
  PyTorch官方关于均方误差（MSE）损失函数的文档，包括其用法和参数。
- [L1Loss](https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html) — PyTorch Contributors (2023)
  Publisher: PyTorch Foundation
  PyTorch官方关于平均绝对误差（MAE）损失函数（L1Loss）的文档，详细说明了其实现和使用方法。

---

[上一节](01-%E8%A1%A1%E9%87%8F%E8%A1%A8%E7%8E%B0%EF%BC%9A%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md) · [下一节](03-%E7%94%A8%E4%BA%8E%E5%88%86%E7%B1%BB%E7%9A%84%E5%B8%B8%E7%94%A8%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0%EF%BC%88%E4%BA%A4%E5%8F%89%E7%86%B5%EF%BC%89.md)
