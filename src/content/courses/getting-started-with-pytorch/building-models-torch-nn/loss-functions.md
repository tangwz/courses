---
course: "getting-started-with-pytorch"
chapter: "building-models-torch-nn"
lesson: "loss-functions"
sourceId: 2165
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-4-building-models-torch-nn/loss-functions"
title: "损失函数 (`torch.nn` 损失)"
description: "理解和使用不同损失函数，如均方误差和交叉熵损失。"
order: 6
plots: []
sourceHash: "d78cf3a636dfc400fd777821298f70fe56fdace56122334fd9bbf954b8cd8f39"
sourceCorrections: []
---

神经网络 (neural network)，由堆叠的层和激活函数 (activation function)构建而成，进行预测。为了使这些网络有效学习，必须有一种方法来量化 (quantization)其预测值与真实目标值之间的偏差有多大。这种量化是**损失函数 (loss function)**的主要作用，损失函数也称为准则函数或目标函数。

`torch.nn` 包提供了深度学习 (deep learning)中常用的一系列标准损失函数。基本思路很简单：损失函数接收模型的输出（预测值）和真实值（目标值）作为输入，并计算出一个标量值，表示“误差”或“损失”。然后，PyTorch 的 Autograd 系统在反向传播 (backpropagation)过程中使用这个标量损失值来计算梯度，这些梯度进而指导优化器（如 `torch.optim` 中的 SGD 或 Adam）如何调整模型的参数 (parameter)（权重 (weight)和偏置 (bias)）以最小化此损失。

选择合适的损失函数非常重要，因为它直接定义了模型试图达成的目标。让我们看看 `torch.nn` 中一些最常用的损失函数。

> 损失函数比较模型预测值与目标数据，以生成一个标量损失值，该值通过反向传播指导参数更新。

### 常用损失函数 (loss function)

PyTorch 将损失函数实现为继承自 `nn.Module` 的类。你首先实例化损失函数类，然后使用模型的预测值和目标值调用该实例。

#### 回归损失

这些通常用于目标是预测连续值的情况。

1. **均方误差 (MSELoss)**: 可能是回归任务中最常用的损失函数。它衡量预测值和实际值之间差异的平方的平均值。

   公式如下：

   
   $$
   \text{损失}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
   $$
   

   其中 $N$ 是批次中的样本数量，$y_i$ 是真实值，$\hat{y}_i$ 是预测值。对差异进行平方会更严厉地惩罚较大的误差。

   使用 `torch.nn.MSELoss`：

   ```python
   import torch
   import torch.nn as nn

   # 实例化损失函数
   loss_fn = nn.MSELoss()

   # 示例预测值和目标值（批大小为 3，1 个输出特征）
   predictions = torch.randn(3, 1, requires_grad=True)
   targets = torch.randn(3, 1)

   # 计算损失
   loss = loss_fn(predictions, targets)
   print(f"MSE Loss: {loss.item()}")

   # 现在可以通过 loss.backward() 计算梯度
   # loss.backward()
   # print(predictions.grad)
   ```
2. **平均绝对误差 (L1Loss)**: 另一种常用的回归损失。它衡量预测值和实际值之间绝对差异的平均值。

   公式如下：

   
   $$
   \text{损失}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
   $$
   

   与 MSE 相比，L1 损失通常被认为对异常值不那么敏感，因为它不对误差进行平方。

   使用 `torch.nn.L1Loss`：

   ```python
   import torch
   import torch.nn as nn

   loss_fn_l1 = nn.L1Loss()
   predictions = torch.tensor([[1.0], [2.5], [0.0]], requires_grad=True)
   targets = torch.tensor([[1.2], [2.2], [0.5]])

   loss_l1 = loss_fn_l1(predictions, targets)
   print(f"L1 Loss: {loss_l1.item()}") # |1-1.2|, |2.5-2.2|, |0-0.5| 的平均值
                                       # (0.2 + 0.3 + 0.5) / 3 = 1.0 / 3 = 0.333...
   ```

#### 分类损失

这些用于目标是预测离散类别标签的情况。

1. **交叉熵损失 (CrossEntropyLoss)**: 这是多类别分类问题的标准损失函数。当你的模型为每个类别输出原始分数（logits）时，它特别有效。

   `torch.nn.CrossEntropyLoss` 方便地将两个步骤合二为一：

   - 对模型的原始输出分数（logits）应用 `LogSoftmax` 函数。Softmax 将 logits 转换为和为 1 的概率，而 LogSoftmax 则取这些概率的对数。
   - 计算 LogSoftmax 输出和目标类别索引之间的负对数似然损失 (`NLLLoss`)。

   它需要：

   - **输入（预测值）：** 每个类别的原始、未归一化 (normalization)分数（logits）。形状通常为 `(N, C)`，其中 `N` 是批大小，`C` 是类别数量。
   - **目标：** 类别索引（从 0 到 C-1 的整数）。形状通常为 `(N)`。

   ```python
   import torch
   import torch.nn as nn

   loss_fn_ce = nn.CrossEntropyLoss()

   # 示例：包含 3 个样本的批次，5 个类别
   # 来自模型的原始分数（logits）
   predictions_logits = torch.randn(3, 5, requires_grad=True)
   # 真实类别索引（必须是 LongTensor）
   targets_classes = torch.tensor([1, 0, 4]) # 3 个样本的类别索引

   loss_ce = loss_fn_ce(predictions_logits, targets_classes)
   print(f"Cross-Entropy Loss: {loss_ce.item()}")

   # loss_ce.backward()
   # print(predictions_logits.grad)
   ```

   通常推荐使用 `nn.CrossEntropyLoss`，而不是手动应用 `LogSoftmax` 和 `NLLLoss`，因为前者具有更好的数值稳定性。
2. **二元交叉熵损失 (BCELoss 和 BCEWithLogitsLoss)**: 用于二元（两类别）分类问题或多标签分类（每个样本可以属于多个类别）。

   - `torch.nn.BCELoss`: 计算目标与输出之间的二元交叉熵。它期望模型的输出已经是概率（例如，在应用 Sigmoid 激活函数 (activation function)之后），通常在 [0, 1] 范围内。

     - 输入（预测值）：概率，形状 `(N, *)`。
     - 目标：概率（通常为 0.0 或 1.0），与输入形状相同。
   - `torch.nn.BCEWithLogitsLoss`: 这个版本在数值上比先使用 Sigmoid 层再使用 `BCELoss` 更稳定和方便。它将 Sigmoid 激活和 BCE 计算合为一步。它期望原始 logits 作为输入。

     - 输入（预测值）：原始 logits（Sigmoid 之前），形状 `(N, *)`。
     - 目标：概率（通常为 0.0 或 1.0），与输入形状相同。

   对于大多数二元分类任务，`BCEWithLogitsLoss` 是更优选择：

   ```python
   import torch
   import torch.nn as nn

   loss_fn_bce_logits = nn.BCEWithLogitsLoss()

   # 示例：包含 4 个样本的批次，1 个输出节点（二元分类）
   predictions_logits_bin = torch.randn(4, 1, requires_grad=True) # 原始 logits
   # 目标应为浮点数（0.0 或 1.0）
   targets_bin = torch.tensor([[1.0], [0.0], [0.0], [1.0]])

   loss_bce = loss_fn_bce_logits(predictions_logits_bin, targets_bin)
   print(f"BCE With Logits Loss: {loss_bce.item()}")
   ```

### 选择合适的损失函数 (loss function)

选择很大程度上取决于你的具体任务：

- **回归（预测连续值）：** 从 `nn.MSELoss` 开始。如果你怀疑异常值严重影响训练，可以考虑 `nn.L1Loss`。
- **二元分类（两个类别）：** 使用 `nn.BCEWithLogitsLoss`。确保你的模型有一个输出节点产生 logits。
- **多类别分类（每个样本只有一个正确类别）：** 使用 `nn.CrossEntropyLoss`。确保你的模型有 `C` 个输出节点产生 logits，其中 `C` 是类别数量。
- **多标签分类（每个样本可以有多个正确类别）：** 使用 `nn.BCEWithLogitsLoss`。确保你的模型有 `C` 个输出节点产生 logits，并且你的目标是多热编码（例如，如果存在类别 0 和 2，则为 `[1.0, 0.0, 1.0, 0.0]`）。

### 在训练中使用损失函数 (loss function)

在典型的训练循环中，你会在循环外部实例化一次所选择的损失函数。在循环内部，获取模型对一批数据的预测后，将预测值和相应的目标标签传递给损失函数实例，以计算该批次的损失。

```python
# 假设模型、优化器、数据加载器已定义

# --- 训练循环外部 ---
# 示例：多类别分类
num_classes = 10
model = nn.Linear(784, num_classes) # 简单的线性模型示例
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# 模拟数据加载器（替换为实际的 DataLoader）
dummy_dataloader = [(torch.randn(64, 784), torch.randint(0, num_classes, (64,))) for _ in range(5)]

# --- 训练循环内部 ---
model.train() # 将模型设置为训练模式
for batch_idx, (data, target) in enumerate(dummy_dataloader):
    # 1. 清零梯度
    optimizer.zero_grad()

    # 2. 前向传播：获取预测值（logits）
    predictions = model(data)

    # 3. 计算损失
    loss = loss_fn(predictions, target)

    # 4. 反向传播：计算梯度
    loss.backward()

    # 5. 优化器步进：更新权重
    optimizer.step()

    if batch_idx % 2 == 0: # 定期打印损失
        print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
```

通过从 `torch.nn` 中选择合适的损失函数并将其正确集成到你的训练过程中，你为模型提供了一个明确的学习目标，与优化器和反向传播 (backpropagation)一起构成了模型训练机制的根本部分。

## 参考资料

- [Loss Functions (torch.nn) - PyTorch Documentation](https://pytorch.org/docs/stable/nn.html#loss-functions) — PyTorch Contributors (2024)
  PyTorch `torch.nn`包的官方文档，提供了所有可用损失函数的详细信息和示例。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 129-148, 184-192
  一本基础教材，提供了各种机器学习概念的数学和理论背景，包括损失函数、误差度量及其在优化中的作用。
- [CS231n: Deep Learning for Computer Vision, Lecture Notes on Loss Functions](https://cs231n.github.io/linear-classify/#loss) — Stanford University CS231n Course Staff (2024)
  斯坦福大学备受推崇的计算机视觉课程讲义，清晰地解释了深度学习中使用的分类和回归损失函数。
