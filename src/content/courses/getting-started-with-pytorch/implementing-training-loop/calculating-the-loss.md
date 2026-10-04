---
course: "getting-started-with-pytorch"
chapter: "implementing-training-loop"
lesson: "calculating-the-loss"
sourceId: 2193
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/calculating-the-loss"
title: "计算损失"
description: "使用所选损失函数比较模型预测与实际标签。"
order: 5
plots: []
sourceHash: "2a6e1ed6a5f2379bfe1da8cbb8be59ee7c856f476f1d87e4b7660d29b8704c45"
sourceCorrections: []
---

模型根据输入数据批次生成预测（通常称为 `outputs` 或 `logits`）。为了评估这些预测与实际真实标签的匹配程度，需要一个评估机制。这就是损失函数 (loss function)的作用。

### 量化 (quantization)模型误差

一个**损失函数 (loss function)**，也称为**标准**或**目标函数**，它以数学方式衡量模型预测（$\hat{y}$）与真实目标值（$y$）之间的差异。训练的目的通常是使这个损失值最小化。损失越小表示模型的预测越接近给定数据批次的实际目标。

在PyTorch中，损失函数在 `torch.nn` 模块中随即可用，就像模型层和激活函数 (activation function)一样。您通常在训练循环外部只实例化一次损失函数。常见选择包括：

- `nn.MSELoss`（均方误差）：常用于回归任务，目标是预测连续值。它计算预测和目标之间的平均平方差。
  
  $$
  L_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
  $$
  
  其中 $N$ 是批次中的样本数量。
- `nn.CrossEntropyLoss`：是多类别分类问题的标准选择。此标准在一个类中方便地结合了 `nn.LogSoftmax` 和 `nn.NLLLoss`（负对数似然损失）。它需要来自模型最后一层的原始、未归一化 (normalization)的分数（logits）作为输入，以及目标类别索引（整数）作为标签。
- `nn.BCEWithLogitsLoss`：用于二元分类或多标签分类任务。与 `CrossEntropyLoss` 类似，它在一个步骤中结合了 Sigmoid 层和二元交叉熵损失，以获得更好的数值稳定性。它也需要原始 logits 作为输入。

### 在PyTorch中计算损失

一旦您实例化了所选的标准（例如，`criterion = nn.CrossEntropyLoss()`），在循环中计算损失就很直接。您只需将模型的输出张量和包含真实标签的张量传递给标准对象即可：

```python
# --- 训练循环内部 ---

# 假设：
# model: 您的神经网络模型（例如，nn.Module 实例）
# criterion: 您选择的损失函数（例如，nn.CrossEntropyLoss()）
# inputs: 来自 DataLoader 的输入数据批次
# labels: 来自 DataLoader 的相应真实标签批次

# 1. 正向传播（已完成）
outputs = model(inputs)

# 2. 计算损失
loss = criterion(outputs, labels)

# 'loss' 现在包含当前批次的计算损失值。
# 它是一个标量张量（只有一个元素的张量）。

# 3. 后续步骤：反向传播 (loss.backward())，优化器步进...
# --- 循环迭代片段结束 ---
```

理解生成的 `loss` 变量代表什么很重要：

1. **标量值：** 它通常是一个数值（一个零维张量），表示整个批次的平均损失。
2. **计算图连接：** 重要的是，这个 `loss` 张量仍然连接着 PyTorch 在正向传播期间构建的计算图。它知道哪些操作和哪些模型参数 (parameter)促成了它的最终值。
3. **梯度计算已启用：** 因为它连接着计算图并依赖于 `requires_grad=True` 的参数，所以 `loss` 张量本身隐式地具有 `requires_grad=True`。这使得我们可以在下一步调用 `loss.backward()`，自动计算损失相对于模型所有可学习参数（$\nabla_{\theta} L$）的梯度。

这个计算出的 `loss` 值作为反向传播 (backpropagation)过程的起点，它调整模型的权重 (weight)，以期在后续迭代中产生更低的损失。

## 参考资料

- [torch.nn.modules.loss](https://pytorch.org/docs/stable/nn.html#loss-functions) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  PyTorch 官方损失函数文档，包含 `MSELoss`、`CrossEntropyLoss` 和 `BCEWithLogitsLoss` 的用法及参数细节。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本综合性教科书，阐述了损失函数的数学和概念基础以及它们在神经网络训练中的作用。
- [Linear Classification: Support Vector Machine, Softmax](http://cs231n.github.io/linear-classify/#softmax) — Andrej Karpathy and Fei-Fei Li (2023)
  斯坦福大学 CS231n 课程讲义，涵盖了损失函数（特别是交叉熵）在分类任务中的理论背景和实际应用。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Christopher M. Bishop (2006)
  Publisher: Springer
  一本基础教科书，从更广泛的机器学习角度，提供了关于各种损失函数（包括平方误差和交叉熵）的统计和概率视角。
