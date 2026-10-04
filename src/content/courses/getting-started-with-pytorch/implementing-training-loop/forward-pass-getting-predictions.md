---
course: "getting-started-with-pytorch"
chapter: "implementing-training-loop"
lesson: "forward-pass-getting-predictions"
sourceId: 2191
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-6-implementing-training-loop/forward-pass-getting-predictions"
title: "前向传播：获取预测结果"
description: "将输入数据通过模型以获得输出预测结果。"
order: 4
plots: []
sourceHash: "8010db929915a69234042356af983b54bba3d458811162be683989782bd86a2e"
sourceCorrections: []
---

设置好模型、损失函数 (loss function)、优化器和数据加载器后，训练循环的核心部分便开始了。每次迭代中的第一个操作步骤是**前向传播**。在此步骤中，模型接收当前批次的输入数据并生成预测结果。

可以将前向传播看作信息流经神经网络 (neural network)的过程，从输入层，经过所有隐藏层，最终到达输出层。每个层都会对其从前一个层接收到的数据执行其定义的计算。

### PyTorch中前向传播的工作原理

前向传播在PyTorch中是一个直接的过程。通过继承`torch.nn.Module`，用户可以在模型的`forward`方法中明确定义其操作的结构和顺序。

PyTorch的`nn.Module`基类提供了一个`__call__`方法。这使您能够将模型实例当作函数一样使用。当您调用`model(inputs)`时，PyTorch会隐式地执行您定义的`forward`方法，将`inputs`传递通过网络的各层。

### 实现前向传播

在您的训练循环内部，从`DataLoader`获取一个批次的数据（特征和标签）后，您将特征（输入）送入您的模型：

```python
# 假设 'model' 是您的 nn.Module 子类的一个实例
# 假设 'data_batch' 是从 DataLoader 加载的
# 假设 'device' 已定义（例如，'cuda' 或 'cpu'）

# 解包批次数据（根据您的 DataLoader 结构进行调整）
inputs, labels = data_batch
inputs = inputs.to(device) # 将输入数据移动到正确的设备上
labels = labels.to(device) # 将标签移动到正确的设备上（损失计算需要）

# --- 前向传播 ---
# 将输入通过模型
outputs = model(inputs)
# -----------------------

# 'outputs' 现在包含模型对 'inputs' 批次的预测结果。
# 下一步将使用这些 'outputs' 和 'labels' 计算损失。
```

以下是此步骤的可视化表示：

> 在前向传播过程中数据流向：加载一个批次，输入数据被准备并发送到合适的设备，然后通过模型生成输出。

### 理解输出

前向传播生成的`outputs`张量包含了模型对给定输入批次的预测结果。这些预测结果的具体性质取决于模型的最后一层和任务：

- **分类：** 通常，输出代表每个类别的原始分数，称为 **logits**。这些logits通常会传递给像`nn.CrossEntropyLoss`这样的损失函数 (loss function)，该函数会在内部应用Softmax函数。
- **回归：** 输出可能直接代表预测的连续值。
- **其他任务：** 对于像分割或目标检测这样的任务，输出结构会更复杂，以反映任务的具体要求。

务必确保您的输入数据（`inputs`）和模型（`model`）位于相同的计算设备上（CPU或特定的GPU）。如果它们在不同的设备上，PyTorch会抛出运行时错误。如代码片段所示，使用`.to(device)`将数据批次移动到指定的`device`是防止这种情况发生的标准做法。

前向传播根据模型的当前参数 (parameter)（权重 (weight)和偏差）计算模型的预测结果。这些预测结果随后将在下一步：计算损失中与真实标签进行比较。

## 参考资料

- [\`torch.nn\` - PyTorch Documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Foundation (2024)
  描述PyTorch中神经网络的基本构建块，包括`nn.Module`基类及其`forward`方法。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: 183-228
  提供神经网络的理论基础，包括对前向传播机制的讨论。
- [Training a Classifier (PyTorch Documentation Tutorial)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) — PyTorch Foundation (2024)
  Publisher: PyTorch Foundation
  展示PyTorch中完整的训练流程，展示在循环中实现前向传播。
- [Tensors - PyTorch Documentation](https://pytorch.org/docs/stable/tensors.html) — PyTorch Foundation (2024)
  Publisher: PyTorch Foundation
  解释张量属性，包括设备放置，以及在设备之间移动张量的方法。
