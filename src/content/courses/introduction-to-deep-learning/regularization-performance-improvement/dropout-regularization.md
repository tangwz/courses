---
course: "introduction-to-deep-learning"
chapter: "regularization-performance-improvement"
lesson: "dropout-regularization"
sourceId: 5065
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-6-regularization-performance-improvement/dropout-regularization"
title: "Dropout正则化"
description: "了解训练期间随机丢弃单元（Dropout）如何提高模型泛化能力。"
order: 4
plots: []
sourceHash: "17cf4419f40f826ccc293445aa0825ab50b4805407ed6f1ada109dee63ba4e3a"
sourceCorrections: []
---

**Dropout**是一种计算开销低廉、效果显著的正则化 (regularization)技术，专为神经网络 (neural network)设计，旨在防止深度学习 (deep learning)模型过拟合 (overfitting)。该方法由Geoffrey Hinton及其同事于2012年提出，采取了一种巧妙的策略。与L1和L2正则化通过在损失函数 (loss function)中增加对权重 (weight)的惩罚来避免模型过于复杂不同，Dropout采用了一种独特的方法。

Dropout不是修改损失函数，而是在训练期间改变网络本身。其核心思想很简单：**在每个训练步骤中，随机将某层中一些神经元的输出设为零。**

### 工作原理：随机停用神经元

想象一下神经网络 (neural network)在训练过程中的前向传播中的一层。在将该层的输出（激活值）传递给下一层之前，Dropout操作会随机选择这些激活值中的一部分，这部分由**丢弃率** $p$ 定义，并强制将其设为零。

例如，如果一个层有100个神经元，并且丢弃率 $p=0.4$（意味着40%的丢弃），那么在单个训练迭代中，该层中大约40个随机选择的神经元的输出将被暂时归零。其余60%的神经元将照常运作，但它们的输出通常会按比例放大以弥补缺失的部分（稍后将说明原因）。

> 显示有无Dropout的层视图。在应用Dropout进行训练时（右侧），一些神经元输出（标有'X'）在该次特定前向传播中被随机设为零。

重要的一点是，被丢弃的神经元集合在每个训练迭代（或小批量）中随机变化。这意味着一个神经元的输出可能在某一步骤中被使用，而在下一步骤中被丢弃。

### 为何奏效？

Dropout防止神经元变得过度特化或过度依赖于特定其他神经元的存在。由于任何神经元的输出在训练期间都可能消失，网络被迫学习更分布式的表示。神经元必须学习独立有用，或与其他神经元的不同随机子集结合使用时有用的特征，而不是依赖脆弱的协同适应。

可以将其类比为体育运动中的交叉训练。一个只与完全相同的队友练习特定战术的运动员，如果队友受伤或对手改变策略，可能会遇到困难。而一个在各种条件下以及与不同团队配置下训练的运动员则变得更具适应性，并普遍更熟练。类似地，Dropout迫使神经元更具个体能力，更少依赖固定环境。

另一种理解Dropout的方式是，它是一种有效的方法，近似于训练大量不同的精简网络结构。每个训练步骤有效地采样并训练一个从原始网络派生出的不同子网络。在测试时，使用没有Dropout的完整网络，其作用有点像对这个庞大精简网络集成体的预测进行平均，这通常能提高泛化能力。

### 实现方式：反向Dropout与缩放

最常见的实现技术称为**反向Dropout**。以下是它在训练期间的工作方式：

1. **生成随机掩码：** 对于某层的激活值 $a$，创建一个形状相同的二进制掩码 $m$，每个元素为1的概率是 $1-p$（保留概率），为0的概率是 $p$（丢弃率）。
2. **应用掩码：** 将激活值与掩码按元素相乘：$a_{dropped} = a * m$。这会将掩码中对应于0的激活值设为零。
3. **缩放剩余激活值：** 将结果除以保留概率 $(1-p)$：$a_{final} = a_{dropped} / (1-p)$。

为何需要缩放步骤？在训练期间，平均而言，只有 $(1-p)$ 比例的神经元参与贡献到层的输出。通过 $1/(1-p)$ 进行缩放可确保激活值的期望和保持不变，就像没有Dropout一样。这很重要，因为**Dropout仅在训练期间应用**。

\*\*在测试或推断时，Dropout被关闭。\*\*所有神经元都处于活跃状态，无需缩放，因为缩放已在训练阶段处理（这就是“反向”的含义）。这使得推断的计算方式与没有Dropout的网络相同。

### Dropout的实际应用

Dropout通常应用于隐藏层的输出，常在激活函数 (activation function)之后。直接应用于输入层较不常见，但并非不可行。它通常不应用于输出层，尤其是对于输出代表概率的分类任务。

常见的丢弃率 ($p$) 范围从0.1到0.5。更高的丢弃率意味着更强的正则化 (regularization)。最佳丢弃率是一个超参数 (parameter) (hyperparameter)，通常需要根据特定的网络结构和数据集进行调整。

以下是您可能在PyTorch中使用 `nn.Sequential` 添加Dropout层的方式：

```python
import torch
import torch.nn as nn

# 示例：带有Dropout的简单前馈网络
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(p=0.5),  # 在第一个隐藏层的激活后应用Dropout
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),  # 在第二个隐藏层的激活后应用Dropout
    nn.Linear(128, 10)   # 输出层（通常不使用Dropout）
)

# --- 训练期间 ---
model.train() # 将模型设置为训练模式（激活Dropout）
# ... 训练循环 ...
# output = model(input_batch)
# loss = criterion(output, target_batch)
# ... 反向传播 ...

# --- 评估/测试期间 ---
model.eval() # 将模型设置为评估模式（停用Dropout）
with torch.no_grad(): # 为推断禁用梯度计算
    # test_output = model(test_input_batch)
    # ... 计算准确率/指标 ...

print(model)
```

请注意调用 `model.train()` 和 `model.eval()`。这很重要，因为它们告诉诸如 `nn.Dropout`（以及 `nn.BatchNorm` 等其他层）是应在训练模式下运行（应用Dropout，更新统计数据）还是在评估模式下运行（禁用Dropout，使用固定统计数据）。

### 优点与考量

**优点：**

- **有效正则化 (regularization)：** 通常显著减少过拟合 (overfitting)并提高泛化能力。
- **实现简单：** 易于在大多数框架中添加到现有网络结构中。
- **计算开销低：** 与不使用Dropout的训练相比，计算开销极小。

**考量：**

- **超参数 (parameter) (hyperparameter)调整：** 丢弃率 $p$ 需要选择，通常需要通过实验确定。
- **训练时间更长：** 因为网络在每个步骤中训练的是略微不同的“精简”版本，所以与没有Dropout的网络相比，可能需要更多训练周期才能收敛（尽管最终结果通常更佳）。
- **与批量归一化 (normalization)的关联：** 尽管常同时使用，但Dropout和批量归一化之间的关联有时会很复杂。一些研究表明，将Dropout置于批量归一化*之后*可能更佳，但标准做法不尽相同。

Dropout仍然是正则化深度神经网络 (neural network)的一项基础技术，提供了一种简单而有效的方法来构建在未见数据上表现更好的模型。它通过防止神经元之间过度依赖来增强能力，从而促成更具泛化能力的特征学习。

## 参考资料

- [Dropout: A Simple Way to Prevent Overfitting in Neural Networks](http://www.jmlr.org/papers/v15/srivastava14a.html) — Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov (2014)
  Journal: Journal of Machine Learning Research; Volume: 15; Pages: 1929−1958
  提出Dropout技术并证明其在防止神经网络过拟合方面有效性的原始基础研究论文。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 7
  一本权威的深度学习教材，详细讨论了包括Dropout在内的正则化技术等基础概念。
- [\`torch.nn.Dropout\`](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch中`Dropout`层的官方文档，提供了其实现、参数和在深度学习模型中使用的详细信息。
