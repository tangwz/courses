---
course: "introduction-to-graph-neural-networks"
chapter: "training-gnn-models"
lesson: "training-loop-for-gnns"
sourceId: 7636
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-4-training-gnn-models/training-loop-for-gnns"
title: "GNN 的训练循环"
description: "编写 GNN 模型标准训练循环的分步指南，包括前向传播、损失计算和反向传播。"
order: 3
plots: []
sourceHash: "292d4643f8a446d54b38c8a9d5f887b65a8d6cc78e261a052a1c1ef36840d68e"
sourceCorrections: []
---

为了让模型能够从数据中学习，一套特定的机制是必不可少的，尤其是在确定了模型架构和损失函数 (loss function)之后。这种机制被称为训练循环，它是一个迭代过程：数据被反复输入模型，计算误差，并调整模型参数 (parameter)以减小这些误差。对于在全批次（full-batch）设置下处理单个图的图神经网络 (neural network)（GNN），该循环在多个周期（或称为轮次，Epochs）中协调整个学习过程。

训练循环中的每个周期都由几个不同的、按顺序进行的步骤组成，这些步骤共同作用来优化模型的权重 (weight)。让我们分解一下这个基础过程。

### 训练步骤剖析

对训练数据的一次完整处理通常被称为一个训练步（step）或一次迭代（iteration），它包含四个主要操作：前向传播、损失计算、反向传播 (backpropagation)和参数 (parameter)更新。

#### 1. 前向传播

过程始于前向传播，我们将图数据输入到 GNN 中。模型接收节点特征 $X$ 和由邻接信息（或边索引）表示的图结构作为输入。然后，模型通过其层级结构处理这些信息，在每一层执行邻域聚合和更新。最后一层的输出通常是每个节点的原始、未归一化 (normalization)的预测值，通常称为 *logits*。

```python
# graph 包含节点特征 (x) 和边结构 (edge_index)
logits = model(graph.x, graph.edge_index)
```

这些 logits 代表了模型在应用 softmax 等归一化函数之前，对每个节点类别的当前判断。

#### 2. 损失计算

接下来，我们需要量化 (quantization)模型预测的偏差。我们将前向传播输出的 logits 与真实的标签进行比较。这种比较由我们之前选择的损失函数 (loss function)处理，例如用于节点分类的 `CrossEntropyLoss`。

GNN 训练中的一个细节是，我们通常只计算指定用于训练的节点上的损失。这通过布尔掩码（mask）来实现。掩码会过滤 logits 和标签，确保只有训练节点参与损失计算。这在半监督或转导学习（transductive）设置中非常有用，因为在这种情况下我们只拥有图中部分节点的标签。


$$
L = \text{损失函数}(\text{logits}[\text{训练掩码}], \text{标签}[\text{训练掩码}])
$$


结果是一个标量值，即损失（Loss），它代表了模型在当前训练数据批次上的误差。

#### 3. 反向传播

计算出损失后，我们需要确定每个模型参数对该误差的贡献程度。这就是反向传播的任务。PyTorch 等深度学习 (deep learning)框架使这个自动化了。通过调用 `loss.backward()`，框架会计算损失相对于模型中每个可学习参数（权重 (weight)和偏置 (bias)）的梯度。


$$
\frac{\partial L}{\partial W^{(l)}} \quad \forall l \in \{1, \dots, L\}
$$


这些梯度指明了每个参数为了降低损失而需要调整的方向和幅度。权重的正梯度意味着增加该权重会增加损失，而负梯度则意味着相反的情况。

#### 4. 参数更新

最后一步是使用这些梯度来更新模型的参数。这由优化器（如 Adam 或 SGD）处理。优化器的 `step()` 方法会根据学习率的大小，沿着减小损失的方向调整每个参数。

在开始下一个训练步之前，必须重置梯度。这是因为深度学习框架默认会累加梯度。我们调用 `optimizer.zero_grad()` 来清除旧梯度，确保当前步骤的参数更新不受之前步骤梯度的影响。

这整个四步序列构成了一个完整的训练迭代。

> 单个训练步的迭代循环。数据流经模型产生预测，预测用于计算损失。源自该损失的梯度指导优化器更新模型权重，以进行下一次迭代。

### 轮次迭代

一个**轮次**（Epoch）被定义为对整个训练数据集的一次完整处理。在小型图常用的全批次训练中，上述单个训练步会一次性处理整个图。因此，一个训练步就等于一个轮次。

训练循环会运行设定的轮次次数。随着每一轮的进行，优化器都会将模型的权重 (weight)向减小训练损失的方向微调 (fine-tuning)，从而让模型学会识别图结构和节点特征中的模式。

以下是 GNN 典型训练循环的代码示例：

```python
# model: 你的 GNN 模型
# graph: 你的图数据对象
# optimizer: Adam 等优化器
# loss_fn: CrossEntropyLoss 等损失函数

def train(model, graph, optimizer, loss_fn):
    model.train()  # 将模型设置为训练模式
    optimizer.zero_grad()  # 清除之前的梯度

    # 1. 前向传播
    logits = model(graph.x, graph.edge_index)

    # 2. 损失计算 (使用训练掩码)
    loss = loss_fn(logits[graph.train_mask], graph.y[graph.train_mask])

    # 3. 反向传播
    loss.backward()

    # 4. 参数更新
    optimizer.step()

    return loss.item()

# 主训练循环
for epoch in range(200):
    loss = train(model, graph, optimizer, loss_fn)
    print(f"轮次 {epoch+1:03d}, 损失: {loss:.4f}")
```

随着循环的进行，我们期望看到训练损失下降。这表明模型正在成功学习拟合训练数据。然而，仅仅训练损失下降是不够的。我们还必须通过验证集监测模型在未见过的数据上的表现，以确保其具备良好的泛化能力，而不仅仅是记住了训练样本。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，涵盖了深度学习的理论基础，包括通用训练循环、前向传播、损失函数、反向传播和各种优化算法。
- [Graph Neural Networks: Foundations, Frontiers, and Applications](https://link.springer.com/book/10.1007/978-981-99-3174-8) — Zonghan Wu, Shirui Pan, Fangzhao Wu, Lina Yao, Guodong Long, Chen Gong, Philip S. Yu (2023)
  Publisher: Springer; DOI: [10.1007/978-981-99-3174-8](https://doi.org/10.1007/978-981-99-3174-8)
  本书全面论述了图神经网络，详细介绍了其训练方法、图结构数据中的具体挑战以及为GNN调整的优化策略。
- [Automatic differentiation package - torch.autograd and Optimizers - torch.optim](https://pytorch.org/docs/stable/autograd.html) — PyTorch Core Team (2016)
  Publisher: PyTorch Foundation
  PyTorch官方文档，解释了用于自动微分的`torch.autograd`模块（对反向传播至关重要）以及提供各种优化算法以更新参数的`torch.optim`包。
