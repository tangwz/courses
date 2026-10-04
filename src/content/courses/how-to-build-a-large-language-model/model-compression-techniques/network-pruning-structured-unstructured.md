---
course: "how-to-build-a-large-language-model"
chapter: "model-compression-techniques"
lesson: "network-pruning-structured-unstructured"
sourceId: 6094
sourceUrl: "https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-27-model-compression-techniques/network-pruning-structured-unstructured"
title: "网络剪枝（结构化与非结构化）"
description: "采用基于幅度的剪枝或结构化剪枝，去除冗余权重或组件。"
order: 4
plots: []
sourceHash: "ddf1511ef5d7109b96b1cbdb2bfbffa8b6812b876c1efa8ad6d7ffbe5dea64ad"
sourceCorrections: []
---

量化 (quantization)侧重于减小模型中使用数字的精度，而网络剪枝则采用不同方式：它旨在移除被认为不重要的参数 (parameter)（权重 (weight)）甚至整个结构部件，有效地使模型更稀疏。直观想法是，大型、过参数化模型通常包含大量冗余，移除部分内容可能不会显著影响性能，特别是在重新训练或微调 (fine-tuning)之后。剪枝可以显著减小模型大小，并通过减少计算量来加快推理 (inference)速度。

剪枝主要分为两类：非结构化剪枝和结构化剪枝。

### 非结构化剪枝

非结构化剪枝作用于最细粒度：模型层内的单个权重 (weight)。最常用的方式是**基于幅度的剪枝**。核心思想很简单：绝对值较小的权重对网络输出的贡献较小，被认为不那么重要。

进行幅度剪枝，通常需要：

1. **定义稀疏目标：** 确定要移除的权重比例（例如，50%稀疏度意味着移除一半权重）。
2. **权重排序：** 收集模型中（或特定层内）的所有权重，并按其绝对值大小进行排序。
3. **确定阈值：** 找到与所需稀疏度对应的幅度阈值。例如，如果目标是50%稀疏度，阈值将是权重绝对值的中位数。
4. **创建掩码：** 创建一个与权重张量形状相同的二进制掩码。将低于阈值的权重对应的掩码元素设为0，高于阈值的设为1。
5. **应用掩码：** 在前向传播（以及微调 (fine-tuning)时的反向传播 (backpropagation)）期间，将权重与此掩码相乘，有效地将剪枝后的权重置零。

以下是一个PyTorch代码片段，说明了基于幅度为单个线性层创建掩码的核心思想：

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# 示例：一个线性层
layer = nn.Linear(100, 50)

# --- 幅度剪枝 ---
# 指定所需的稀疏度（例如，移除30%的权重）
amount_to_prune = 0.3

# 使用PyTorch的剪枝工具进行非结构化L1幅度剪枝
prune.l1_unstructured(layer, name="weight", amount=amount_to_prune)

# 剪枝通过“前向预钩子”应用。原始权重被保存。
# 检查剪枝后的权重（一些将为零）
print(layer.weight)

# 要使剪枝永久化（移除掩码并直接将权重置零）：
prune.remove(layer, 'weight')
print(layer.weight) # 现在包含永久零值

# 注意：在实践中，剪枝通常伴随微调。
# 掩码在微调期间保留，确保剪枝后的权重保持为零。
```

**非结构化剪枝的优点：**

- **高稀疏度潜力：** 通常可以在不显著损失精度的情况下移除大部分权重，尤其是在高度过参数 (parameter)化的模型中。
- **粒度细：** 只针对最不重要的单个连接。

**非结构化剪枝的缺点：**

- **不规则稀疏性：** 导致稀疏权重矩阵中的零不规则地分散。这种模式难以被标准硬件（GPU、TPU）和库（cuBLAS、cuDNN）有效加速，因为它们针对密集矩阵操作进行了优化。
- **专用硬件/软件：** 实现显著的推理 (inference)加速通常需要专门的硬件或软件库，以高效处理稀疏计算（例如，CSR/CSC等稀疏矩阵格式）。
- **元数据开销：** 存储非零元素的索引（稀疏掩码）会增加一些内存开销，部分抵消了移除权重带来的大小减小，尤其是在较低稀疏度时。

### 结构化剪枝

结构化剪枝不是移除单个权重 (weight)，而是移除整个、明确定义的参数 (parameter)块或组。这可能包括移除：

- **神经元/滤波器：** 权重矩阵中对应特定神经元的整个行或列。在卷积神经网络 (neural network)（CNN）中，这类似于移除整个滤波器。在Transformer中，这可能意味着移除前馈网络（FFN）层中的神经元。
- **注意力头：** 移除多头注意力 (multi-head attention)层中完整的注意力头。
- **层：** 在极端情况下，整个层也可能被移除。

移除结构的依据可以不同。它可能基于结构内权重的聚合幅度（例如，与神经元相关的权重的L2范数）、神经元在数据集上的平均激活值，或与结构对模型输出或损失的贡献相关的更复杂的度量。

```python
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

# 示例：一个线性层并剪枝“神经元”（输出通道）
layer = nn.Linear(100, 50) # 50个输出神经元

# --- 结构化剪枝（示例：剪枝神经元/输出通道） ---
# 假设我们想剪枝50个神经元中的10个（20%）
num_neurons_to_prune = 10

# 计算与每个输出神经元相关的权重的L2范数
# layer.weight的形状为 [输出特征数, 输入特征数] = [50, 100]
# 我们沿着输入维度（dim=1）计算范数
neuron_norms = torch.norm(layer.weight.data, p=2, dim=1)

# 找到范数最小的神经元的索引
threshold = torch.kthvalue(neuron_norms, k=num_neurons_to_prune).values
indices_to_prune = torch.where(neuron_norms <= threshold)[0]

# 使用PyTorch的结构化剪枝工具
# （剪枝整个输出通道）
# 我们指定对应输出通道的维度（dim=0）
prune.ln_structured(
    layer,
    name="weight",
    amount=num_neurons_to_prune,
    n=2,
    dim=0
)

# 同样，剪枝通过钩子应用。
# 检查权重——对应于剪枝神经元的整行
# 将变为零。
# print(layer.weight)

# 永久化
prune.remove(layer, 'weight')
# print(layer.weight)

# 注意：结构化剪枝后，层的输出维度
# 会实际改变。
# 后续层可能需要调整，或者剪枝后的模型需要
# 微调。
# 与非结构化剪枝不同，结构化剪枝通常在移除零值结构后，生成一个
# 真正更小、更密集的模型。
# 永久。
```

**结构化剪枝的优点：**

- **硬件友好：** 产生更小、更密集的矩阵或移除整个组件。标准硬件和库可以高效处理结果模型。
- **直接加速：** 通常能立即带来推理 (inference)加速，无需专门的稀疏计算库。
- **实现更简单：** 通过移除整个块来修改架构有时比管理稀疏矩阵格式更简单。

**结构化剪枝的缺点：**

- **粒度更粗：** 移除整个结构可能不够精确，并且可能比仅移除最不重要的单个权重对精度影响更大，尤其是在较高剪枝率下。
- **稀疏度限制较低：** 与非结构化剪枝相比，通常在精度开始显著下降之前，实现的整体稀疏度更低。

### 比较结构化与非结构化剪枝

结构化剪枝和非结构化剪枝的选择取决于具体的目标和约束：

| 特性 | 非结构化剪枝 | 结构化剪枝 |
| --- | --- | --- |
| **粒度** | 单个权重 (weight) | 神经元、注意力头、层、通道 |
| **稀疏模式** | 不规则 | 规则（更小、更密集的张量/层） |
| **硬件加速** | 困难（需要专门支持） | 更容易（使用标准密集操作） |
| **潜在稀疏度** | 更高 | 通常更低 |
| **实现** | 掩码管理、稀疏核 | 架构修改、密集核 |
| **精度影响** | 可能更低（在高稀疏度下） | 可能更高（在相同稀疏度下） |

### 剪枝与微调 (fine-tuning)

几乎所有剪枝方法的一个重要方面是需要**微调**。简单地移除权重 (weight)或结构通常会降低模型性能。为了恢复精度，剪枝后的模型必须在原始数据集或相关任务特定数据集上重新训练（微调）若干轮次。在此微调阶段，未剪枝的权重会调整以弥补移除的组件。

剪枝也可以迭代进行：剪枝一小部分权重，微调，再次剪枝，再次微调，依此循环。这种渐进过程通常比一次性剪枝模型大部分内容产生更好结果。

网络剪枝提供了一种有效方法来减少LLM的计算开销。尽管非结构化剪枝预示着更高的压缩比，但其实际好处通常取决于专用硬件或软件。结构化剪枝通过创建更小、更密集的模型，为在标准硬件上实现加速提供了更直接的途径，尽管这可能以牺牲较低的最大稀疏度为代价。这两种方法通常都需要仔细微调以恢复模型的预测能力。

## 参考资料

- [Learning both Weights and Connections for Efficient Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2015/file/ae0eb3eed39d2bcef4622b2499a05fe6-Paper.pdf) — Song Han, Jeff Pool, John Tran, William J. Dally (2015)
  Journal: Advances in Neural Information Processing Systems (NeurIPS); Publisher: MIT Press; Volume: 28; Pages: 1135-1143; DOI: [10.55989/nips.2015.0003](https://doi.org/10.55989/nips.2015.0003)
  介绍了基于幅度的非结构化剪枝，这是一种减少模型大小的核心技术。
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Subnetworks](https://arxiv.org/abs/1803.03635) — Jonathan Frankle, Michael Carbin (2019)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1803.03635](https://doi.org/10.48550/arXiv.1803.03635)
  提出了“彩票假说”，解释了为什么深度网络可以通过迭代剪枝和微调，在不损失准确性的前提下被剪枝成高度稀疏的子网络。
- [What is the State of Neural Network Pruning?](https://proceedings.mlsys.org/paper/2020/hash/d2ddea18f00665ce8623e36bd4e3c7c5-Abstract.html) — Davis Blalock, Jose Javier Gonzalez Ortiz, Jonathan Frankle, John Guttag (2020)
  Journal: Proceedings of Machine Learning and Systems; Volume: 2; Pages: 129-146; DOI: [10.48550/arXiv.2003.03033](https://doi.org/10.48550/arXiv.2003.03033)
  全面回顾了各种剪枝技术，包括结构化和非结构化方法，及其对效率的实际影响。
- [Pruning techniques - PyTorch 2.3 documentation](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html) — Torch Contributors (2019)
  Publisher: Torch Contributors
  PyTorch剪枝工具的官方文档，展示了非结构化和结构化剪枝方法的实际实现。
