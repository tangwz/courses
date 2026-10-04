---
course: "fine-tuning-adapting-large-language-models"
chapter: "optimization-deployment-considerations"
lesson: "post-tuning-pruning"
sourceId: 3089
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations/post-tuning-pruning"
title: "微调后优化：剪枝"
description: "移除微调模型中冗余参数的方法。"
order: 4
plots: []
sourceHash: "494f0f3d1e30b5ecf63526529452f84f5a7520d041f6de7a148872e3d4f041e9"
sourceCorrections: []
---

“量化 (quantization)等方法侧重于通过改变模型权重 (weight)的精度来减少内存占用，而剪枝则采取另一种方法：它实际移除那些被认为不那么重要的参数 (parameter)，旨在创建更小且可能更快的模型。在投入大量资源微调 (fine-tuning)大型语言模型后，你可能会发现，尽管得到的模型是专门化的，但对于你的部署限制来说仍然过大或过慢。剪枝提供了一种在微调后精简这些模型的途径，使它们在应用中更具实用性。”

剪枝的核心思想是，在一个大型、通常参数过多的神经网络 (neural network)中，并非所有参数都对性能有相同的作用。通过找出并去除冗余或不那么显著的参数，我们可以减小模型的大小和计算需求，理想情况下对目标任务的准确性影响很小。

### 非结构化与结构化剪枝

剪枝技术主要分为两大类，它们在参数 (parameter)移除方式以及对硬件加速的影响上差异显著：

#### 非结构化剪枝

这是根据某些标准（通常是权重 (weight)大小）移除模型层中单个权重的过程。其基本假设是，绝对值较小的权重对网络输出的贡献较小，可以在不导致显著性能下降的情况下被移除。

- **方法**：一种常见的方法是幅度剪枝。选择一个稀疏度目标（例如，50% 稀疏度），并确定一个阈值 $\theta$，使得移除所有满足 $|w_{ij}| < \theta$ 的权重能达到此目标。这些选定的权重随后被永久设置为零。
- **优点**：可以潜在地实现非常高的稀疏度（许多权重被置零），同时保持模型准确性，特别是在一个短暂的再训练阶段之后。
- **缺点**：生成的权重矩阵呈现不规则的稀疏模式。GPU和TPU等标准硬件针对密集矩阵运算进行了优化。加速非结构化稀疏模型的推理 (inference)通常需要能够有效处理这些稀疏格式的专用硬件或软件库（如NVIDIA的cuSPARSE）。没有这种支持，参数数量的减少可能无法转化为显著的延迟改善。

#### 结构化剪枝

与移除单个权重不同，结构化剪枝以规则模式移除整组参数。这可能涉及移除：

- **神经元**：权重矩阵中对应于特定神经元的整行/整列。
- **注意力头**：Transformer层中完整的注意力头。
- **滤波器/通道**：卷积层中的整个滤波器（在纯Transformer LLM中较不常见，但在多模态 (multimodal)背景下有关联）或线性层中的等效结构。
- **层**：整个层（一种非常粗粒度的结构化剪枝形式）。
- **方法**：计算这些结构的重要性得分（例如，基于结构内权重的L2范数、平均激活幅度或梯度信息）。重要性得分最低的结构被完全移除。
- **优点**：生成的模型架构保持密集（或变成更小的密集架构）。这意味着剪枝后的模型通常可以在标准硬件上高效执行，无需专用稀疏计算库，从而更可预测地减少内存使用和延迟。
- **缺点**：移除整个结构对模型学到的表示可能比移除单个权重造成更大的干扰。因此，与非结构化剪枝相比，在移除相同数量参数的情况下，结构化剪枝可能导致准确性下降幅度更大，可能需要更广泛的再训练。

> 非结构化幅度剪枝（单个低幅度权重被置零）与结构化神经元剪枝（代表一个神经元连接的整列被移除）的比较。灰色单元格表示原始权重，红色表示被剪枝的元素。

### 剪枝策略与重要性衡量

如何以及何时进行剪枝很要紧：

- **一次性剪枝**：在初始微调 (fine-tuning)完成后，一次性应用剪枝标准。这种方法计算成本较低，但可能显著影响准确性。对剪枝后的模型进行随后的短期微调阶段（有时称为“再训练”或“精细剪枝”）通常对于通过让剩余权重 (weight)适应来恢复性能非常重要。
- **迭代剪枝**：这涉及剪枝和再训练的循环。你可能会剪枝一小部分权重，再训练模型几个周期，然后再次剪枝、再训练，以此类推，直到达到所需的稀疏度。这种渐进过程通常比一次性剪枝更好地保持准确性，但需要更多计算。

为了确定*剪枝什么*，使用了各种重要性标准：

- **幅度**：最简单且通常出奇有效的方法，特别是非结构化剪枝。假设绝对值较小的参数 (parameter)不那么重要。
- **梯度信息**：使用在短暂的前向/后向传播过程中计算的梯度来评估损失对移除参数或结构的敏感度。移除后导致损失变化最小（或梯度幅度最小）的参数是剪枝的候选。
- **基于激活**：分析神经元产生或通过权重的激活值。持续低激活的结构可能不那么重要。
- **敏感性分析**：更正式地衡量当特定参数或结构被暂时移除或置零时，模型损失或输出的变化。

### 实际实现考量

在对微调 (fine-tuning)后的LLM应用剪枝时，请考虑以下几点：

1. **稀疏度目标**：确定所需的稀疏度（例如，30%、50%、70%）。此选择涉及直接权衡：更高的稀疏度意味着更小的模型和可能更快的推理 (inference)，但通常以降低准确性为代价。此目标应由你的特定部署限制（内存、延迟）和可接受的性能阈值指导。
2. **再训练计划**：如果执行一次性剪枝或迭代剪枝，请仔细安排再训练阶段。这通常涉及使用比初始微调更低的学习率和更短的训练时间，刚好足以帮助网络适应被移除的参数 (parameter)。
3. **硬件兼容性**：记住，非结构化剪枝的益处在很大程度上依赖于专用软件/硬件。结构化剪枝通常在标准CPU和GPU上提供更可靠的加速和内存节省，因为它会产生更小、更密集的运算。
4. **与其他技术的结合**：剪枝可以与量化 (quantization)等其他优化方法结合使用。常见的顺序是先剪枝模型，然后量化剩余权重 (weight)，这可能实现更大的压缩和加速。
5. **工具**：虽然PyTorch等框架提供基本的剪枝工具（例如，`torch.nn.utils.prune`），但要将它们有效地应用于复杂的Transformer架构，特别是结构化剪枝，可能需要自定义实现或研究社区中出现的专用库（例如，专注于Transformer压缩的库）。务必查阅你所选框架及其可用扩展的文档和功能。

以下是使用PyTorch类似语法的非结构化幅度剪枝示例：

```python
import torch
import torch.nn.utils.prune as prune

# 假设 'model' 是你微调后的 Transformer 模型
# 假设 'module' 是一个特定层，例如 model.encoder.layer[0].attention.self.query

# 1. 定义剪枝方法（幅度剪枝）
pruning_method = prune.L1Unstructured # 或 prune.RandomUnstructured 等

# 2. 定义要剪枝的参数和稀疏度
parameters_to_prune = [(module, 'weight')]
sparsity_level = 0.5 # 目标稀疏度 50%

# 3. 应用剪枝（添加前向钩子和掩码参数）
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=pruning_method,
    amount=sparsity_level,
)

# 4. 使剪枝永久化（移除钩子，直接将权重置零）
# 在保存或部署剪枝后的模型之前，此步骤很重要
prune.remove(module, 'weight')

# 5. （推荐）对剪枝后的模型进行短暂微调
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
# train_model(model, dataloader, optimizer, num_epochs=1) # 短期再训练阶段
```

> 应用全局非结构化幅度剪枝到层权重矩阵的PyTorch代码。需要后续的微调步骤以获得最佳结果。

### 总结

剪枝是一种重要技术，用于减小微调 (fine-tuning)后LLM的大小并可能降低推理 (inference)延迟。通过移除不那么关键的参数 (parameter)，无论是单个（非结构化）还是成组（结构化），你可以创建更易于部署的模型。结构化剪枝通常在常规硬件上提供更实用的加速，而非结构化剪枝可能实现更高的稀疏度。方法选择、稀疏度目标以及再训练的必要性在很大程度上取决于具体模型、任务和部署环境。评估模型压缩、推理速度和任务性能之间的权衡，对于在微调后成功应用剪枝非常重要。

## 参考资料

- [Learning both Weights and Connections for Efficient Neural Networks](https://arxiv.org/abs/1506.02626) — Song Han, Jeff Pool, John Tran, William J. Dally (2015)
  Journal: Advances in Neural Information Processing Systems; Volume: 28; Pages: 1135-1143; DOI: [10.48550/arXiv.1506.02626](https://doi.org/10.48550/arXiv.1506.02626)
  提出了基于幅值的权重剪枝这一基础方法及其在减小模型尺寸方面的有效性。
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Subnetworks](https://link.springer.com/chapter/10.1007/978-3-030-22137-7_32) — Jonathan Frankle, Michael Carbin (2019)
  Journal: International Conference on Learning Representations (ICLR); Publisher: Springer, Cham; Volume: 11576; Pages: 129-140; DOI: [10.1007/978-3-030-22137-7_32](https://doi.org/10.1007/978-3-030-22137-7_32)
  提出了在随机初始化的网络中存在稀疏子网络，这些子网络经过训练可以达到与完整网络相同的性能的假设。
- [Are Sixteen Heads Really Better than One? Multi-Head Attention Is a Bag-of-Experts](https://arxiv.org/abs/1905.09424) — Paul Michel, Omer Levy, Graham Neubig (2019)
  Journal: International Conference on Learning Representations (ICLR); Publisher: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1905.09424](https://doi.org/10.48550/arXiv.1905.09424)
  研究了Transformer模型中多头注意力机制的冗余性，提出了注意力头结构化剪枝的方法。
- [torch.nn.utils.prune](https://pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html) — PyTorch Documentation (2023)
  PyTorch官方文档，介绍了`torch.nn.utils.prune`模块，提供了应用各种剪枝功能。
