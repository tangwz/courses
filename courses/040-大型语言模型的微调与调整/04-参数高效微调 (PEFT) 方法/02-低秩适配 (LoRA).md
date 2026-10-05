# 低秩适配 (LoRA)

来源：[原文](https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-4-parameter-efficient-fine-tuning/lora-introduction)

[返回章节目录](README.md) · [返回课程目录](../README.md)

全量微调 (fine-tuning)虽然有效，但需要修改大型模型中的每一个权重 (weight)，这会导致很大的计算量和内存需求。低秩适配（LoRA）提供了一种巧妙且实用的方法，可以在许多适配任务中大幅减少可训练参数 (parameter)的数量，同时性能没有明显损失。

### 低秩假设

LoRA 基于一个发现：将预训练 (pre-training)模型适配到特定任务所需的改变通常具有低的“内在秩”。这意味着，尽管预训练权重 (weight) $W_0$ 占据高维空间 (high-dimensional space)，但专用化所需的调整 $\Delta W$ 位于低得多的子空间中。LoRA 不再学习完整的 $\Delta W$ 矩阵（它与 $W_0$ 具有相同的维度），而是使用两个更小的低秩矩阵来近似它。

### 机制：分解权重 (weight)更新

考虑预训练 (pre-training)模型中的一个权重矩阵 $W_0$，例如自注意力 (self-attention)机制 (attention mechanism)或前馈网络层中的权重矩阵。在全量微调 (fine-tuning)期间，我们学习一个更新 $\Delta W$，使得适配后的权重变为 $W = W_0 + \Delta W$。

LoRA 提出将更新 $\Delta W$ 表示为两个更小矩阵 $B$ 和 $A$ 的乘积：


$$
\Delta W = BA
$$


这里，如果 $W_0$ 的维度是 $d \times k$，那么 $B$ 的维度是 $d \times r$，而 $A$ 的维度是 $r \times k$。超参数 (parameter) (hyperparameter) $r$ 是分解的秩，重要的是，选择 $r$ 要远小于 $d$ 或 $k$（即 $r \ll \min(d, k)$）。

在 LoRA 微调期间：

1. 原始权重 $W_0$ 被**冻结**；优化器不会更新它们。
2. 只有低秩矩阵 $A$ 和 $B$ 中的参数是**可训练的**。

经 LoRA 修改的层的正向传播变为：


$$
h = W x = (W_0 + \Delta W) x = W_0 x + BAx
$$


$x$ 是层的输入，$h$ 是输出。注意，通过 $W_0$ 的原始路径得以保留，并且学习到的适配 $BAx$ 作为修改添加。

> LoRA 修改层的流程图。使用冻结权重 $W_0$ 的原始路径通过可训练的低秩矩阵 $A$ 和 $B$ 的并行路径得到补充。秩 $r$ 远小于原始矩阵的维度 ($d, k$)。

### 参数 (parameter)效率

可训练参数的减少是显著的。LoRA 不再训练 $\Delta W$ 的 $d \times k$ 个参数，而是只训练 $A$ 和 $B$ 中的参数，这些参数总计为 $r \times k + d \times r = r(d+k)$。

例如，考虑一个尺寸为 $4096 \times 4096$ 的权重 (weight)矩阵 $W_0$。

- 全量微调 (fine-tuning)需要更新 $4096 \times 4096 \approx 1670$ 万个参数。
- 对于这个特定矩阵，使用秩 $r=8$ 的 LoRA 只需要更新 $8 \times (4096 + 4096) = 65,536$ 个参数。

这表示该层的可训练参数减少了 99% 以上，大幅降低了梯度和优化器状态（如 Adam 动量）所需的内存。

### 初始化与缩放

为确保微调 (fine-tuning)过程从预训练 (pre-training)模型的状态平稳开始，矩阵 $A$ 和 $B$ 会被仔细初始化。

- $A$ 通常用小的随机值初始化（例如，高斯初始化）。
- $B$ 初始化为全零。

这使得初始更新 $\Delta W = BA$ 等于零，这意味着在训练开始时 $(W_0 + BA)x = W_0 x$。模型开始时表现与基础模型相同，并逐渐学习适配。

此外，LoRA 的输出通常会通过一个因子 $\frac{\alpha}{r}$ 进行缩放，$\alpha$ 是另一个超参数 (parameter) (hyperparameter)：


$$
h = W_0 x + \frac{\alpha}{r} BAx
$$


$\alpha$ 作用类似于适配的学习率，控制 LoRA 矩阵引入的改变幅度相对于秩 $r$ 的大小。$\alpha$ 的常见值可能是 $r$、$2r$ 或仅仅是 1，这取决于所选的 $r$ 和具体任务。调整 $\alpha$ 对于获得理想性能可能很重要。

### 实现考量

- **目标模块：** LoRA 通常不会应用于 LLM 中的每一个权重 (weight)矩阵。它最常应用于自注意力 (self-attention)层中的查询 ($W_q$) 和值 ($W_v$) 投影矩阵，因为这些矩阵常常被发现对适配有效。有时它也会应用于键 ($W_k$)、输出 ($W_o$) 投影，甚至前馈模块中的层。`target_modules` 的选择是使用 Hugging Face PEFT 等库时的配置选项。
- **秩选择：** 秩 $r$ 是一个主要的超参数 (parameter) (hyperparameter)。常见值范围从 4 到 64。较高的秩可能允许更具表现力的适配，但会增加参数数量和计算成本。理想的 $r$ 通常取决于适配任务的复杂性和特定数据集。它通常通过实验确定。
- **推理 (inference)合并：** LoRA 的一个重要优点是，在训练之后，学习到的权重 $A$ 和 $B$ 可以合并回原始权重矩阵：$W_{merged} = W_0 + BA$。这意味着您可以计算一次最终的权重矩阵，并进行部署，而无需在推理期间使用单独的 $A$ 和 $B$ 矩阵。因此，与添加额外层的方法（如 Adapters）不同，LoRA 与原始模型或全量微调 (fine-tuning)模型相比，通常**不会引入额外的推理延迟**。

### LoRA 的优点

1. **大幅参数 (parameter)减少：** 相比全量微调 (fine-tuning)，训练所需的参数少得多，大幅降低了梯度和优化器状态的 VRAM 需求。
2. **更快的训练：** 更少的参数通常会带来更快的训练迭代，尽管整体训练时间取决于数据集大小和收敛速度。
3. **更低的存储成本：** 每个任务只需保存小的 LoRA 权重 (weight)（$A$ 和 $B$），而不是模型的完整副本。这使得存储多个任务专用适配的效率很高。
4. **高效的任务切换：** 可以加载基础模型，并快速切换不同的 LoRA 适配器权重，从而改变模型的专门行为，而无需重新加载整个模型。
5. **无推理 (inference)延迟（合并后）：** 通过在训练后合并权重，LoRA 避免了在推理期间引入额外的计算步骤。
6. **有竞争力的性能：** 在许多下游任务上，其性能通常与全量微调相当，特别是当适配不需要基础模型能力发生剧烈变化时。

### 潜在缺点

1. **超参数 (parameter) (hyperparameter)敏感性：** 性能可能取决于秩 $r$、缩放因子 $\alpha$ 和目标模块集合的选择。找到理想配置可能需要实验。
2. **表达能力限制：** 尽管对许多任务有效，但如果所需的改变 $\Delta W$ 确实具有较高的内在秩，低秩约束可能会限制模型的适配能力。在这种情况下，全量微调 (fine-tuning)可能会带来更好的结果。
3. **潜在干扰：** 将 LoRA 更新同时应用于模型的许多不同部分，可能导致复杂的相互影响，这些相互影响比直接更新完整参数更难优化。

LoRA 表现突出，是一种非常有效且广泛采用的 PEFT 方法。它的简洁、高效以及出色的经验性能，使其成为在计算资源受限或需要管理多个任务专用模型时适配大型模型的首选技术。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍 LoRA 的基础论文，详细阐述了其机制、理论基础（低秩假设）和实验结果。
- [PEFT (Parameter-Efficient Fine-tuning) Documentation: LoRA Configuration](https://huggingface.co/docs/peft/en/package_reference/lora) — Hugging Face (2024)
  官方文档，提供了 Hugging Face PEFT 库中 LoRA 的实际实现细节和配置选项。

---

[上一节](01-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E6%80%A7%E7%9A%84%E5%8E%9F%E7%90%86.md) · [下一节](03-%E9%87%8F%E5%8C%96%E4%BD%8E%E7%A7%A9%E9%80%82%E9%85%8D%20%28QLoRA%29.md)
