# 低秩适应（LoRA）：原理与运作

来源：[原文](https://apxml.com/zh/courses/introduction-to-llm-fine-tuning/chapter-4-parameter-efficient-fine-tuning-peft/lora-theory-and-operation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

低秩适应（LoRA）基于一个简单而高效的前提运作：它不直接修改预训练 (pre-training)模型的全部权重 (weight)，而是学习一小组改变。这种方法保持原始模型权重不变，并向模型的特定层注入一对可训练的低秩矩阵，通常在注意力机制 (attention mechanism)内。这意味着对于一个庞大的权重矩阵 $W_0$，我们不计算它的新值。相反，我们学习一个更新量 $\Delta W$，正向传播时使用的实际权重变为 $W_0 + \Delta W$。

LoRA 的高效性源于它表示更新矩阵 $\Delta W$ 的方式。它将更新限制为一个低秩矩阵，该矩阵可以分解为两个小得多的矩阵的乘积。这是基于这样的观察：为了让模型适应新任务所需的调整通常具有较低的“内在秩”，这意味着无需满秩矩阵就能捕获这些变化。

### 低秩分解的数学原理

考量一个来自预训练 (pre-training)模型的原始权重 (weight)矩阵 $W_0$，其维度为 $d \times k$。完全微调 (fine-tuning)方法会更新这个矩阵中所有 $d \times k$ 个参数 (parameter)。LoRA 通过将更新 $\Delta W$ 表示为两个较小矩阵的乘积来避免这种情况：一个维度为 $d \times r$ 的矩阵 $B$ 和一个维度为 $r \times k$ 的矩阵 $A$。秩 $r$ 是一个远小于 $d$ 或 $k$ 的超参数 (hyperparameter)。

该层的修改后的正向传播变为：


$$
h = W_0x + \Delta W x = W_0x + BAx
$$


这里，$x$ 是该层的输入，$h$ 是输出。训练期间，$W_0$ 保持冻结，只有矩阵 $A$ 和 $B$ 中的参数通过反向传播 (backpropagation)进行更新。

这种分解能大幅减少可训练参数。原始矩阵 $W_0$ 有 $d \times k$ 个参数。LoRA 矩阵 $A$ 和 $B$ 合计有 $(d \times r) + (r \times k) = r(d+k)$ 个参数。对于 Transformer 中典型的方阵（其中 $d=k$），可训练参数的数量仅为 $2dr$ 而非 $d^2$。

举个例子，如果我们有一个大小为 $4096 \times 4096$ 的权重矩阵（超过1600万个参数），并且我们使用 LoRA 秩 $r=8$，则可训练参数的数量变为 $2 \times 4096 \times 8 = 65,536$。这使得该单层的参数减少了超过 99.8%，从而使训练过程更易于处理。

> 该图展示了 LoRA 层如何处理输入。输入 `x` 经过原始的冻结权重矩阵 `W₀`（主路径）。同时，它并行通过可训练的适配器矩阵 `A` 和 `B`。两条路径的输出随后相加，生成最终的层输出 `h`。

### 超参数 (parameter) (hyperparameter)

在实现 LoRA 时，您会遇到一些影响微调 (fine-tuning)过程行为和表现的配置参数。

- **秩 `r`**：秩是最有影响力的超参数。它决定了可训练参数的数量以及 LoRA 适配器的表达能力。较小的秩（例如 4、8）会导致更少的参数和更快的训练，但可能无法获取复杂任务所需的所有信息。较大的秩（例如 32、64）提供更大的容量，但代价是增加 VRAM 使用和减慢训练速度。`r` 的选择是性能与效率之间的直接权衡。
- **Alpha `α`**：LoRA 实现通常包含一个缩放超参数 `alpha`。适配器的输出在添加到基础模型的输出之前，会按 $\alpha/r$ 进行缩放。这意味着正向传播在技术上是 $h = W_0x + (\alpha/r)BAx$。这种缩放有助于使适配器的影响正常化，防止在更改秩 `r` 时更新过大或过小。常见的默认设置是将 `alpha` 等于 `r`，这将缩放因子简化为 1。然而，独立调整 `alpha` 有时可以带来更好的结果。
- **目标模块**：您还必须决定将 LoRA 应用到模型的哪些层。尽管可以将其应用于任何线性层，但最常将其应用于自注意力 (self-attention)块内的查询 (`q_proj`) 和值 (`v_proj`) 投影矩阵。这些层被认为对模型行为适应新数据模式具有很大影响。将其应用于更多层会增加可训练参数的数量，但可能提供更好的性能。

### 实际益处

LoRA 的设计提供了比完全微调 (fine-tuning)更多的实际益处：

1. **内存大幅降低**：通过冻结基础模型，LoRA 消除了存储数十亿参数 (parameter)的梯度和优化器状态的必要。这大幅降低了 VRAM 需求，使您能够在广泛可用的硬件上微调非常大型的模型。
2. **快速高效的训练**：仅训练总参数的一小部分使得每个训练步骤快得多。
3. **模块化和便携式适配器**：训练好的适配器只是小的矩阵 $A$ 和 $B$，使得检查点文件只有几兆字节大小。这使您能够维护基础模型的一个副本，并为各种下游任务切换不同的轻量级适配器，这比为每个任务保存一个完整的微调模型要节省得多存储空间。
4. **无推理 (inference)延迟**：对于部署，LoRA 适配器权重 (weight)可以通过计算 $W' = W_0 + BA$ 直接合并到基础模型权重中。合并后的模型是一个标准的 Transformer 模型，没有额外的层或计算开销，这意味着与原始模型相比，它在推理过程中没有延迟损失。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  这篇基础性论文介绍了LoRA，详细阐述了其理论、数学公式和实验验证。
- [Parameter-Efficient Fine-Tuning (PEFT)](https://huggingface.co/docs/peft/en/index) — Hugging Face (2024)
  Publisher: Hugging Face
  提供使用Hugging Face库实现LoRA及其他PEFT方法的实践指南，包含超参数的细节。

---

[上一节](01-%E5%8F%82%E6%95%B0%E9%AB%98%E6%95%88%E5%BE%AE%E8%B0%83%E6%A6%82%E8%BF%B0.md) · [下一节](03-%E4%BD%BF%E7%94%A8PEFT%E5%BA%93%E5%AE%9E%E7%8E%B0LoRA.md)
