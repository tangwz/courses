---
course: "llm-compression-acceleration"
chapter: "parameter-efficient-fine-tuning-peft"
lesson: "low-rank-adaptation-lora"
sourceId: 4389
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-5-parameter-efficient-fine-tuning-peft/low-rank-adaptation-lora"
title: "低秩适应（LoRA）"
description: "详细了解LoRA的数学公式、实现细节及其变体。"
order: 4
plots: ["plots/4389-0.json"]
sourceHash: "8e50b192403689c7eaec0f9d35236a0a85d0a17bdbac0e10931563b3348c14e4"
sourceCorrections: []
---

基于本章前面讨论的对高效适应方法的需求，低秩适应（LoRA）作为一种非常有效且广泛采用的参数 (parameter)高效微调 (fine-tuning)（PEFT）技术脱颖而出。LoRA不是对大型语言模型的整个预训练 (pre-training)权重 (weight)集 $W_0$ 进行微调，而是冻结 $W_0$，并引入少量可训练参数，这些参数代表了针对特定下游任务的权重*变化*，即 $\Delta W$。

LoRA的核心思想是，模型适应期间的权重更新矩阵 $\Delta W$ 具有低的内在秩。也就是说，尽管 $\Delta W$ 与 $W_0$ 具有相同的维度，但其信息可以被秩远低于其自身的矩阵有效捕获或近似。

### 数学公式

LoRA将更新矩阵 $\Delta W$ 分解为两个较小的低秩矩阵 $A$ 和 $B$ 的乘积。对于一个预训练 (pre-training)权重 (weight)矩阵 $W_0 \in \mathbb{R}^{d \times k}$，更新 $\Delta W \in \mathbb{R}^{d \times k}$ 表示为：


$$
\Delta W = B A
$$


其中，$B \in \mathbb{R}^{d \times r}$ 且 $A \in \mathbb{R}^{r \times k}$，其中秩 $r$ 是一个超参数 (parameter) (hyperparameter)，满足 $r \ll \min(d, k)$。在使用LoRA进行微调 (fine-tuning)时，原始权重 $W_0$ 保持冻结，并且只训练矩阵 $A$ 和 $B$ 的参数。

经过该层的修改后的前向传播包含了这个低秩更新。如果原始层的计算是 $h = W_0 x$，则经LoRA适应的层计算为：


$$
h = W_0 x + \Delta W x = W_0 x + B A x
$$

> 一个图示，说明了LoRA更新机制。输入 `x` 通过原始冻结权重 `W₀`，并同时通过可训练的低秩矩阵 `A` 和 `B`。输出经过缩放并相加，以产生最终输出 `h`。

### 初始化与缩放

正确的初始化对确保训练稳定很重要。矩阵 $A$ 通常使用随机高斯值进行初始化，而矩阵 $B$ 则初始化为全零。这确保了在训练开始时，$\Delta W = BA = 0$，这意味着适应后的模型在开始时具有与预训练 (pre-training)模型 $W_0$ 完全相同的性能。

LoRA通常包含一个应用于更新 $\Delta W$ 的缩放因子 $\alpha$。前向传播变为：


$$
h = W_0 x + \frac{\alpha}{r} B A x
$$


其中，$\alpha$ 是一个常量超参数 (parameter) (hyperparameter)。将 $BA$ 的输出通过 $\alpha/r$ 进行缩放，有助于归一化 (normalization)组合输出的幅度，从而减少了在改变秩 $r$ 时需要大幅调整其他超参数的需求。一种常见做法是将 $\alpha$ 设置为秩 $r$，这在初始时实际上使缩放因子为1，但将其解耦可以进行进一步的调优。

### 实现细节与选择

#### 目标层

LoRA最常应用于Transformer模型注意力机制 (attention mechanism)中的权重 (weight)矩阵，特别是查询 ($W_q$)、键 ($W_k$)、值 ($W_v$) 和输出 ($W_o$) 投影矩阵。将LoRA应用于这些矩阵在实践中显示出明显效果。它也可以应用于前馈网络（FFN）层中的权重矩阵，有时根据任务和模型的不同，可以产生进一步的改进。选择哪些层使用LoRA进行适应是一个设计选择，它影响参数 (parameter)效率和任务性能之间的权衡。

#### 参数效率

LoRA引入的可训练参数数量远小于原始参数数量。对于一个 $d \times k$ 权重矩阵 $W_0$，原始参数数量是 $dk$。LoRA更新 $BA$ 引入 $dr + rk = r(d+k)$ 个参数（不考虑偏置 (bias)）。由于 $r$ 通常远小于 $d$ 和 $k$，这种减少是明显的。例如，将一个 $4096 \times 4096$ 矩阵（$16.7$ 百万参数）使用 $r=8$ 进行适应，仅增加了 $8 \times (4096 + 4096) \approx 65,536$ 个可训练参数，对于该特定矩阵，参数减少了250倍以上。

#### 秩选择 ($r$)

秩 $r$ 是LoRA中最重要的超参数 (hyperparameter)。它直接控制着适应的能力以及可训练参数的数量。

- **低秩 ($r$):** 最大限度地提高参数效率，并降低训练期间的计算成本。然而，过低的秩可能缺乏有效适应复杂任务所需的表达能力，可能导致欠拟合 (underfitting)。
- **高秩 ($r$):** 增加可训练参数的数量，可能使得模型更好地适应目标任务。然而，这会带来更高的训练计算开销，并削弱参数效率优势。它还可能增加对微调 (fine-tuning)数据过拟合 (overfitting)的风险。

秩 $r$ 的常用值范围从4到64。最优秩通常取决于具体的任务、数据集大小以及正在适应的层。通常需要通过实证评估来确定最佳值。



![LoRA：秩 (r) 与参数和性能的关系](plots/4389-0.json)



> LoRA秩 ($r$)、引入的可训练参数数量（对于单个4096x4096层）与任务性能之间的关系。增加秩会线性增加参数，但通常在性能改进方面产生收益递减。

### 推理 (inference)时的权重 (weight)合并

LoRA的一个重要优点是，低秩更新可以在推理时重新并入原始权重矩阵。训练完成后，组合权重矩阵 $W$ 可以计算为：


$$
W = W_0 + \frac{\alpha}{r} B A
$$


这个合并后的矩阵 $W$ 可以替代模型中的 $W_0$。因此，经LoRA适应的模型的推理延迟与原始预训练 (pre-training)模型相同。在部署时没有额外的计算或参数 (parameter)，这与Adapter等引入持久附加层的方法不同。这使得LoRA对于推理速度非常重要的生产环境非常有吸引力。

### 优点总结

- **高参数 (parameter)效率：** 与完全微调 (fine-tuning)相比，可训练参数数量减少了几个数量级。
- **降低存储成本：** 每个任务只需存储小矩阵 $A$ 和 $B$（以及可能的缩放因子 $\alpha$），共享大型冻结的 $W_0$。这在管理多个任务的适应时，显著降低了存储需求。
- **无推理 (inference)开销：** 将 $BA$ 合并到 $W_0$ 后，模型结构和计算保持不变，推理时不会产生额外延迟。
- **有效性能：** 通常在各种下游任务上达到与完全微调相近的性能。
- **轻松任务切换：** 将模型适应到不同任务，只需简单地替换相应的 $A$ 和 $B$ 权重 (weight)矩阵。

LoRA为适应大型预训练 (pre-training)模型提供了一种强大而高效的机制。其数学简洁性、实证效果以及零推理开销等实用优点，使其成为PEFT领域的基础技术。理解LoRA也为例如QLoRA等更高级的变体打下了基础，QLoRA将LoRA与量化 (quantization)结合，在微调过程中实现更大的内存节省。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: arXiv preprint arXiv:2106.09685; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  这篇基础论文介绍了LoRA，详细阐述了其数学公式、经验有效性以及在大型语言模型参数高效微调方面的优势。
- [QLoRA: Efficient Finetuning of Quantized LLMs on Consumer GPUs](https://arxiv.org/abs/2305.14314) — Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (2023)
  Journal: arXiv preprint arXiv:2305.14314; DOI: [10.48550/arXiv.2305.14314](https://doi.org/10.48550/arXiv.2305.14314)
  这篇论文介绍了QLoRA，一种将LoRA与量化结合的优化技术，旨在以更少的内存消耗高效微调大型语言模型。
- [PEFT: Parameter-Efficient Fine-tuning of Foundation Models](https://huggingface.co/docs/peft/en/index) — Hugging Face (2023)
  Hugging Face PEFT库的官方文档，提供了实现LoRA和其他参数高效微调技术的实用指南和代码示例。
