# 低秩适配 (LoRA)

来源：[原文](https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-5-few-shot-adaptation-foundation-models/low-rank-adaptation-lora)

[返回章节目录](README.md) · [返回课程目录](../README.md)

对拥有数十亿参数 (parameter)的预训练 (pre-training)模型进行适配，会遇到一些特别的难题。尽管完全微调 (fine-tuning)效果好，但它需要为每个后续任务存储并管理模型的完整副本，这通常难以实现。此外，在有限的少样本数据上训练所有参数，可能导致过拟合 (overfitting)以及预训练权重 (weight)中宝贵知识的灾难性遗忘。低秩适配 (LoRA) 是一种高效且实用的参数高效微调 (PEFT) 技术，专门为解决这些问题而设计。

LoRA 的主要思想是，将预训练模型适配到特定任务所需的调整存在于一个低固有秩子空间中。LoRA 不修改层的整个高维权重矩阵 $W$（例如，在注意力或前馈网络中），而是建议使用低秩分解来表示权重的*变化* $\Delta W$。

### LoRA 的工作原理

考虑一个预训练 (pre-training)的权重 (weight)矩阵 $W_0 \in \mathbb{R}^{d \times k}$。在适配过程中，LoRA 保持 $W_0$ 不变，并引入两个较小的、可训练的“更新”矩阵：$B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$。其中，秩 $r$ 比原始维度 $d$ 和 $k$ 小很多（即 $r \ll \min(d, k)$）。对原始权重的更新由这些矩阵的乘积表示：


$$
\Delta W = BA
$$


使用这个适配后的权重矩阵 $W = W_0 + \Delta W$ 的层的前向传播可以表示为：


$$
h = W_0 x + \Delta W x = W_0 x + BAx
$$


重要的是，在适配过程中只有 $A$ 和 $B$ 的参数 (parameter)得到优化；原始权重 $W_0$ 保持不变。这大大减少了可训练参数的数量，从 $d \times k$ 减少到 $r \times (d + k)$。 $r$ 的典型取值范围是 4 到 64，使得可训练参数的数量比原始模型大小小几个数量级。

为了控制适配的幅度并保证其稳定，LoRA 通常会引入一个缩放因子 $\alpha$。组合后的权重矩阵变为 $W = W_0 + \frac{\alpha}{r} BA$。矩阵 $B$ 通常用零初始化，而 $A$ 则使用随机高斯分布初始化。这种初始化策略保证了在训练开始时 ($t=0$)，$\Delta W = BA$ 为零，这意味着适配过程从预训练模型的状态准确开始，随着 $A$ 和 $B$ 的学习逐渐引入任务相关的更新。

> LoRA 的一个图示。原始权重矩阵 $W_0$ 保持固定。适配通过低秩分解矩阵 $B$ 和 $A$ 进行学习，它们相乘并（经过缩放后）加到 $W_0$ 上，从而构成有效权重矩阵 $W$。只有 $B$ 和 $A$ 得到训练。

### LoRA 的优点

LoRA 为适配大型预训练 (pre-training)模型提供了多项优势：

1. **可训练参数 (parameter)的大幅减少：** 通过仅训练低秩矩阵 $A$ 和 $B$，LoRA 大幅降低了训练相关的计算成本和内存占用，通常比完全微调 (fine-tuning)减少 100 倍甚至 1000 倍。
2. **存储开销极小：** 对于每个适配过的任务，只需要存储小矩阵 $A$ 和 $B$（以及 $r$ 和 $\alpha$ 等配置信息）。这通常是兆字节级别，而不是完整模型副本所需的千兆字节，使得为许多不同任务保留适配变得可行。
3. **无额外推理 (inference)延迟：** 在推理时，通过计算 $W = W_0 + \frac{\alpha}{r} BA$，学习到的权重 (weight)更新 $BA$ 可以直接并入原始权重矩阵 $W_0$。这个合并后的权重 $W$ 随后可以用于推理，无需更改模型结构或引入额外的计算步骤。与适配器模块等方法相比，这是一个值得注意的优点，因为适配器模块会增加额外的层从而导致延迟。
4. **高效的任务切换：** 部署适配不同任务的模型，只需加载相应的 $A$ 和 $B$ 矩阵并替换之前的（或将其并入 $W_0$）即可，过程十分便捷。
5. **性能接近：** 尽管LoRA效率高，但在许多标准基准测试中，对于少样本适配任务，其性能已展现出与完全微调相近的水平，涵盖了不同方面（自然语言处理、视觉）的应用。

### 实现细节与超参数 (parameter) (hyperparameter)

在实现 LoRA 时，需要考虑几个选择：

- **目标模块：** LoRA 不一定应用于*所有*权重 (weight)矩阵。在 Transformer 模型中，通常的做法是将 LoRA 应用于自注意力 (self-attention)机制 (attention mechanism)中的权重矩阵（特别是查询 ($W_q$) 和值 ($W_v$) 投影矩阵是常选目标，有时也包括键 ($W_k$) 和输出 ($W_o$)）。将其应用于前馈网络 (FFN) 层也是可行的，但相对于参数增加而言，可能回报递减。根据实验结果或现有研究进行细致的选择是明智的。
- **秩 `r`：** 这是一个主要的超参数。更高的秩 $r$ 允许更强的适配能力（$\Delta W$ 的更大容量），但会增加可训练参数的数量。较低的 $r$ 参数使用更少，但可能限制模型有效适配的能力。4、8、16、32 等值是常见的初始选择，通常根据验证性能与参数数量的平衡进行调整。
- **缩放因子 `alpha`：** 这个超参数用于缩放 LoRA 更新 $BA$ 的影响。它在一定程度上像是适配矩阵的学习率。通常的做法是将 $\alpha$ 设置为首次尝试的秩 $r$ 的值，但它也可以独立调整。
- **初始化：** 如前所述，将 $B=0$ 和 $A$ 用小的随机值（例如高斯分布）初始化，可使适配过程从原始模型的行为平稳开始。

### 适配策略中的 LoRA

与完全微调 (fine-tuning)相比，LoRA 在可训练参数 (parameter)和存储方面带来巨大节省。与适配器模块相比，LoRA 通过允许权重 (weight)合并来避免引入推理 (inference)延迟。虽然与 MAML 等元学习算法（它们学习优化*快速适配*的初始化）不同，但 LoRA 本身提供了一种直接的*适配方法*。它着重于提高特定任务适配步骤的效率，而不是在元训练阶段学习一个适用于多任务的通用适配流程。不过，LoRA 可以被视为一种补充；LoRA 的高效性甚至可能通过减轻内循环更新的计算压力，使某些元学习方法对预训练 (pre-training)模型变得更实用。

总而言之，LoRA 为大型预训练模型的少样本适配提供了一种简单、有效且实用的方法。它能够大幅减少可训练参数和存储成本，同时保持性能且不引入推理延迟，使其成为参数高效微调 (PEFT) 领域一个很有用的工具。本章后面的实践部分将提供将 LoRA 用于适配预训练模型的具体操作指导。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685.pdf) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen (2022)
  Journal: International Conference on Learning Representations (ICLR 2022); Publisher: OpenReview.net; DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了LoRA方法，用于高效微调大型语言模型，详细阐述了其机制、优势和实验结果。
- [Parameter-Efficient Fine-Tuning of Large-Scale Pre-trained Language Models: A Survey](https://arxiv.org/pdf/2303.15647.pdf) — Ning Lou, Hongye Song, Wenxiao Shang, Xiao Liu, Ziyang Li, Yuxiao Dong, Xin Xu, Jing Chen, Yiqi Wang, Yu Zhang, Jiazeng Fang, Xiaoqing Zheng, and Jie Zhou (2023)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.2303.15647](https://doi.org/10.48550/arXiv.2303.15647)
  一篇综述论文，系统回顾了针对大型语言模型的各种参数高效微调方法，提供了详细的比较和分析。
- [PEFT: Parameter-Efficient Fine-Tuning library](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT库的官方文档，提供了实现LoRA等参数高效微调技术的实用指南和示例。

---

[上一节](02-%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E9%80%82%E9%85%8D%E5%99%A8%E6%A8%A1%E5%9D%97.md) · [下一节](04-Prompt%20Tuning%20%E5%92%8C%20Prefix%20Tuning.md)
