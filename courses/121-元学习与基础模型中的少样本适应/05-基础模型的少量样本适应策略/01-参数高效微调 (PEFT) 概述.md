# 参数高效微调 (PEFT) 概述

来源：[原文](https://apxml.com/zh/courses/meta-learning-foundation-models/chapter-5-few-shot-adaptation-foundation-models/peft-overview)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将完全微调 (fine-tuning)甚至复杂的元学习策略直接应用于拥有数十亿参数 (parameter)的基础模型，虽然强大，但带来了显著的计算难题。训练需要大量的GPU内存和时间，并且为每个专门任务存储整个模型的独特副本变得不切实际。参数高效微调 (PEFT) 提供了一个很有吸引力的选择，提供了一系列专门为高效调整这些大型模型而设计的方法。

PEFT方法基于一个基本原则：它们不修改预训练 (pre-training)模型的所有参数，而是专注于调整一小部分参数，或者引入少量新的、可训练的参数。原始基础模型的绝大部分权重 (weight)在调整过程中保持不变。

目标是获得与完全微调相当的调整性能，特别是在少样本情况下，但计算开销大大降低。这种高效性体现在几个主要优点上：

- **计算需求降低：** 与更新整个模型相比，训练PEFT方法所需的GPU内存和计算时间大幅减少。这使得在性能较低的硬件上进行调整成为可能，或允许更快的迭代周期。
- **存储成本更低：** 由于每个任务只需要存储少量修改或新增的参数，存储占用空间大幅减少。不必为每个任务保存多GB的模型检查点，通常只需保存几MB的PEFT参数。这有助于从单一基础模型部署和管理大量针对特定任务的模型。
- **潜在缓解灾难性遗忘：** 通过保持预训练模型的核心参数不变，PEFT有时能比完全微调更好地保留预训练期间学到的通用能力，而完全微调有覆盖这些知识的风险。
- **具有竞争力的性能：** 对于许多下游任务和基准测试，PEFT方法已表现出接近甚至有时胜过完全微调的性能，特别是在目标任务的标注数据稀缺时。

> 调整方法的比较。完全微调修改所有参数。元学习通常侧重于为快速调整寻找一个好的初始化。PEFT冻结基础模型，只微调少量参数。

PEFT包含多种策略，这些策略可以根据它们如何实现参数高效性进行大致分类。有些方法，例如Adapter，在预训练模型的层中插入小的、新的神经网络 (neural network)模块。其他方法，例如低秩适配 (LoRA)，对某些权重矩阵进行重新参数化，以学习低秩更新。像Prompt Tuning这样的技术通过学习连续输入嵌入 (embedding)来修改模型的行为，而不是直接改变权重。

将PEFT视为一种补充工具集会很有帮助，它并非元学习的直接竞争者，不会使其失效，而是专注于非常大型模型调整步骤的*计算效率*。元学习解决的是从少量例子中*快速学习调整*的问题，通常是找到最优初始点或更新规则。PEFT确保*调整过程本身*是可行的。正如我们稍后将讨论的，这些方法并非互斥，有时可以结合使用。

通常，PEFT过程包括获取一个带有参数 $\theta_{pre}$ 的预训练基础模型，并将其冻结，然后引入一组小得多的可训练参数 $\phi$，参数数量 $|\phi|$ 比 $|\theta_{pre}|$ 小几个数量级。只有这些参数 $\phi$ 使用来自特定下游任务的数据进行优化。

接下来的章节将详细审视特定且广泛使用的PEFT技术，包括Adapter模块、LoRA和Prompt/Prefix Tuning，并分析它们的机制和权衡。我们还将直接比较它们的特点与之前讨论的元学习策略。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (2021)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.2106.09685](https://doi.org/10.48550/arXiv.2106.09685)
  介绍了低秩适应（LoRA），这是一种广泛采用的用于高效微调大型语言模型的技术。
- [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/abs/1902.00751) — Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, Sylvain Gelly (2019)
  Journal: arXiv preprint; DOI: [10.48550/arXiv.1902.00751](https://doi.org/10.48550/arXiv.1902.00751)
  介绍了适配器模块，这是一种通过注入小型可训练层实现参数高效微调的早期且有影响力的技术。
- [PEFT: Parameter-Efficient Fine-tuning](https://huggingface.co/docs/peft/index) — Hugging Face (2024)
  Publisher: Hugging Face
  Hugging Face PEFT 库的官方文档，提供了各种 PEFT 方法的实践指导和实现细节。

---

[上一节](../04-%E5%85%83%E5%AD%A6%E4%B9%A0%E7%9A%84%E4%BC%98%E5%8C%96%E8%A7%86%E8%A7%92/05-%E7%90%86%E8%AE%BA%E6%94%B6%E6%95%9B%E6%80%A7%E5%88%86%E6%9E%90.md) · [下一节](02-%E5%9F%BA%E7%A1%80%E6%A8%A1%E5%9E%8B%E9%80%82%E9%85%8D%E5%99%A8%E6%A8%A1%E5%9D%97.md)
