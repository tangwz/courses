# QAT 与 PTQ 的优缺点对比

来源：[原文](https://apxml.com/zh/courses/practical-llm-quantization/chapter-4-quantization-aware-training-qat/qat-vs-ptq-comparison)

[返回章节目录](README.md) · [返回课程目录](../README.md)

选择训练后量化 (quantization) (PTQ) 和量化感知训练 (QAT) 需要了解它们各自的优点和不足。QAT 将量化模拟整合到训练流程中，使模型能够适应，而 PTQ 则在模型训练完成后才进行量化。这一根本差异导致在准确性、复杂性、成本和实施要求上有着明显的取舍。

接下来，我们细分一下 QAT 相较于 PTQ 的优点和缺点。

### 量化 (quantization)感知训练 (QAT) 的优点

1. **更高的准确度潜力，尤其是在低精度下：** 这是选择 QAT 的主要原因。通过在训练或微调 (fine-tuning)过程中模拟量化噪声（使用伪量化节点），模型学习调整其权重 (weight)，以尽量减少精度降低的影响。优化过程直接考虑了浮点值映射到整数所引入的误差。因此，QAT 通常可以恢复 PTQ 过程中损失的准确度，尤其是在针对 INT4 等极低比特宽度时，PTQ 可能表现得很吃力。模型学习如何适应量化。
2. **更好地处理模型中敏感的组件：** 模型中的某些层或参数 (parameter)可能比其他部分对量化更敏感。PTQ 在训练后静态应用量化，可能不成比例地损害这些敏感部分。QAT 通过反向传播 (backpropagation)和如直通估计器 (STE) 等技术，允许模型在训练期间选择性地调整所有层中的权重，可能找到更好地保护这些敏感组件免受量化误差影响的方案。
3. **对激进量化有更强的适应性：** 因为模型在存在量化效应的情况下进行训练，使用 QAT 通常可以达到更低的比特宽度（例如 INT4，甚至是混合精度方案），同时保持可接受的性能，而 PTQ 对同一模型可能达不到这样的效果。

### 量化 (quantization)感知训练 (QAT) 的缺点

1. **复杂性增加：** QAT 本质上比 PTQ 更复杂。它需要修改训练流程，将伪量化操作整合到模型图中，管理量化模拟的细节，并可能需要调整训练超参数 (parameter) (hyperparameter)。这与 PTQ 简单的工作流程形成鲜明对比，PTQ 只是对已训练的模型直接应用量化。
2. **计算成本较高：** QAT 需要完整或部分的训练或微调 (fine-tuning)周期。这包括处理大型数据集、执行梯度更新以及消耗大量的计算资源（GPU 时间、能源），通常与原始模型训练或标准微调的开销相当。另一方面，PTQ 通常只需要少量校准数据集，且量化过程本身的计算量要少得多。
3. **需要训练数据和流程的权限：** 为了进行 QAT，你需要访问具有代表性的训练或微调数据集以及完整的训练基础设施（代码、环境、超参数）。如果你使用的是专有模型或只能获得最终的预训练 (pre-training)权重 (weight)，这可能不可行。PTQ 对数据需求极小（少量校准集），使得它在这种情况下更易于使用。
4. **开发和实验时间更长：** QAT 涉及的训练周期意味着实验运行时间比 PTQ 长得多。迭代不同的 QAT 策略、超参数或量化配置是一个耗时的过程。
5. **潜在的训练不稳定：** 引入模拟量化操作和使用 STE 有时会影响训练过程的稳定性和收敛性。可能需要仔细实现并调整学习率或优化策略。

### 如何选择：QAT vs. PTQ

这一决定通常归结为在准确度要求与可用资源和对复杂性的承受能力之间取得平衡。

- **从 PTQ 开始：** 考虑到它的速度和简洁性，PTQ 通常是建议的第一步。应用各种 PTQ 技术（静态、动态、以及适用的高级方法如 GPTQ），并评估准确度和性能。如果结果符合你的要求，可能就不需要 QAT 增加的复杂性。
- **在以下情况考虑 QAT：**

  - PTQ 导致模型准确度对你的目标应用来说下降得无法接受。
  - 你需要量化 (quantization)到极低的比特宽度（例如 INT4），而 PTQ 在此方面已知难以处理。
  - 你有权使用必要的训练数据、基础设施和计算资源。
  - 实现量化模型最高的准确度是一个主要目标，值得付出额外的努力和成本。

下表总结了主要区别：

| 特性 | 训练后量化 (PTQ) | 量化感知训练 (QAT) |
| --- | --- | --- |
| **主要目标** | 速度、简洁性 | 最大化准确度 |
| **准确度** | 通常较好，在低比特下可能下降 | 通常更高，尤其是在低比特下 |
| **复杂性** | 低 | 高（需要训练/微调 (fine-tuning)） |
| **计算成本** | 低（校准 + 转换） | 高（训练/微调） |
| **数据需求** | 少量校准集 | 完整训练/微调数据集 |
| **时间** | 快 | 慢（训练时间） |
| **要求** | 预训练 (pre-training)模型 | 训练流程、数据、计算资源 |
| **使用时机** | 第一步，准确度足够，资源有限 | PTQ 不足，激进量化，需要最高准确度 |

了解这些取舍有助于你做出明智的决定，选择哪种量化策略最适合你的具体项目目标、限制以及你的大型语言模型的性能特点。

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  这篇论文介绍了 GPTQ，一种专为大型语言模型设计的高效准确的训练后量化方法。它展示了 PTQ 如何有效地应用于 LLM 以实现显著压缩并保持低精度损失，与课程重点高度相关。
- [Deep Learning with Low-Precision Quantization: A Review](https://www.sciencedirect.com/science/article/pii/S074373152030090X) — Zechun Cai, Dong Huang, Zizheng Pan, Yunhe Wang, Kai Han, Wenshi Zhang, Errui Ding, Yiping Deng, Yubei Chen, and Xiangyu Zhang (2020)
  Journal: Journal of Parallel and Distributed Computing; Publisher: Elsevier; Volume: 144; Pages: 110-120; DOI: [10.1016/j.jpdc.2020.05.006](https://doi.org/10.1016/j.jpdc.2020.05.006)
  这篇全面的综述文章回顾了深度学习中各种低精度量化技术，涵盖了训练后量化 (PTQ) 和量化感知训练 (QAT)。它提供了对其方法、优势和局限性的广阔视角。

---

[上一节](05-%E4%BD%BF%E7%94%A8%E9%87%8F%E5%8C%96%E8%8A%82%E7%82%B9%E5%BE%AE%E8%B0%83%E6%A8%A1%E5%9E%8B.md) · [下一节](07-QAT%E5%AE%9E%E6%96%BD%E4%B8%AD%E7%9A%84%E5%AE%9E%E9%99%85%E8%80%83%E9%87%8F.md)
