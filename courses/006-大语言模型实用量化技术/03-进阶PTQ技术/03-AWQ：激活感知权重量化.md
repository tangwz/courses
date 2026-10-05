# AWQ：激活感知权重量化

来源：[原文](https://apxml.com/zh/courses/practical-llm-quantization/chapter-3-advanced-ptq-techniques/awq-concepts)

[返回章节目录](README.md) · [返回课程目录](../README.md)

基本训练后量化 (quantization) (PTQ) 方法，例如MinMax校准，在确定量化参数 (parameter)时，对层内所有权重 (weight)的处理大致相同。然而，这种统一的做法有时会产生问题，尤其是在目标精度非常低（如4比特整数 (INT4)）时。原因何在？因为并非所有权重对模型的输出贡献相同。有些权重与持续高幅度的激活值相乘，因此其精确表示对于维持层整体功能的影响更大。反之，量化与小激活值相关的权重，其影响可能较小。

激活感知权重量化 (AWQ) 直接解决了这一观察。这是一种高级PTQ技术，通过分析推理 (inference)过程中遇到的典型激活值大小，旨在保护最重要的权重。

### 核心思想：保护显著权重 (weight)

AWQ 的中心原则是：量化 (quantization)误差对于与大激活值相乘的权重危害更大。考虑线性层中的简单矩阵乘法：$Y = WX$，其中 $W$ 是权重矩阵，$X$ 是输入激活向量 (vector)。如果特定权重 $W_{ij}$ 经常与较大的激活值 $X_j$ 相乘，那么量化 $W_{ij}$ 时引入的任何误差都将被放大，从而对输出 $Y_i$ 产生显著影响。

AWQ 建议我们应优先保持这些“显著”权重的精度。它识别这些权重，不是通过查看权重本身，而是通过检查它们所作用的*激活值*。AWQ 使用少量校准数据集，观察每个输入通道到权重矩阵的激活值大小分布。持续显示大激活值的通道表示相应的权重更重要。

### 通过激活值识别重要权重 (weight)

AWQ 根据激活尺度来评估每个权重通道的重要性。它通过模型处理校准数据，记录每个输入特征通道进入线性层时的激活统计数据（通常是最大绝对值或高百分位数）。持续表现出最大激活值的一小部分通道（例如，1% 甚至 0.1%）被认为是最重要的。与这些激活通道相连的权重，正是 AWQ 在量化 (quantization)过程中旨在保护的对象。

### 用于保护的每通道缩放

AWQ 不采用复杂的非均匀量化 (quantization)方案，而是使用一种巧妙的预处理步骤：每通道权重 (weight)缩放。目的是在应用标准量化*之前*，减小显著权重组的动态范围。

设想一个权重矩阵 $W$。AWQ 为每个输入通道 $j$ 计算一个缩放因子 $s_j$。此因子主要由相应激活通道 $X_j$ 观察到的大小决定。一种常见方法是设置缩放因子 $s_j$，使得缩放后的权重 $W'_{:,j} = W_{:,j} / s_j$ 的范围减小，特别是对于激活值 $X_j$ 较大的通道 $j$。

为保持层计算的数学等效性，此缩放必须进行补偿。原始运算 $Y = WX$ 可重写为：


$$
Y = (W S^{-1}) (S X)
$$


$S$ 是一个包含缩放因子 $s_j$ 的对角矩阵。权重 $W$ 通过 $S^{-1}$ 进行缩放（对每个列 $j$ 进行 $s_j$ 的元素级除法），从而得到 $W' = W S^{-1}$。然后将量化（通常是每通道对称 INT4）应用于此缩放后的权重矩阵 $W'$。相应的激活值 $X$ 在乘法前通过 $S$ 进行*向上*缩放。

此缩放操作有效地将部分量化难度从权重转移到激活值。AWQ 的前提是：激活值通常更容易精确表示，或者缩放有时可以并入或合并到前面的操作（如层归一化 (normalization)）中，且开销极小。

通道 $j$ 的缩放因子 $s_j$ 通常根据激活值和权重范围计算，以平衡缩放，例如：


$$
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_{:,j}|)}
$$


此处，$\max(|X_j|)$ 是在校准数据集中观察到的激活通道 $j$ 的最大绝对值，$\max(|W_{:,j}|)$ 是相应权重通道（列）的最大绝对值，$\alpha$ 是一个超参数 (parameter) (hyperparameter)（通常在 0.5 到 1 之间），它控制基于激活值大小的缩放强度。 较高的 $\alpha$ 赋予激活尺度更大的重要性。

> AWQ 的流程。原始权重矩阵 $W$ 根据激活统计数据进行每通道缩放，得到 $W'$。此 $W'$ 被量化。激活值 $X$ 在乘法前进行反向缩放，得到 $X'$，以保持输出。

### 优点与考量

AWQ 具备多项优点：

- **精度提升：** 通过保护显著权重 (weight)，AWQ 在低比特宽度（特别是 INT3/INT4）下通常比基本 PTQ 方法获得更好的精度，有时接近原始 FP16 的表现。
- **无需重训练：** 与其他 PTQ 方法一样，它在预训练 (pre-training)模型上运行，无需微调 (fine-tuning)或访问原始训练流程。
- **效率高：** 校准和缩放过程通常比涉及 Hessian 计算（如 GPTQ）或完全重训练（如 QAT）的方法更快。

然而，也存在一些考量：

- **校准数据：** 校准数据集的质量和代表性对于准确识别显著激活通道很重要。
- **缩放开销：** 缩放因子（$S$ 或 $S^{-1}$）需要在推理 (inference)过程中存储和应用。虽然有时可以并入相邻层（例如 LayerNorm），但这可能会在需要明确缩放操作时引入少量计算开销。
- **超参数 (parameter) (hyperparameter)：** 识别显著通道的百分位数阈值和缩放指数 $\alpha$ 等参数可能需要根据特定模型和任务进行调整，以获得最佳性能。

### AWQ 的实际应用

AWQ 已被证明能有效将各种大型语言模型 (LLMs) 量化 (quantization)到 4 比特精度，且精度损失极小。库和框架通常提供实现，处理推理 (inference)过程中的校准、缩放计算和应用。使用 AWQ 时，通常需要提供预训练 (pre-training)模型和少量校准数据集。该过程输出量化后的权重 (weight)和必要的缩放因子，可供部署。

与同样处理激活异常值的 SmoothQuant 相比，AWQ 的方法有所不同。SmoothQuant 通过静态缩放将量化难度从激活值转移到权重。AWQ 根据激活值的重要性选择性地缩放权重，旨在使*重要*权重的量化变得更容易。这两种技术都旨在提高低比特量化精度，但使用不同的机制。与 GPTQ 相比，AWQ 避免了 GPTQ 层级优化中使用的计算量更大的 Hessian 矩阵近似。

## 参考资料

- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: arXiv; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  介绍激活感知权重量化（AWQ）方法的原创研究论文，详细阐述其在大语言模型中保护重要权重的策略。
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) — Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han (2022)
  Journal: arXiv; DOI: [10.48550/arXiv.2211.10438](https://doi.org/10.48550/arXiv.2211.10438)
  一篇关于训练后量化的重要论文，通过使用静态缩放将量化难度从激活转移到权重来解决激活异常值问题，为AWQ提供了比较视角。
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (2023)
  Journal: ICLR 2023; DOI: [10.48550/arXiv.2210.17323](https://doi.org/10.48550/arXiv.2210.17323)
  提出了一种逐层的最优脑量化技术，用于训练后量化，为在低位宽下实现LLM高精度提供了另一种方法。

---

[上一节](02-%E7%90%86%E8%A7%A3%20GPTQ%20%E7%AE%97%E6%B3%95%E6%9C%BA%E5%88%B6.md) · [下一节](04-SmoothQuant%EF%BC%9A%E5%87%8F%E8%BD%BB%E6%BF%80%E6%B4%BB%E7%A6%BB%E7%BE%A4%E5%80%BC.md)
