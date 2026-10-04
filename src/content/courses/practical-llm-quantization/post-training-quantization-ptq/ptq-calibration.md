---
course: "practical-llm-quantization"
chapter: "post-training-quantization-ptq"
lesson: "ptq-calibration"
sourceId: 4611
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-2-post-training-quantization-ptq/ptq-calibration"
title: "校准：选择有代表性的数据"
description: "了解校准的重要性以及如何选择合适的数据来确定量化参数。"
order: 2
plots: ["plots/4611-0.json"]
sourceHash: "ec7ea7f4b54dee2ee6f327ef7ac18f3704d8bb35759d5daa3c9c5cb080927457"
sourceCorrections: []
---

训练后量化 (quantization)（PTQ）提供了一种无需重新训练即可量化预训练 (pre-training)模型的方法。许多PTQ方法，特别是静态量化，其中一个重要步骤是**校准**。可以将校准看作一个有针对性的测量过程。我们需要了解流经模型的典型值范围，特别是激活值，以便将它们有效地从`FP32`映射到`INT8`或`INT4`等低精度类型。

为什么需要这种测量？尽管预训练模型的权重 (weight)是固定的，但激活值会根据输入模型的数据动态变化。简单使用`FP32`的整个理论范围会非常低效，导致`INT8`或`INT4`中有限范围的利用率低下。同样，尝试从整个原始训练数据集中确定范围可能计算量大，并可能被罕见的异常值影响。

校准弥补了这一差距。它涉及将一小组经过仔细选择的输入数据输入原始`FP32`模型，并观察不同点（通常是我们要量化的层的输入）的激活值。目的不是重新训练模型，而是收集有关激活值分布的统计信息。

### 校准数据的作用

用于此过程的数据称为**校准数据集**。它的目的是*代表性地反映*模型在实际推理 (inference)过程中将遇到的输入。通过处理这些有代表性的数据，我们可以观察到每个层激活值的典型范围（例如，最小值和最大值）。这些观察到的范围随后用于计算特定的量化 (quantization)参数 (parameter)，主要是**比例因子** ($s$) 和**零点** ($z$)，它们定义了浮点值与目标整数表示之间的映射。

回顾基本量化公式：


$$
\text{量化值} = \text{round}(\frac{\text{浮点值}}{s} + z)
$$


校准为确定激活值的最佳 $s$ 和 $z$ 提供了经验依据，最大限度地减少了转换过程中的信息损失。

### 选择有代表性的数据

校准的有效性完全取决于校准数据集的质量和代表性。什么构成了“代表性”数据？

1. **分布匹配：** 校准数据理想情况下应反映模型在其部署环境中将处理的数据的统计分布。如果您在一个大型语言模型（LLM）上使用维基百科文章进行校准，但将其部署用于客户服务聊天，那么校准期间观察到的激活范围可能无法准确反映生产环境中遇到的范围，这可能导致次优的量化 (quantization)和精度损失。
2. **多样性：** 数据应涵盖预期的输入多样性。对于大型语言模型（LLM）而言，这可能包括其预期处理的不同句子结构、主题、长度和交互类型。
3. **规模：** 校准通常不需要大量数据。通常，几百到几千个样本就足够了。

   - *样本过少：* 可能无法捕捉到激活值的真实方差和范围，导致参数 (parameter)选择不当（例如，截断常见值或浪费范围）。
   - *样本过多：* 会增加校准所需时间，并带来边际效益递减，可能与PTQ的“快速”特性相冲突。

   找到合适的规模通常需要一些经验性测试，但从大约100-1000个多样化样本开始是一个常见做法。

### 校准数据的来源

您可以在哪里获取这些数据？常见来源包括：

- **原始训练数据集的子集：** 如果训练数据准确反映了部署用例。
- **验证数据集：** 常用，因为它代表模型尚未明确训练但预期表现良好的数据。
  "\* **未标记 (token)的生产数据：** 如果可用，使用模型将看到的数据样本会非常有效。"

校准数据的特性应与预期推理 (inference)数据的特性保持一致。

### 校准数据的影响

将校准样本输入模型可以让我们捕获激活统计信息。例如，一种简单方法（MinMax量化 (quantization)）涉及记录所有校准样本中每个层观察到的最小和最大激活值。这些`min`和`max`值随后直接用于计算比例和零点。

考虑处理校准数据后特定层的激活值分布：



![校准数据中的激活值分布](plots/4611-0.json)



> 一个直方图，显示了校准期间特定张量观测到的激活值频率。MinMax校准将使用观测到的最小值（-3.2）和最大值（4.8）来设置量化范围。

如果校准数据不具代表性，观测到的`min`和`max`可能过窄（在推理 (inference)时截断常见值）或过宽（未充分利用`INT8`范围），这两种情况都会导致量化误差增大。

### 静态量化 (quantization)与动态量化中的校准

需要再次强调的是，这种使用数据集的明确校准步骤主要与**静态量化**相关。在静态PTQ中，我们基于校准数据预先计算权重 (weight)*和*激活值的量化参数 (parameter)（$s$和$z$）。这些参数随后被固定并在推理 (inference)期间使用。

相比之下，**动态量化**通常只在离线状态下量化权重。激活值在推理期间“即时”量化。对于每个输入激活张量，其范围（最小值/最大值）是动态计算的，然后确定并应用量化参数。这避免了激活值需要单独校准数据集的问题，但在推理期间引入了动态计算这些范围的计算开销。

因此，选择合适的校准数据是静态训练后量化技术获得良好性能的根本步骤。接下来的章节将介绍使用这些校准统计信息的不同算法，并比较静态方法与动态方法。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://doi.org/10.1109/CVPR.2018.00097) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 870-879; DOI: [10.1109/CVPR.2018.00097](https://doi.org/10.1109/CVPR.2018.00097)
  一篇基础论文，介绍了训练后量化和量化感知训练的原理，详细阐述了激活校准在静态量化中的作用。
- [Post Training Static Quantization](https://pytorch.org/docs/stable/quantization.html#post-training-static-quantization) — PyTorch Documentation (2019)
  Publisher: PyTorch Foundation
  提供了在PyTorch中实现静态训练后量化的实用指南和API细节，包括如何收集和使用校准数据。
- [Integer Quantization for Deep Learning Inference: Principles and Empirical Advances](https://link.springer.com/article/10.1007/s11265-019-01460-1) — Ron Banner, Yaniv Landa, Alexander Fefelov, Elad Hoffer (2019)
  Journal: Journal of Signal Processing Systems; Publisher: Springer US; Volume: 91; Pages: 635-648; DOI: [10.1007/s11265-019-01460-1](https://doi.org/10.1007/s11265-019-01460-1)
  讨论了整数量化的基本原理和进展，涵盖了激活范围估计以及使用校准数据确定比例因子和零点。
