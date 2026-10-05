# 训练后量化 (PTQ) 编译流程

来源：[原文](https://apxml.com/zh/courses/compiler-runtime-optimization-ml/chapter-8-quantization-low-precision-optimizations/compiler-passes-ptq)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练后量化 (quantization) (PTQ) 提供了一种实用方法，可以在无需模型重训练的显著成本和复杂性的情况下，实现低精度推理 (inference)的性能优势，这与量化感知训练 (QAT) 不同。主要思路是获取一个预训练 (pre-training)的浮点模型，并在训练完成后，将其权重 (weight)和激活值转换为INT8或FP8等低精度格式。编译器在此转换自动化中扮演主要角色，负责转换模型表示、优化量化图，并最终生成高效的低精度代码。

典型的PTQ编译流程包含几个不同阶段，旨在将高层模型图转换为优化过的低精度可执行版本。

### 校准：确定量化 (quantization)参数 (parameter)

大多数PTQ流程的第一步是校准。由于我们正在从连续浮点范围转换为离散定点范围（例如，INT8的256个级别），因此需要确定最优映射。此映射由每个需要量化的张量（权重 (weight)和激活值）的尺度 ($s$) 和零点 ($z$) 参数定义。这种关系通常是仿射的：


$$
\text{浮点值} \approx s \times (\text{量化值} - z)
$$


为了找到合适的 $s$ 和 $z$ 值，编译器或专用量化工具会分析每个张量内的值分布。这需要一个有代表性的数据集，通常称为校准数据集。模型以浮点模式运行，并使用此数据集的输入，收集权重和中间激活值的运行时统计信息（范围、分布）。

常见的校准方法包括：

1. \*\*Min/Max：\*\*最简单的方法。尺度和零点直接根据观测到的浮点值的最小值和最大值计算，以覆盖低精度数据类型（例如，有符号INT8的 $[-128, 127]$）的整个范围。虽然简单，但这可能对异常值敏感，可能导致量化范围未充分使用。
2. **均方误差 (MSE)：** 寻找尺度/零点值，使原始浮点张量及其量化-反量化等效值之间的MSE最小化。这在计算上更为密集。
3. **Kullback-Leibler (KL) 散度 / 熵：** 将浮点值视为概率分布，并寻找使原始分布与量化后分布之间的KL散度最小化的量化参数。这通常比简单的min/max方法获得更好的准确性，因为它侧重于保留信息内容而非仅仅是范围。

校准完成后，编译器就获得了每个目标量化张量所需的 $s$ 和 $z$ 参数。

### IR 转换：插入量化 (quantization)操作

收集到校准数据后，编译器会修改模型的中间表示 (IR)。主要转换是向图中插入 `quantize`（量化） 和 `dequantize`（反量化） 操作（通常缩写为 `q` 和 `dq`）。

- 一个 `quantize` 操作接收一个浮点张量及其学得的尺度/零点，生成一个低精度张量（例如INT8）。
- 一个 `dequantize` 操作执行逆向转换，使用其关联的尺度/零点将低精度张量转换回浮点数。

一个原始的FP32操作，如 `Conv2D`，被替换为如下序列：

```
输入 (FP32) -> quantize -> 输入 (INT8)
权重 (FP32) -> quantize -> 权重 (INT8)

输入 (INT8), 权重 (INT8) -> Conv2D_INT8 -> 输出 (INT32 or INT8)

输出 (INT32 or INT8) -> dequantize -> 输出 (FP32)
```

这种初步插入创建了一个图，其中计算以低精度执行，但操作之间的接口可能仍然涉及转换回FP32。校准期间确定的尺度和零点作为属性嵌入 (embedding)到这些 `quantize`、`dequantize` 以及IR中新的低精度算子节点内。

### 量化 (quantization)模型的图优化

Q/DQ 节点的简单插入通常由于过多的转换而导致次优性能。因此，编译器会应用针对量化图的特定优化处理步骤：

1. **QDQ 取消：** 使用相同（或兼容）量化参数 (parameter)的相邻 `dequantize` -> `quantize` 对是冗余的，可以去除。当一个量化层的输出直接作为另一个层的输入时，这种情况经常发生。
2. **量化算子合并：** 标准算子合并（例如合并 Conv + Bias + ReLU）需要考虑量化。目的是将 Q/DQ 操作 *结合* 到主要计算核中。

   - 输入的 `quantize` 操作通常可以合并到 *消费* 操作中（例如，INT8卷积读取FP32输入并在内部量化）。
   - 输出的 `dequantize` 操作通常可以合并到 *生产* 操作中（例如，INT8卷积写入FP32输出，并在内部执行反量化）。
   - 或者，如果后续操作也以INT8进行，合并可能组合多个INT8操作，完全避免中间的FP32转换。这需要仔细管理尺度和零点（“再量化”）。
3. **再量化：** 当一个INT8操作的输出（可能累积在更宽的整数类型如INT32中）需要作为另一个INT8操作的输入时，需要一个 `requantization`（再量化）步骤。这涉及将中间结果（例如INT32累加器）重新缩放到下一层INT8输入所期望的尺度和零点。这通常涉及通过计算出的尺度因子（源自输入、权重 (weight)和输出尺度）进行整数乘法，然后进行右移操作。

   
   $$
   输出_{int8} = \text{四舍五入}\left( \frac{s_{in} s_{w}}{s_{out}} \times (\text{累加器}_{int32}) \right) + z_{out}
   $$
   

   （注意：确切的公式取决于具体的量化方案和硬件实现，通常涉及尺度因子 $s_{in}s_{w}/s_{out}$ 的定点算术近似。）

以下图表说明了优化如何转换图的简化视图：

> PTQ 图转换视图。初步插入量化/反量化节点（左侧），随后进行优化处理步骤，例如合并（右侧），其中Q/DQ操作被吸收到计算核中。

### 降低与代码生成

图级优化之后，编译器将高层量化 (quantization)操作（例如 `int8_conv2d`）降低为更底层的IR构造，可能包括显式整数算术、用于再量化的移位以及适用的向量 (vector)操作。这种降低后的表示随后被后端用于生成目标特定代码，如果可用，则采用专门的低精度硬件指令（例如Intel VNNI、ARM NEON点积指令、NVIDIA Tensor Core IMMA指令）。

### 考量与局限

尽管PTQ显著简化了量化 (quantization)模型的部署，但它并非万能。

- **准确性：** 与QAT相比，PTQ通常会导致更大的准确性下降，特别是对于高度敏感的模型或极低精度（低于INT8）的情况。缺乏重训练使模型无法使其权重 (weight)适应量化噪声。
- **校准数据：** 校准数据集的质量和代表性对于获得良好准确性非常重要。
- **混合精度：** 通常并非所有层都被量化。敏感性分析可能显示量化某些层会导致不可接受的准确性损失。编译器必须支持为这些混合精度图生成代码，高效处理FP32和INT8（或其他格式）之间的转换。

PTQ 编译流程是机器学习 (machine learning)部署工具包中的重要组成部分，通过自动化模型转换、优化和低精度执行的代码生成等复杂过程，提供了相对快速的性能改进途径。了解这些流程有助于工程师有效应用PTQ并诊断可能与准确性或性能相关的问题。

## 参考资料

- [Quantization for Deep Learning: A Comprehensive Survey](https://ieeexplore.ieee.org/document/9623837) — Zhenghong Zhao, Jincheng Zheng, Yunxiao Li, Yanggang An, Fan Xu, Junqing Xia, and Shengli Zhang (2021)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 44; Pages: 7767-7793; DOI: [10.1109/TPAMI.2021.3129532](https://doi.org/10.1109/TPAMI.2021.3129532)
  综述了多种量化技术，包括训练后量化、校准方法及其对模型性能的影响。
- [Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) — TensorFlow (2024)
  Publisher: Google AI
  一份主流机器学习框架中训练后量化的官方文档，涵盖了校准、优势和局限性等实际操作方面。
- [TVM: An Automatic End-to-End Optimizing Compiler for Deep Learning](https://doi.org/10.1145/3294344.3259969) — Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan, Haichen Shen, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy (2018)
  Journal: OSDI '18: Proceedings of the 13th USENIX Symposium on Operating Systems Design and Implementation; Pages: 578-594; DOI: [10.1145/3294344.3259969](https://doi.org/10.1145/3294344.3259969)
  介绍了一种深度学习的端到端优化编译器，讨论了其对低精度操作和与PTQ相关的图优化的支持。
- [Data-Free Quantization Through Weight Equalization and Bias Correction](https://ieeexplore.ieee.org/document/9010531) — Markus Nagel, Anna Alverio, Tal Hakim, Vaibhav Kumar, and Tijmen Blankevoort (2019)
  Journal: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 8344-8353; DOI: [10.1109/ICCV.2019.00936](https://doi.org/10.1109/ICCV.2019.00936)
  提出了一种无需额外训练数据即可提高训练后量化准确性的技术，解决了PTQ的一个局限性。

---

[上一节](03-%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83%20%28QAT%29%20%E7%9A%84%E7%BC%96%E8%AF%91%E5%99%A8%E5%A4%84%E7%90%86%E8%BF%87%E7%A8%8B.md) · [下一节](05-%E7%94%9F%E6%88%90%E4%BD%8E%E7%B2%BE%E5%BA%A6%E6%A0%B8%E5%87%BD%E6%95%B0.md)
