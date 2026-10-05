# 量化特定LLM组成部分（注意力机制、归一化）

来源：[原文](https://apxml.com/zh/courses/quantized-llm-deployment/chapter-5-addressing-advanced-challenges/quantizing-specific-llm-components)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管对大型语言模型（LLM）中的大型线性层进行量化 (quantization)通常会带来显著的性能提升，但其他重要组成部分，特别是注意力机制 (attention mechanism)和归一化 (normalization)层，面临着独特的挑战。对这些部分简单地应用标准量化技术可能会过度降低模型准确性，因为它们的操作通常比简单的矩阵乘法对数值精度更敏感。了解这些敏感性对于制定有效的量化策略很要紧。

## 量化 (quantization)注意力机制 (attention mechanism)的详细情况

自注意力 (self-attention)机制是Transformer架构的基础，使模型在处理序列数据时能够权衡不同token的重要性。然而，注意力计算中的几个步骤对量化噪声很敏感。

### Softmax的敏感性

Softmax函数用于将原始注意力分数（$QK^T$）转换为概率，它众所周知难以有效量化 (quantization)。其指数性质意味着输入值的微小变化，特别是接近零的值，可能导致输出概率的较大变化。此外，输出分布被限制在0到1之间。量化，特别是低比特量化，会引入噪声，从而扭曲这些概率，可能导致模型关注不正确的token或将注意力分布得过宽或过窄。


$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}} \quad \text{对于 } i = 1, \dots, K
$$


量化输入$z_i$（原始注意力分数）或softmax计算中的中间值可能导致最终注意力权重 (weight)出现较大误差。由于这种敏感性，一个常见策略是**保持softmax计算使用更高精度**，例如FP16甚至FP32，即使周围的矩阵乘法（Q、K、V投影）被量化到INT8或INT4等较低比特宽度。

### 注意力分数中的高动态范围

查询点积（$QK^T$）的中间结果通常呈现高动态范围。一些分数可能非常大，而另一些则很小但对于捕获细微关系仍可能重要。标准量化 (quantization)方法（如min-max缩放）难以处理此类分布。如果量化范围设置为适应较大的异常值分数，则较小分数的精度会变得非常粗糙，实际上抹去它们的信息。

考虑对称量化中的缩放因子$\alpha$：


$$
\alpha = \frac{\max(|X|)}{2^{b-1}-1}
$$


其中$X$是被量化的张量，$b$是比特宽度。较大的最大绝对值$\max(|X|)$会导致较大的$\alpha$，从而产生较大的量化步长。这意味着$X$中较小的值在量化后会被映射到接近零的值，丢失它们的信息内容。

解决此问题的策略包括：

- **分块量化：** 独立量化$QK^T$矩阵的较小块，允许使用更适合局部动态范围的不同缩放因子。
- **对数量化：** 使用对数尺度有时能更好地表示具有大动态范围的数据，尽管这通常需要专门的硬件支持。
- **异常值裁剪：** 在量化前裁剪极端值，尽管这必须小心操作，以避免丢失这些异常值所代表的重要信息。

### 值聚合误差

注意力机制 (attention mechanism)的最后一步涉及聚合由softmax计算出的注意力概率加权的值向量 (vector)（$V$）。在量化 (quantization)$V$或注意力概率过程中引入的误差会在此加权和过程中累积，导致注意力机制输出的不准确性。

## 归一化 (normalization)层面临的挑战

像层归一化（LayerNorm）和RMS归一化（RMSNorm）这样的归一化层对稳定训练和提升LLM性能很要紧。它们通过标准化层内的激活值来工作。

### 层归一化 (normalization)（LayerNorm）

LayerNorm计算层内激活值的均值（$\mu$）和方差（$\sigma^2$），归一化激活值，然后应用可学习的缩放（$\gamma$）和平移（$\beta$）参数 (parameter)：


$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$


### RMS归一化 (normalization)（RMSNorm）

RMSNorm是一种更简单的变体，它使用均方根统计量进行归一化，省略了中心化（均值减法）和平移参数 (parameter)（$\beta$）：


$$
\text{RMSNorm}(x) = \gamma \frac{x}{\sqrt{\text{RMS}(x)^2 + \epsilon}} \quad \text{这里} \quad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}
$$


### 统计量计算的敏感性

量化 (quantization)归一化 (normalization)层的主要挑战在于统计量（$\mu$、$\sigma^2$或RMS）的计算。这些统计量是根据激活值计算的。如果激活值本身以低精度量化，产生的噪声会显著扰动计算出的均值、方差或RMS。此误差随后通过归一化公式传播，可能使网络不稳定或改变其表征能力。

例如，在计算$\mu$和$\sigma^2$之前量化输入$x$可能导致这些统计量的不准确估计。除法运算（通过$\sqrt{\sigma^2 + \epsilon}$或$\sqrt{\text{RMS}(x)^2 + \epsilon}$）也对分母中的误差敏感，尤其当方差或RMS较小时。

### 归一化 (normalization)层的策略

- **更高精度计算：** 类似于softmax，统计量计算和归一化操作本身通常保持更高精度（例如FP16/FP32）。层的输入和输出可能仍被量化 (quantization)，但中间计算保持保真度。
- **融合核函数：** 在硬件允许的情况下，将归一化操作与前置或后续操作（如线性层或激活函数 (activation function)）融合，有时可以通过减少量化/反量化步骤的数量来缓解精度问题。
- **量化感知训练（QAT）：** QAT可以帮助模型适应归一化层内量化的影响，使可学习参数 (parameter)（$\gamma$、$\beta$）补偿统计量计算过程中引入的一些噪声。

## 混合精度：一种实用方法

鉴于这些挑战，对整个LLM进行纯粹统一的低比特量化 (quantization)通常是次优的。一种更实用和有效的策略是**混合精度量化**。这涉及根据模型不同部分的敏感性有选择地对它们应用不同精度级别。

通常：

- **线性层（权重 (weight)矩阵）：** 通常可以使用GPTQ或AWQ等技术容忍激进的量化（例如INT8、INT4、NF4）。
- **激活值：** 可能会量化为INT8或有时FP8，这取决于模型和任务。
- **敏感操作（Softmax、归一化 (normalization)）：** 通常保持更高精度（FP16或BFloat16）以保持准确性。

下图描绘了一个Transformer块段内潜在的混合精度方案。

> 这是一个Transformer块内潜在混合精度应用的简化视图，其中强调了softmax和归一化计算使用更高精度，而线性投影使用较低精度。输入/输出类型取决于整体模型配置。

成功量化LLM需要摒弃将模型视为统一操作序列的做法。注意力机制 (attention mechanism)和归一化层因其操作特性和对数值精度的敏感性而需要专门考虑。采用混合精度方法，可能与QAT和专门的量化方案结合，经常需要以取得显著的性能提升，而不造成不可接受的模型准确性损失。前几章讨论的PTQ（如GPTQ/AWQ）技术主要侧重于线性层；调整或结合这些与针对敏感组件的策略是高级量化实践的一个重要方面。

## 参考资料

- [LLM.int8(): 8-bit Matrix Multiplication for Large Language Models](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022)
  Journal: NeurIPS 2022; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  介绍了LLM的8位量化混合精度方法，专门解决了导致高动态范围的离群值问题，这是注意力分数的一个关键挑战。
- [Q-BERT: Quantizing BERT for Fast Inference](https://papers.nips.cc/paper_files/paper/2019/hash/a84a0d9237f374828358406562507850-Abstract.html) — Sheng Shen, Zhenglun Ma, Kai Hou, Ruohui Ye, Zizheng Niu, Wenshuo Li, Feng Wu, Amir Gholami, Shunning Wei, Michael W. Mahoney, and Kurt Keutzer (2019)
  Journal: Advances in Neural Information Processing Systems 32 (NeurIPS 2019); Publisher: NeurIPS; DOI: [10.48550/arXiv.1909.05846](https://doi.org/10.48550/arXiv.1909.05846)
  一项基础工作，详细阐述了量化Transformer模型的挑战，包括Softmax和归一化层的敏感性，并提出了保持准确性的解决方案。

---

[上一节](02-%E5%A4%84%E7%90%86%E6%BF%80%E6%B4%BB%E5%92%8C%E6%9D%83%E9%87%8D%E7%9A%84%E7%A6%BB%E7%BE%A4%E5%80%BC.md) · [下一节](04-%E7%A1%AC%E4%BB%B6%E9%99%90%E5%88%B6%E4%B8%8E%E5%86%85%E6%A0%B8%E6%94%AF%E6%8C%81.md)
