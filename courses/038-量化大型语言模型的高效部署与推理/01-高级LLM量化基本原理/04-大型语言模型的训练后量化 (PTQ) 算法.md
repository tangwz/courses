# 大型语言模型的训练后量化 (PTQ) 算法

来源：[原文](https://apxml.com/zh/courses/quantized-llm-deployment/chapter-1-advanced-llm-quantization-fundamentals/ptq-algorithms-llms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管训练后量化 (quantization) (PTQ) 提供了无需昂贵再训练即可对模型进行量化的显著优势，但将传统PTQ技术直接应用于大型语言模型 (LLMs) 通常会导致不可接受的精度下降。LLMs具有独特的规模和架构特性，使它们对量化固有的精度降低特别敏感。标准PTQ方法通常根据权重 (weight)或激活的简单范围（最小值/最大值）来确定量化参数 (parameter)（缩放因子 $s$ 和零点 $z$），但它们难以保持LLM庞大参数空间中编码的信息，特别是在目标是INT4等激进的低位格式时。

这一局限性促使了专为LLMs设计的更精密的PTQ算法的发展。这些方法不仅限于简单的范围估计，还纳入了关于模型结构和数据流的见解，以更智能的方式减少量化误差。两个获得广泛关注的典型例子是GPTQ和AWQ。

### GPTQ：逐层量化 (quantization)与误差补偿

GPTQ (Generative Pre-trained Transformer Quantizer) 通过采用一种更精细的逐层量化策略并结合误差补偿来应对精度挑战。与基于全局统计信息同时量化层中所有权重 (weight)不同，GPTQ按顺序处理权重，通常以小块或列的形式进行。

核心思想是最小化层输出的重建误差。当一个权重（或权重块）被量化时，会引入误差。GPTQ试图通过对同一层中*剩余的*、尚未量化的权重进行小幅调整来修正这个误差。这种补偿机制旨在局部抵消量化引起的扰动。

从数学角度看，对于给定的层权重矩阵 $W$ 和输入 $X$，我们希望找到一个量化权重矩阵 $W_q$，使得输出 $W_q X$ 尽可能接近原始输出 $W X$。GPTQ通过为每一层解决一个优化问题来实现这一点：


$$
\underset{W_q}{\text{argmin}} \| WX - W_q X \|_F^2 = \underset{W_q}{\text{argmin}} \| (W - W_q) X \|_F^2
$$


限制条件是 $W_q$ 的元素属于目标低位表示。GPTQ采用迭代的贪心方法。它逐个（或逐列）量化权重，并使用从逆Hessian矩阵 $(X X^T)^{-1}$（或其近似值）获得的信息更新剩余权重，以最小化平方误差。这使得剩余权重能够适应并补偿由之前权重量化引入的误差。

> 一个比较，显示了GPTQ在量化过程中如何迭代更新权重，这与一次性量化所有权重的朴素方法不同。

由于它在过程中主动补偿量化误差，GPTQ通常比简单方法能更好地保持精度，尤其在4位精度下，尽管代价是量化过程计算量更大。

### AWQ：激活感知权重 (weight)量化 (quantization)

激活感知权重量化 (AWQ) 采取了不同的方法，其动机是观察到LLM中并非所有权重都同等重要。AWQ假设与具有较大激活幅度的激活相关的权重对模型性能更具影响。标准量化方法平等对待所有权重，这可能不成比例地损害这些重要权重。

AWQ提出了一种简单而有效的解决方案：通过缩放来保护重要权重。它不直接修改量化过程本身，但在量化*之前*为权重引入了逐通道的缩放因子。

过程如下：

1. **分析激活：** 将校准数据送入模型，观察进入每个线性层的激活幅度分布。
2. **识别重要通道：** 确定激活幅度始终较大的通道。这些对应于被认为更重要的权重通道。
3. **确定缩放因子：** 计算权重矩阵的逐通道缩放因子。目标是找到缩放因子 $s_c$，使得量化 $W_c / s_c$（其中 $W_c$ 是权重通道）能最小化量化误差，特别是对于在步骤2中识别出的重要通道。这有效地减小了重要权重相对于不那么重要权重的量化步长。
4. **量化缩放后的权重：** 对缩放后的权重 $W' = W / s$ 应用标准PTQ（例如对称逐通道量化）。
5. **推理 (inference)：** 在推理过程中，缩放操作被反转。运算变为 $(W'_q \cdot s) X$，其中 $W'_q$ 是缩放后权重的量化版本。这个缩放因子 $s$ 被吸收到层的计算中，通常不会增加显著的额外开销。

> 激活感知权重量化 (AWQ) 过程的流程，重点说明了激活分析和权重缩放步骤。

AWQ的主要优势在于其相对于GPTQ的简洁性和速度，因为它避免了复杂的迭代更新。它依赖于一个有力的启发式方法，即激活幅度与权重重要性高度相关。尽管在所有情况下可能无法达到GPTQ的绝对最高精度，但AWQ在量化速度、易于实现性和最终模型精度之间提供了令人满意的平衡，使其成为LLM中PTQ的另一个受欢迎的选项。

### GPTQ与AWQ的选择

GPTQ和AWQ相较于朴素PTQ，对于LLMs而言都有显著进步。

- **GPTQ** 通常能带来更好的精度，特别是对于非常低的比特数（例如3位或4位），得益于其直接误差最小化方法。然而，量化 (quantization)过程本身要慢得多。
- **AWQ** 提供了快得多的量化过程，并通过保护重要权重 (weight)提供良好的精度。它更简洁，通常更容易实现和调整。

两者之间的选择可能取决于具体模型、目标位宽、量化步骤可用的计算资源以及所需的精度保持水平。

“这些先进的PTQ算法是高效部署LLM的重要工具。通过实现在可接受精度损失下对低位宽的有效量化，它们显著降低了推理 (inference)的计算和内存需求。在下一章中，我们将查看可用于将GPTQ和AWQ等算法应用于LLM的实用工具包和库。”

## 参考资料

- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.01730) — Elias Frantar, Saleh Ashkaneh, Jerry Zhao, Sabine M. Fathi, Shuo Yang, Siddarth Malreddy, Artem Gorevoy, Daniel Adiwardana, Jonathan Herzig, Daniel N. Gillman, Oleg Rybakov, Adam Roberts, David R. So, Shivani Agrawal, Sharan Narang, Michael S. Duke, William J. Dally, Hattie Zhou, James Bradbury, Matthew Buddy, Brian Catanzaro, Michael G. Mozer, Somasekhar Vemuri, Wojciech Zaremba, Alon Halevy, Robert Schapire (2022)
  Journal: arXiv preprint arXiv:2210.01730; DOI: [https://doi.org/10.48550/arXiv.2210.01730](https://doi.org/10.48550/arXiv.2210.01730)
  介绍了GPTQ算法，这是一种针对LLM的分层、误差补偿型训练后量化方法，旨在以最小的精度损失实现低比特量化。
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (2023)
  Journal: arXiv preprint arXiv:2306.00978; DOI: [10.48550/arXiv.2306.00978](https://doi.org/10.48550/arXiv.2306.00978)
  提出了AWQ算法，一种根据激活幅度缩放权重以保护重要权重的训练后量化方法，为LLM提供了一种快速准确的解决方案。
- [SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438) — Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han (2023)
  Journal: ICML 2023; DOI: [10.48550/arXiv.2211.10438](https://doi.org/10.48550/arXiv.2211.10438)
  介绍了SmoothQuant算法，该算法通过逐通道重新缩放来解决LLM中的激活异常值问题，从而实现更精确的低比特量化。

---

[上一节](03-%E7%90%86%E8%A7%A3%E9%87%8F%E5%8C%96%E6%95%B0%E6%8D%AE%E7%B1%BB%E5%9E%8B%E5%92%8C%E6%A0%BC%E5%BC%8F.md) · [下一节](05-%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83%20%28QAT%29%20%E7%9A%84%E8%80%83%E9%87%8F.md)
