---
course: "llm-compression-acceleration"
chapter: "foundations-llm-efficiency-challenges"
lesson: "memory-compute-bottlenecks-inference"
sourceId: 4297
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/memory-compute-bottlenecks-inference"
title: "LLM推理中的内存带宽和计算瓶颈"
description: "辨识和分析LLM推理操作中的内存带宽和计算限制。"
order: 2
plots: ["plots/4297-0.json"]
sourceHash: "afaec0eb03425007c08196cbf98ae2aa1aab489886d73f048fe2492b1f3397de"
sourceCorrections: []
---

现代大型语言模型（LLM）的庞大规模，受缩放法则制约，直接带来了庞大的计算和内存需求。虽然训练这些模型是一项庞大工程，但将其部署用于推理 (inference)则面临一系列严峻的效率挑战。在推理过程中，尤其是在词元 (token)按序生成的自回归 (autoregressive)生成任务中，内存带宽和计算能力是常决定性能的两个主要限制因素。辨明哪个瓶颈占据主导，是有效优化的基础。

### 内存墙：带宽瓶颈

对于许多LLM推理 (inference)工作负载，特别是对延迟敏感的逐词元 (token)文本生成任务，主要制约因素通常是**内存带宽**。这指的是数据（主要是模型的参数 (parameter)/权重 (weight)）从主内存（通常是DRAM）传输到实际进行计算的处理单元（GPU SRAM、缓存、寄存器）的速率。

考虑一个拥有数十亿参数的大型模型。即使采用FP16或BF16等低精度格式，其总大小也能达到数十到数百千兆字节。

- **参数大小与片上内存：** 现代GPU的极快片上内存（SRAM、缓存）相较于其大量的较慢DRAM池而言容量较小。例如，一块高端GPU可能拥有数百兆字节的L2缓存，但有数十千兆字节的HBM（高带宽内存）。绝大部分LLM的参数（$W$）存储在这些较慢的HBM/DRAM中。
- **推理流程：** 在推理过程中，特别是自回归 (autoregressive)解码时，每个生成的词元都需要通过整个网络进行处理。这包括将每一层的权重从DRAM顺序加载到计算单元。即使计算本身速度很快，等待这些权重到达所花费的时间也会成为主要影响因素。
- **Transformer组件：** 自注意力 (self-attention)机制 (attention mechanism)以其内存访问模式而闻名。计算注意力分数涉及对查询（$Q$）、键（$K$）和值（$V$）矩阵的操作。在自回归解码中，$K$和$V$张量随每个生成的词元而增长（即KV缓存）。访问和更新此缓存，以及加载注意力层权重，需要大量数据移动，这常受DRAM带宽限制。在单词元解码过程中，注意力层和前馈层中常见的矩阵-向量 (vector)乘法也趋于受内存限制，因为每字节加载的数据计算量相对较低。

我们可以通过操作的**算术强度**（$AI$）来理解这一点，它定义为浮点运算（FLOPs）与从主内存移动的数据字节数之比：


$$
AI = \frac{\text{总浮点运算}}{\text{总访问字节数 (DRAM)}}
$$


算术强度较低的操作通常受内存带宽限制。许多单词元LLM推理操作属于此类。处理器等待数据的时间多于执行计算的时间。这种情况常被称为**内存限制**。

> 推理期间数据流的简化视图，突显了权重所在的较慢主内存（DRAM）与为计算单元提供数据的较快片上缓存之间可能存在的瓶颈。

### 计算上限：当计算占据主导

反之，当处理器的计算速度（以FLOPs衡量——每秒浮点运算次数）是限制因素时，操作便是**计算限制**的。这通常发生在以下情况：

1. **高算术强度：** 操作对于从内存加载的每个字节数据执行大量计算。大型矩阵-矩阵乘法，特别是当矩阵能放入较快缓存时，是主要的例子。
2. **足够的数据局部性：** 所需数据已存在于快速片上内存（SRAM/缓存）中，最大程度减少了从较慢DRAM获取数据的需要。
3. **大批量大小：** 同时处理多个输入序列（增加批量大小）常使平衡趋向于受计算限制。这会将加载权重 (weight)的成本分摊到更多计算上，使得矩阵-矩阵运算比矩阵-向量 (vector)运算更具主导性。然而，大批量会增加延迟，使其不太适合实时交互应用。

在大型语言模型（LLM）的背景下：

- **前馈网络（FFNs）：** Transformer中的FFN层涉及大型矩阵乘法。如果输入批量大小足够大，或者如果操作融合等方法将中间结果保存在快速内存中，这些操作可能会受计算限制。
- **预填充阶段：** 在处理初始提示词 (prompt)（在自回归 (autoregressive)生成开始前）时，计算常可在提示词的词元 (token)之间并行化。这涉及比随后的逐词元生成更大的矩阵-矩阵运算，可能使预填充阶段比解码阶段更受计算限制。

硬件能力在此发挥直接作用。处理器的理论峰值浮点运算次数（例如，通过NVIDIA的Tensor Cores等专用单元提升）设定了受计算限制性能的上限。

### 两者关系：解码对比预填充与批量处理

认识到瓶颈并非固定不变很重要；它在很大程度上取决于具体操作、推理 (inference)策略和硬件。

- **预填充对比解码：** 如前所述，处理初始提示词 (prompt)（预填充）常涉及跨词元 (token)的并行计算，导致可能受计算限制的大型矩阵乘法。随后的自回归 (autoregressive)解码阶段一次生成一个词元，严重依赖对内存带宽敏感的操作，例如注意力查找和矩阵-向量 (vector)乘法。
- **批量大小：** 增加批量大小通常会提高算术强度，促使操作趋向于受计算限制。然而，这会以增加延迟为代价。为最大吞吐量 (throughput)优化（例如离线处理）可能偏好大批量并针对计算限制，而为最小延迟优化（例如聊天机器人）则需要小批量（常为批量大小1）并着重于解决内存带宽限制。



![LLM操作的房顶线模型](plots/4297-0.json)



> 房顶线图展示性能限制。算术强度较低的操作（左侧）受限于内存带宽（橙线），而算术强度较高的操作（右侧）受限于峰值计算性能（绿线）。典型的LLM推理操作常根据具体情况（例如单词元解码注意力与批量FFN）落入不同区域。

辨别LLM工作负载是主要受内存限制还是受计算限制，是选择合适优化策略的第一步。量化 (quantization)和优化的内存访问模式等方法直接针对内存墙问题，而剪枝（减少FLOPs）或使用更快计算核等方法则解决计算上限问题。接下来的章节将提供应对这两个基本瓶颈的工具和方法。

## 参考资料

- [Roofline: An architectural model for floating-point performance on multicore architectures](https://dl.acm.org/doi/10.1145/1654009.1654015) — Samuel Williams, Andrew Waterman, David Patterson (2009)
  Journal: SC '09: Proceedings of the ACM/IEEE Conference on Supercomputing; Publisher: ACM; Pages: 1-11; DOI: [10.1145/1654009.1654015](https://doi.org/10.1145/1654009.1654015)
  介绍了基础的Roofline模型，一个用于识别应用程序是内存受限还是计算受限的视觉性能模型。
- [Computer Architecture: A Quantitative Approach](https://www.elsevier.com/books/computer-architecture/hennessy/978-0-12-811905-1) — John L. Hennessy and David A. Patterson (2017)
  Publisher: Morgan Kaufmann
  计算机体系结构原理的标准教材，其内容涵盖内存层次结构、缓存一致性和性能指标，这些都是内存和计算瓶颈概念的基础。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  定义了大型语言模型中模型大小、数据集大小、计算和性能之间关系的开创性论文，为理解其巨大的资源需求提供了背景。
