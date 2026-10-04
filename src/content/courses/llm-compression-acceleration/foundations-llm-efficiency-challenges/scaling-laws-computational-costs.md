---
course: "llm-compression-acceleration"
chapter: "foundations-llm-efficiency-challenges"
lesson: "scaling-laws-computational-costs"
sourceId: 4295
sourceUrl: "https://apxml.com/zh/courses/llm-compression-acceleration/chapter-1-foundations-llm-efficiency-challenges/scaling-laws-computational-costs"
title: "LLM的规模法则与计算成本"
description: "分析LLM大小、性能和计算需求与规模法则的关系。"
order: 1
plots: ["plots/4295-0.json"]
sourceHash: "c600d2a18fbbebab528c64d6e98d8370c68368369514a30414c5cefe3cfe829e"
sourceCorrections: []
---

现代大语言模型 (LLM)所具有的强大能力与其规模紧密关联。理解模型性能如何与模型大小、数据量和计算量联系，是应对效率难题的根本。这种关系常通过经验性观察来表示，即**规模法则**。

### 幂律关系

大语言模型 (LLM)性能（通常通过在独立数据集上的交叉熵损失衡量）与三个主要因素常呈幂律关系。这种关系常由被称为**缩放定律**的经验性观察所体现，其基础工作由Kaplan等人于2020年首次系统性地确立，并经Hoffmann等人于2022年进一步完善。

1. **模型大小 (N)：** 模型中非嵌入 (embedding)参数 (parameter)的数量。
2. **数据集大小 (D)：** 训练过程中处理的令牌数量。
3. **计算量 (C)：** 训练使用的总浮点运算次数（FLOPs），大致与$N \times D$成比例。

核心发现是，在非瓶颈情况下，测试损失$L$与$N$、$D$和$C$呈幂律关系。例如，当计算资源和数据充足时，损失常随模型大小按以下方式变化：


$$
L(N) \approx \left( \frac{N_c}{N} \right)^{\alpha_N}
$$


这里，$N_c$和$\alpha_N$是特定于模型架构和训练设置的常数。类似的幂律关系也存在于$L(D)$和$L(C)$。这些关系表明，增加模型大小、数据量或计算量会带来递减但可预测的性能提升（损失降低）。



![LLM规模法则（示意）](plots/4295-0.json)



> 在对数-对数坐标系下，测试损失与模型参数/计算量之间的示意性幂律关系。图表显示出类似趋势，尽管噪声和特定的架构选择可能导致偏差。

### 计算量最优的规模：Chinchilla的发现

一项重要的改进随之而来，即“Chinchilla”规模法则（Hoffmann等人，2022）。之前的工作常在相对较小的数据集上训练大型模型（相较于模型大小）。Chinchilla论文表明，在给定计算预算$C$下，最优分配涉及大致按比例增大模型大小$N$和数据集大小$D$。具体而言，他们发现，在固定FLOP预算下达到最佳性能，模型应该用比以往普遍认为的更多的令牌进行训练。一个在1.4万亿令牌上训练的700亿参数 (parameter)Chinchilla模型，在多项基准测试中表现优于大得多的1750亿参数GPT-3（在3000亿令牌上训练）和5300亿参数Gopher（同样在3000亿令牌上训练）。

这表明，在给定计算预算下获得最佳性能，需要平衡增加模型参数和训练数据，而不是只优先考虑模型大小。最优比率表明$C \approx 6ND$，这意味着计算预算大约是参数量和已处理令牌数量乘积的六倍。

### 量化 (quantization)计算成本

理解规模法则需要量化所涉及的计算成本。

**训练成本：** 训练大语言模型 (LLM)的计算成本极高。主要成本是前向和反向传播 (backpropagation)中涉及的巨大矩阵乘法次数。标准Transformer的训练FLOPs粗略估计为：


$$
C_{train} \approx 6 \times N \times D
$$


这里：

- $N$是非嵌入 (embedding)参数 (parameter)的数量。
- $D$是训练数据集中的令牌数量。
- 系数6包含了前向传播（每个令牌约$2N$ FLOPs）和反向传播（每个令牌约$4N$ FLOPs，包括梯度计算）。

对于一个在3000亿令牌上训练的GPT-3模型（1750亿参数），这相当于大约$6 \times (175 \times 10^9) \times (300 \times 10^9) \approx 3.15 \times 10^{23}$ FLOPs。执行这些计算需要大规模高端加速器（GPU或TPU）集群运行数周或数月，产生高昂的硬件和能源成本。

**推理 (inference)成本：** 虽然低于训练成本，但推理成本也不容忽视，尤其是在大规模部署时。对于生成单个令牌，FLOPs大约为：


$$
C_{inference\_per\_token} \approx 2 \times N
$$


生成长度为$L$的序列大约需要$2 \times N \times L$ FLOPs。然而，推理常受**内存带宽**的瓶颈限制，而不仅仅是原始计算量（FLOPs）。这是因为，对于每个生成的令牌，整个模型的参数（$N$个参数，FP16/BF16精度下常需要$2N$字节）需要从内存（例如GPU HBM）中读取。所需时间常主要由这种内存访问决定，而非计算本身，尤其对于参数量达数十亿的大型模型。我们将在下一节更详细地分析这些瓶颈。

**内存成本：**

- **参数：** 存储模型权重 (weight)需要内存。一个使用16位精度（FP16或BF16）的$N$参数模型需要$2N$字节。对于一个1750亿参数的模型，这将是350 GB，超出单个加速器的内存容量。
- **激活值：** 在推理过程中，Transformer层内的中间激活值（特别是在注意力机制 (attention mechanism)中的键值缓存）必须存储。该KV缓存的大小随批处理大小、序列长度、层数和隐藏维度变化，对于长上下文 (context)长度，通常会占用数十或数百GB的存储空间。

这些规模法则和相关成本直接促使了优化的必要性。通过剪枝或蒸馏等技术减小模型大小（$N$）直接降低计算和内存需求。量化减少了参数和激活值的内存占用，并能在专用硬件上实现更快的计算。高效微调 (fine-tuning)方法（PEFT）降低了适配大型预训练 (pre-training)模型的成本。硬件加速技术旨在提高这些高要求计算的每秒浮点运算次数（FLOPs/second）和内存带宽。理解这些基本的规模关系为评估本课程中将介绍的优化技术的有效性提供了量化依据。

## 参考资料

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  建立了描述LLM性能与模型大小、数据集大小和计算预算之间关系的经验缩放定律。
