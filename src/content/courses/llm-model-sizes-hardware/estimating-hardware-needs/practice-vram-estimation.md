---
course: "llm-model-sizes-hardware"
chapter: "estimating-hardware-needs"
lesson: "practice-vram-estimation"
sourceId: 4261
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs/practice-vram-estimation"
title: "实践：简单的显存估算"
description: "使用经验估算方法，计算假定大语言模型尺寸的近似显存需求。"
order: 5
plots: ["plots/4261-0.json"]
sourceHash: "1e10c4de412a19ef261c58b3f119ba8db9e9a27b4ca5e968d2895b7dc34ad181"
sourceCorrections: []
---

让我们将讨论过的经验估算方法应用起来。请记住，这些都是*估算值*，主要关注于存储模型权重 (weight)所需的内存。实际使用量会更高，因为会涉及到激活值、操作系统开销以及您使用的特定软件。

我们的主要估算工具是这个关系式：

$所需 \, 显存 \approx 参数 \, 数量 \times 每个 \, 参数的 \, 字节数$

我们将关注FP16（16位浮点数）精度，这在推理 (inference)中非常常见。在FP16中，每个参数 (parameter)需要2字节的存储空间。

### 示例1：估算70亿参数 (parameter)模型（FP16）的显存 (VRAM)需求

考虑一个通常被称为“7B”的模型，这意味着它大约有70亿个参数。

1. **参数数量：** 7,000,000,000
2. **每个参数的字节数（FP16）：** 2字节
3. **计算：**
   $VRAM_{FP16} \approx 7,000,000,000 \, 参数 \times 2 \, 字节/参数$
   $VRAM_{FP16} \approx 14,000,000,000 \, 字节$

要将字节转换为吉字节（GB），我们除以 $1024^3$（或者大致除以十亿进行快速估算）。

$14,000,000,000 \, 字节 \div (1024 \times 1024 \times 1024) \approx 13.04 \, GB$

**结果：** 一个70亿参数模型在FP16精度下运行，仅仅为了存储模型权重 (weight)，大约需要**13-14 GB的显存**。

### 示例2：估算130亿参数 (parameter)模型（FP16）的显存 (VRAM)需求

现在我们来看一个稍大一点的“13B”模型。

1. **参数数量：** 13,000,000,000
2. **每个参数的字节数（FP16）：** 2字节
3. **计算：**
   $VRAM_{FP16} \approx 13,000,000,000 \, 参数 \times 2 \, 字节/参数$
   $VRAM_{FP16} \approx 26,000,000,000 \, 字节$

转换为GB：

$26,000,000,000 \, 字节 \div (1024^3) \approx 24.21 \, GB$

**结果：** 一个130亿参数模型在FP16精度下，其权重 (weight)需要大约**24-25 GB的显存**。这已经超出了许多消费级GPU的显存容量。

### 示例3：比较70亿模型在FP16和INT8下的显存 (VRAM)需求

让我们再次审视70亿参数 (parameter)模型，但这次考虑通过量化 (quantization)使用INT8（8位整数）精度。在INT8中，每个参数仅需要1字节。

1. **参数数量：** 7,000,000,000
2. **每个参数的字节数（INT8）：** 1字节
3. **计算：**
   $VRAM_{INT8} \approx 7,000,000,000 \, 参数 \times 1 \, 字节/参数$
   $VRAM_{INT8} \approx 7,000,000,000 \, 字节$

转换为GB：

$7,000,000,000 \, 字节 \div (1024^3) \approx 6.52 \, GB$

**结果：** 通过使用INT8量化，70亿参数模型的权重 (weight)显存需求降至大约**6.5-7 GB**。这使得它在显存较少的GPU上运行成为可能，例如那些拥有8GB或12GB显存的GPU，尽管您仍然需要为激活值和开销预留空间。



![70亿参数模型权重显存估算（不同精度）](plots/4261-0.json)



> 仅70亿参数模型的权重所需的估算显存，使用FP16（16位浮点数）和INT8（8位整数）精度。

### 关于权重 (weight)的考量

请记住，这些计算仅提供了模型权重的基准。您还需要*额外*的显存 (VRAM)用于：

- **激活值：** 推理 (inference)过程中执行的中间计算。所需量很大程度上取决于上下文 (context)长度（模型一次处理的文本量）和批处理大小（同时处理的请求数量）。
- **KV缓存：** 存储关于正在生成的序列的信息，随生成文本的长度而增加。
- **软件开销：** 运行时环境、库（如CUDA）和操作系统都会占用一些显存。

一个更安全的经验估算方法是，在权重显存估算值之上增加20-40%的缓冲区，或者简单地确保您的GPU显存显著多于计算出的权重需求。例如，要流畅运行70亿参数 (parameter)的FP16模型（权重估算需要13-14 GB），拥有16GB或理想情况下24GB显存的GPU会更合适，特别是对于更长的上下文。

将这些估算值与您硬件的规格进行比较（如“检查硬件规格”一节中所述），以判断在您的系统上运行特定模型是否可行。

## 参考资料

- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  一篇关于神经语言模型缩放规律的基础论文，讨论了模型大小、数据集大小和计算预算如何影响语言模型性能，确立了参数数量对模型扩展的重要性。
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) — Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer (2022)
  Journal: NeurIPS 2022; DOI: [10.48550/arXiv.2208.07339](https://doi.org/10.48550/arXiv.2208.07339)
  提出了一种针对大型语言模型执行8位矩阵乘法的方法，与本节讨论的INT8量化带来的内存效率提升直接相关。
