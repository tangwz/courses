---
course: "llm-model-sizes-hardware"
chapter: "estimating-hardware-needs"
lesson: "activation-memory"
sourceId: 4253
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs/activation-memory"
title: "考虑激活内存"
description: "简要说明推理过程中的激活也会占用内存。"
order: 2
plots: ["plots/4253-0.json"]
sourceHash: "cf58f4a32bbd1e1a758059a9ba61c5139858f5fc14a558ec740cc3d5e2c8eea1"
sourceCorrections: []
---

估算大型语言模型（LLM）的内存需求涉及多个因素。模型参数 (parameter)是此估算中的一个主要组成部分。此外，另一个主要因素是**激活**所消耗的内存。当LLM处理您的输入（例如一个问题）并生成输出（例如一个回答）时，它会逐层执行大量的计算。这些计算的中间结果被称为激活。

可以把它想象成在白板上解决一个复杂的数学问题。模型参数就像您永久写下的已学公式和常数。而激活就像您在逐步计算最终答案时，在工作区临时记下的数字和结果。正如您需要在白板上留出空间来放置这些临时笔记一样，GPU在工作时也需要内存（显存 (VRAM)）来存储这些激活。

### 为什么激活会占用显存 (VRAM)

神经网络 (neural network)中的每一层都会接收输入（可以是原始输入，也可以是来自上一层的激活），使用其参数 (parameter)进行处理，并生成新的激活作为下一层的输出。这些激活必须保留在内存中，直到它们在该特定处理步骤（通常称为“前向传播”）的后续计算中不再被需要为止。

重要的是，激活所需的显存量不像模型参数那样是固定不变的。它是动态的，并且很大程度上取决于所执行任务的特点：

1. **序列长度（上下文 (context)长度）：** 更长的输入提示或生成更长的响应，要求模型记录更多信息。这通常会导致激活尺寸更大，从而需要更多显存。想象一下，对于一个更长的问题，您需要在白板上保留更多的中间步骤。
2. **批处理大小：** 如果您同时处理多个输入序列（一个“批次”）以提高效率，那么批次中*每个*序列的激活都需要存储。运行批处理大小为4将比批处理大小为1大约需要四倍的激活内存。
3. **模型架构：** 大型语言模型的内部设计，例如层数、这些层的“宽度”（隐藏维度）以及使用的注意力机制 (attention mechanism)，直接影响所生成激活的大小和数量。

### 对总显存 (VRAM)估算的影响

因此，在估算所需的总显存时，您需要同时考虑静态模型参数 (parameter)和动态激活。一个更完整（尽管仍是简化）的视图如下：

$总显存 \approx 参数内存 + 激活内存 + 软件开销$

软件开销包含了操作系统、GPU驱动程序以及运行模型的特定人工智能框架（如PyTorch或TensorFlow）所占用的内存。



![推理过程中的显存使用](plots/4253-0.json)



> 此细分图显示，总显存使用量包括模型参数、激活和软件开销所需的内存。相对大小仅供示意。

事先精确计算所需的激活内存可能很复杂，因为它取决于运行时因素。然而，了解它的存在非常重要。仅仅基于参数的经验法则提供了一个*最低*显存估算。您必须始终预留额外空间以容纳激活和开销，特别是如果您计划使用长上下文 (context)长度或批量处理输入。这说明了为什么一个理论上根据参数大小应该能适配的模型，在实际运行中仍然可能导致“内存不足”错误。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); DOI: [10.48550/arXiv.1706.03762](https://doi.org/10.48550/arXiv.1706.03762)
  介绍Transformer架构的基础论文，该架构是现代LLM及其激活生成的基础。
- [Deep Learning Systems: Algorithms, Compilers, and Processors for Large-Scale AI](https://link.springer.com/book/10.1007/978-981-99-4704-5) — Dawei Chen, Haibin Li, Zidong Zhang (2023)
  Publisher: Springer; DOI: [10.1007/978-981-99-4704-5](https://doi.org/10.1007/978-981-99-4704-5)
  全面概述深度学习系统，包含内存管理、硬件考量和大型模型优化技术的详细讨论，与理解VRAM使用情况相关。
- [Hardware Requirements and Memory Considerations for Transformers](https://huggingface.co/docs/transformers/perf_hardware_requirements#memory-considerations) — Hugging Face (2024)
  提供Transformer模型运行内存考量和优化（包括批处理大小和硬件要求）的实用建议的官方文档。
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (2022)
  Journal: Advances in Neural Information Processing Systems 35 (NeurIPS 2022); DOI: [10.48550/arXiv.2205.14135](https://doi.org/10.48550/arXiv.2205.14135)
  介绍了一种优化的注意力算法，显著减少HBM（GPU内存）读写和激活的内存占用，展示了架构对VRAM的影响。
