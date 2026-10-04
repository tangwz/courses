---
course: "llm-model-sizes-hardware"
chapter: "model-size-hardware-connection"
lesson: "parameters-memory"
sourceId: 4217
sourceUrl: "https://apxml.com/zh/courses/llm-model-sizes-hardware/chapter-3-model-size-hardware-connection/parameters-memory"
title: "模型参数与内存占用"
description: "模型参数如何主要加载到VRAM内存中。"
order: 1
plots: []
sourceHash: "e2ecd4653c2e62fb08af076b334d1533360db8e551c9984114e5491282aa7d82"
sourceCorrections: []
---

大型语言模型是由其参数 (parameter)定义的，这些参数是训练期间学习到的数值，它们包含了模型的能力。GPU及其专用内存VRAM是大型语言模型运行涉及的**主要**硬件组件。模型参数数量与其运行所需内存量之间的直接关系是一个**重要考量**。

可以把模型的参数想象成一部巨型百科全书的全部文本。当这部百科全书储存在书架上（你电脑的磁盘存储）时，它包含了大量信息，但你无法立即阅读。要真正使用这些信息（运行大型语言模型），你需要将相关卷册（参数）搬到你的阅读桌上，这样你就能快速查阅它们。

### 将参数 (parameter)加载到活动内存中

对于大型语言模型来说，这个“阅读桌”就是电脑的活动内存。虽然可以使用系统RAM，但最有效率的参数加载位置是为了快速运行而设计的GPU VRAM。为什么是VRAM？正如第2章讨论的，GPU是为了大规模并行计算而设计的，这正是处理大型语言模型中复杂数学运算所需要的类型。为了达到其卓越的速度，GPU需要对其正在处理的数据进行极快的访问。将模型的参数直接加载到连接GPU的VRAM中，就能提供这种高速访问。

如果参数只在系统RAM中，GPU将不得不持续等待数据通过相对较慢的连接（系统总线）从RAM传输过来，这将产生一个明显的瓶颈，并大幅降低性能。因此，为了高效地进行大型语言模型推理 (inference)，主要目标是将所有必要的模型参数放入可用的VRAM中。

> 模型参数从较慢的磁盘存储加载到GPU的快速VRAM中，以便在推理过程中实现高效处理。

### 基本关系：更多参数 (parameter) = 更多内存

大型语言模型中的每个参数本质上都是一个数字。存储数十亿个这样的数字自然需要大量的内存。核心关系很简单明了：

- **参数越多的模型，存储这些参数所需的内存就越多。**

如果一个模型有70亿参数，你需要足够的内存来存储70亿个数字。如果另一个模型有700亿参数，你大约需要十倍的内存来仅存储其参数。

这是连接模型大小与硬件需求的最基本因素。参数的庞大数量决定了所需的最小内存容量，主要体现在VRAM方面。尽管激活内存（我们稍后会提及）等其他因素也会消耗VRAM，但参数本身所需的空间通常是最大的组成部分。

然而，每个单独的参数数字会占用多少空间呢？这取决于用于存储它的数值格式，即*精度*。我们将在下一节关于数据类型的内容中查看这个重要细节。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  本书为深度学习提供了全面的理论和实践基础，涵盖了神经网络中参数的作用以及训练和推理的计算方面，这些对于理解内存消耗至关重要。
- [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) — Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 5998-6008; DOI: [10.5591/978-1-57766-302-3.375](https://doi.org/10.5591/978-1-57766-302-3.375)
  这篇开创性的论文介绍了Transformer架构，该架构构成了大多数大型语言模型的基础。理解Transformer的结构有助于把握为什么这些模型拥有如此庞大的参数数量，进而导致高内存需求。
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) — NVIDIA Corporation (Latest Edition)
  Publisher: NVIDIA Corporation
  提供了NVIDIA GPU架构的详细信息，包括设备内存（VRAM）的层次结构和特性，以及其在深度学习等应用中高性能计算的作用。
- [CS224N: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — Diyi Yang, Tatsunori Hashimoto (2025)
  Publisher: Stanford University
  提供了讲义和作业，讨论了训练和部署大型语言模型的计算要求和实际考虑，包括对内存的讨论。
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei (2020)
  Journal: arXiv preprint arXiv:2001.08361; DOI: [10.48550/arXiv.2001.08361](https://doi.org/10.48550/arXiv.2001.08361)
  这篇论文直接探讨了模型参数如何影响性能，以及因此而产生的有效LLM操作所需的计算和内存资源。
