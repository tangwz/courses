# 第 18 章：大型语言模型训练的硬件考量

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-18-hardware-considerations-llm-training)

[返回课程目录](../README.md)

训练大型语言模型需要大量计算资源。这些模型规模庞大，通常包含数十亿或数万亿参数，对传统硬件的能力造成了挑战。本章将介绍使大型语言模型训练得以实现的硬件系统。

我们将研究图形处理单元（GPU）和张量处理单元（TPU）的具体特性，它们专门用于加速深度学习计算。你将了解内存需求，尤其是高带宽内存（HBM），以及NVLink和InfiniBand等高速互连技术在分布式训练配置中的作用。我们还将分析硬件选择中涉及的权衡，包括成本、性能和可用性。扎实理解这些硬件考量对于规划和管理大规模模型训练来说非常重要。

## 小节

- 1. [GPU 架构 (NVIDIA Ampere, Hopper)](01-GPU%20%E6%9E%B6%E6%9E%84%20%28NVIDIA%20Ampere%2C%20Hopper%29.md)
- 2. [TPU 架构（Google TPU）](02-TPU%20%E6%9E%B6%E6%9E%84%EF%BC%88Google%20TPU%EF%BC%89.md)
- 3. [内存需求（HBM、GPU显存）](03-%E5%86%85%E5%AD%98%E9%9C%80%E6%B1%82%EF%BC%88HBM%E3%80%81GPU%E6%98%BE%E5%AD%98%EF%BC%89.md)
- 4. [互连技术 (NVLink, InfiniBand)](04-%E4%BA%92%E8%BF%9E%E6%8A%80%E6%9C%AF%20%28NVLink%2C%20InfiniBand%29.md)
- 5. [硬件选择的权衡 (成本、性能、可用性)](05-%E7%A1%AC%E4%BB%B6%E9%80%89%E6%8B%A9%E7%9A%84%E6%9D%83%E8%A1%A1%20%28%E6%88%90%E6%9C%AC%E3%80%81%E6%80%A7%E8%83%BD%E3%80%81%E5%8F%AF%E7%94%A8%E6%80%A7%29.md)
