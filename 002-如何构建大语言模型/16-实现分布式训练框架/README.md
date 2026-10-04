# 第 16 章：实现分布式训练框架

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-16-implementing-distributed-training-frameworks)

[返回课程目录](../README.md)

训练真正的大模型，不只是要理解像数据并行 ($DP$)、张量并行 ($TP$) 和流水线并行 ($PP$) 这样的并行策略；还需要专为处理其复杂性而设计的工具。本章将从之前讨论的理论思路转向使用专门框架的实际运用。

你将学习如何通过配置和使用 DeepSpeed 和 Megatron-LM 等流行库来运用这些策略。我们将介绍 DeepSpeed 的 ZeRO 内存优化设置（阶段 $1$、$2$ 和 $3$），以及使用 Megatron-LM 配置张量并行和流水线并行。本章结束时，你将能够把分布式训练理论付诸实践，应用到你自己的大模型项目中。

## 小节

- 1. [分布式训练库概述](01-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83%E5%BA%93%E6%A6%82%E8%BF%B0.md)
- 2. [DeepSpeed 介绍](02-DeepSpeed%20%E4%BB%8B%E7%BB%8D.md)
- 3. [使用 DeepSpeed ZeRO 优化](03-%E4%BD%BF%E7%94%A8%20DeepSpeed%20ZeRO%20%E4%BC%98%E5%8C%96.md)
- 4. [Megatron-LM 介绍](04-Megatron-LM%20%E4%BB%8B%E7%BB%8D.md)
- 5. [配置 Megatron-LM 中的张量和流水线并行](05-%E9%85%8D%E7%BD%AE%20Megatron-LM%20%E4%B8%AD%E7%9A%84%E5%BC%A0%E9%87%8F%E5%92%8C%E6%B5%81%E6%B0%B4%E7%BA%BF%E5%B9%B6%E8%A1%8C.md)
- 6. [结合框架与策略](06-%E7%BB%93%E5%90%88%E6%A1%86%E6%9E%B6%E4%B8%8E%E7%AD%96%E7%95%A5.md)
