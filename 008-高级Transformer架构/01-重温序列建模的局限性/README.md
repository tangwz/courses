# 第 1 章：重温序列建模的局限性

来源：[原章节](https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-1-revisiting-sequence-modeling-limitations)

[返回课程目录](../README.md)

为理解Transformer架构的设计动因，我们首先回顾了之前用于序列建模的主要方法。循环神经网络（RNNs）、长短期记忆网络（LSTMs）和门控循环单元（GRUs）在处理序列数据方面代表了重要的进展。然而，其固有的结构存在一些局限性，阻碍了进展，尤其是在序列长度不断增加的情况下。

本章将审视这些具体的挑战。我们将讨论：

*   循环模型固有的顺序处理特性及其对计算效率的影响。
*   训练深度循环网络时遇到的数学上的难题，特别是梯度消失和梯度爆炸问题（例如，早期时间步 $t$ 时梯度消失，表现为 $ \frac{\partial L}{\partial \theta_t} \approx 0 $）。
*   LSTMs和GRUs中的门控机制如何尝试缓解梯度问题。
*   循环模型在有效建模输入序列中很长距离的依赖关系方面持续存在的问题。

认识到这些限制，能为理解Transformer模型在后续章节中引入的架构创新提供必要的背景信息。

## 小节

- 1. [循环网络中的顺序计算](01-%E5%BE%AA%E7%8E%AF%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E9%A1%BA%E5%BA%8F%E8%AE%A1%E7%AE%97.md)
- 2. [梯度消失与梯度爆炸问题](02-%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E4%B8%8E%E6%A2%AF%E5%BA%A6%E7%88%86%E7%82%B8%E9%97%AE%E9%A2%98.md)
- 3. [长短期记忆（LSTM）门控机制](03-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E9%97%A8%E6%8E%A7%E6%9C%BA%E5%88%B6.md)
- 4. [门控循环单元 (GRU) 架构](04-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRU%29%20%E6%9E%B6%E6%9E%84.md)
- 5. [远距离依赖的挑战](05-%E8%BF%9C%E8%B7%9D%E7%A6%BB%E4%BE%9D%E8%B5%96%E7%9A%84%E6%8C%91%E6%88%98.md)
- 6. [循环模型中的并行化限制](06-%E5%BE%AA%E7%8E%AF%E6%A8%A1%E5%9E%8B%E4%B8%AD%E7%9A%84%E5%B9%B6%E8%A1%8C%E5%8C%96%E9%99%90%E5%88%B6.md)
