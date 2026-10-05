# 第 3 章：回顾序列处理架构

来源：[原章节](https://apxml.com/zh/courses/how-to-build-a-large-language-model/chapter-3-revisiting-sequence-processing-architectures)

[返回课程目录](../README.md)

为了理解 Transformer 架构的创新之处，首先了解其之前的序列处理模型结构是有帮助的。本章简要回顾循环神经网络 (RNN) 及其更为复杂的变体：长短期记忆 (LSTM) 网络和门控循环单元 (GRU)。

我们将考察：
*   RNN 中使用隐藏状态进行序列处理的核心思想。
*   简单 RNN 面临的难题，例如由于梯度消失而难以捕捉长期依赖关系。
*   LSTM 和 GRU 中的门控机制如何被设计来缓解这些问题。
*   这些循环架构在序列到序列 (seq2seq) 任务中的应用。

此次回顾为理解为何注意力机制和 Transformer 架构在序列数据建模中代表了重大转变提供了背景，我们将在后续章节中进行讲解。

## 小节

- 1. [循环神经网络 (RNN) 的基本内容](01-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28RNN%29%20%E7%9A%84%E5%9F%BA%E6%9C%AC%E5%86%85%E5%AE%B9.md)
- 2. [简单RNN的局限性](02-%E7%AE%80%E5%8D%95RNN%E7%9A%84%E5%B1%80%E9%99%90%E6%80%A7.md)
- 3. [长短期记忆（LSTM）网络](03-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E7%BD%91%E7%BB%9C.md)
- 4. [门控循环单元 (GRU)](04-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRU%29.md)
- 5. [基于RNN的序列到序列模型](05-%E5%9F%BA%E4%BA%8ERNN%E7%9A%84%E5%BA%8F%E5%88%97%E5%88%B0%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B.md)
