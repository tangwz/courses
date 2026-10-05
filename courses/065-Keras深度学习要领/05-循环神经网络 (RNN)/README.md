# 第 5 章：循环神经网络 (RNN)

来源：[原章节](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns)

[返回课程目录](../README.md)

前几章主要处理的是顺序不是主要考量点的模型输入数据，例如分类静态图片。本章将把重点转向序列很重要的资料，例如文本中的句子或时间序列中的数值。我们会介绍循环神经网络 (RNN)，这是一种特别适合处理序列资料的神经网络结构。

你会学习 RNN 如何运作，通过维护一个内部状态（通常称为隐藏状态），这使它们在处理当前元素时，能“记住”序列中先前元素的信息。我们会从 Keras 中基础的 `SimpleRNN` 层开始，并考察其能力与不足之处，包括梯度消失问题。接着，我们会学习更复杂的变体，例如长短期记忆 (LSTM) 和门控循环单元 (GRU) 网络，这些网络旨在更有效地捕获更长距离的关联。你将获得实际操作经验，如何为这些模型准备序列资料，以及如何使用 Keras 实现 RNN 和 LSTM，用于文本分类等常见任务。

## 小节

- 1. [序列数据概述](01-%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E6%A6%82%E8%BF%B0.md)
- 2. [循环神经网络要点](02-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E8%A6%81%E7%82%B9.md)
- 3. [Keras中的SimpleRNN层](03-Keras%E4%B8%AD%E7%9A%84SimpleRNN%E5%B1%82.md)
- 4. [梯度消失问题](04-%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98.md)
- 5. [长短期记忆（LSTM）网络](05-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E7%BD%91%E7%BB%9C.md)
- 6. [Keras中的LSTM层](06-Keras%E4%B8%AD%E7%9A%84LSTM%E5%B1%82.md)
- 7. [门控循环单元 (GRU)](07-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRU%29.md)
- 8. [用于RNN的序列数据准备](08-%E7%94%A8%E4%BA%8ERNN%E7%9A%84%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%87%86%E5%A4%87.md)
- 9. [实践：构建RNN/LSTM用于文本分类](09-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BARNN-LSTM%E7%94%A8%E4%BA%8E%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.md)

章节测验：[在线测验](https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/quiz)
