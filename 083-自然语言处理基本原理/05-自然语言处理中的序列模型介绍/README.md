# 第 5 章：自然语言处理中的序列模型介绍

来源：[原章节](https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro)

[返回课程目录](../README.md)

前几章讨论了文本表示方法，如TF-IDF，这些方法通常不考虑词语的顺序。然而，语言本身是序列化的；词语的排列方式带有重要意义。本章着重介绍旨在处理数据顺序很重要的模型。

我们将从循环神经网络（RNNs）的基本原理讲起，解释它们如何通过维护状态来处理序列。你将了解训练RNN时一个常见难题，即梯度消失问题。之后，我们将查看为解决此局限性而发展出的更复杂的架构：长短期记忆（LSTM）网络和门控循环单元（GRUs）。我们将讲解它们的核心机制，并最后讨论这些序列感知的模型如何应用于各种自然语言任务，以及一个实际的实现练习。

## 小节

- 1. [序列感知的必要性](01-%E5%BA%8F%E5%88%97%E6%84%9F%E7%9F%A5%E7%9A%84%E5%BF%85%E8%A6%81%E6%80%A7.md)
- 2. [循环神经网络（RNN）基础](02-%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%88RNN%EF%BC%89%E5%9F%BA%E7%A1%80.md)
- 3. [理解梯度消失问题](03-%E7%90%86%E8%A7%A3%E6%A2%AF%E5%BA%A6%E6%B6%88%E5%A4%B1%E9%97%AE%E9%A2%98.md)
- 4. [长短期记忆（LSTM）网络](04-%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%EF%BC%88LSTM%EF%BC%89%E7%BD%91%E7%BB%9C.md)
- 5. [门控循环单元 (GRUs)](05-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRUs%29.md)
- 6. [将序列模型应用于文本](06-%E5%B0%86%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B%E5%BA%94%E7%94%A8%E4%BA%8E%E6%96%87%E6%9C%AC.md)
- 7. [动手实践：构建一个简单的序列模型](07-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E5%BA%8F%E5%88%97%E6%A8%A1%E5%9E%8B.md)

章节测验：[在线测验](https://apxml.com/zh/courses/nlp-fundamentals/chapter-5-nlp-sequence-models-intro/quiz)
