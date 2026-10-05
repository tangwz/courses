# 第 5 章：长短期记忆 (LSTM) 网络

来源：[原章节](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm)

[返回课程目录](../README.md)

我们之前讨论了训练简单循环神经网络（RNN）的困难之处，特别是梯度消失和梯度爆炸问题。这些问题使得基本RNN难以捕捉序列中相距较远元素之间的依赖关系。

本章介绍长短期记忆（LSTM）网络，这是一种专门的RNN架构，旨在克服这些局限。我们将观察让LSTM能够选择性地记忆或遗忘长序列中信息的核心组成部分。

您将学到：

*   调节网络内信息流动的*门控机制*的原理。
*   LSTM单元的详细结构，包括**遗忘门**、**输入门**和**输出门**。
*   **细胞状态**如何作为信息的传送带，使其在网络中传输时极少衰减。
*   控制LSTM单元内部更新的数学运算（$ \sigma $，$ \tanh $）。
*   为什么LSTM在需要建模长距离依赖关系的任务中，通常比简单RNN更有效。

在本章结束时，您将理解LSTM单元的内部工作原理，并认识到它们在现代序列建模中的重要性。

## 小节

- 1. [通过门控应对循环神经网络的局限](01-%E9%80%9A%E8%BF%87%E9%97%A8%E6%8E%A7%E5%BA%94%E5%AF%B9%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%9A%84%E5%B1%80%E9%99%90.md)
- 2. [LSTM单元结构](02-LSTM%E5%8D%95%E5%85%83%E7%BB%93%E6%9E%84.md)
- 3. [遗忘门](03-%E9%81%97%E5%BF%98%E9%97%A8.md)
- 4. [输入门](04-%E8%BE%93%E5%85%A5%E9%97%A8.md)
- 5. [更新细胞状态](05-%E6%9B%B4%E6%96%B0%E7%BB%86%E8%83%9E%E7%8A%B6%E6%80%81.md)
- 6. [输出门](06-%E8%BE%93%E5%87%BA%E9%97%A8.md)
- 7. [LSTM单元中的信息流动](07-LSTM%E5%8D%95%E5%85%83%E4%B8%AD%E7%9A%84%E4%BF%A1%E6%81%AF%E6%B5%81%E5%8A%A8.md)
- 8. [LSTM 的优势](08-LSTM%20%E7%9A%84%E4%BC%98%E5%8A%BF.md)

章节测验：[在线测验](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/quiz)
