# 循环神经网络（RNN）：缘由

来源：[原文](https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-7-intro-specialized-architectures/rnn-motivation)

[返回章节目录](README.md) · [返回课程目录](../README.md)

前馈网络，例如多层感知机（MLP），独立处理输入。给定一个输入向量 (vector) $x$，网络计算一个输出 $y$。如果你提供另一个输入向量 $x'$，它会计算一个相应的 $y'$，而不会记住就在此前处理过 $x$。当每个输入都是独立的，例如分类单张图片或分析电子表格中每行代表一个独立的观测时，这种方式运作良好。

然而，许多重要的数据类型本质上是序列化的。试想：

- **自然语言：** 句子的含义严重依赖于词语的顺序。“猫追狗”与“狗追猫”的含义截然不同。为了理解“狗”在第一个句子中的作用，你需要知道“猫”和“追”在这之前出现过。
- **时间序列数据：** 预测明天的股价可能需要知道前几天的价格。过去事件的序列包含预测信息。分析随时间变化的传感器读数通常涉及理解从序列中出现的趋势和模式。
- **语音识别：** 口语随时间展开。识别一个词语取决于前后相连的声音（音素）序列。
- **视频分析：** 理解视频中的动作需要处理一系列帧，因为运动和事件随时间发生。

对于这类数据，标准前馈网络做出的输入独立性假设不再成立。独立处理序列中的每个元素（例如句子中的一个词或时间序列中的一个数据点）会丢失隐含在顺序中的有价值的上下文 (context)信息。如果一个标准MLP在处理到句子末尾时忘记了句子的开头，它如何有效地翻译一个句子呢？它如何在不记住前面音符的情况下预测旋律中的下一个音符呢？

这种局限性凸显了对一种不同网络架构的需求，一种具有某种形式记忆的架构。我们需要能够逐步处理序列输入，同时保留关于先前步骤信息的模型。序列中特定点的输出或决策不仅应受到当前输入的影响，还应受到迄今为止已处理输入历史的影响。

这正是\*\*循环神经网络 (neural network)（RNN）\*\*背后的缘由。RNN专门设计用于处理序列。它们引入了循环机制，并维护一个内部“状态”或“记忆”，捕获关于序列中先前处理过的信息。此状态在每个步骤更新，使网络能够展现动态的时间行为，并建模输入序列中不同位置之间的依赖关系。这种处理顺序和上下文的固有能力使RNN成为处理涉及序列数据任务的基本工具。

## 参考资料

- [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1) — Jeffrey L. Elman (1990)
  Journal: Cognitive Science; Publisher: Wiley; Volume: 14; Pages: 179-211; DOI: [10.1207/s15516709cog1402_1](https://doi.org/10.1207/s15516709cog1402_1)
  介绍简单循环网络（Elman网络）的基础性著作，阐明了它们通过维护内部状态来学习和处理序列信息的能力。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本权威教材，其中有一章专门介绍循环神经网络，对其原理、架构和应用进行了全面阐述。
- [CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — Christopher Manning and Richard Socher (2023)
  Publisher: Stanford University
  一所顶尖大学课程的教材，该课程持续涵盖RNN的基础知识和用途，特别是在自然语言处理方面，与序列数据示例保持一致。

---

[上一节](05-%E5%85%B8%E5%9E%8BCNN%E6%9E%B6%E6%9E%84.md) · [下一节](07-%E5%BE%AA%E7%8E%AF%E4%B8%8E%E9%9A%90%E8%97%8F%E7%8A%B6%E6%80%81.md)
