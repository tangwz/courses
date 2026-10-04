# LSTM 的优势

来源：[原文](https://apxml.com/zh/courses/rnns-and-sequence-modeling/chapter-5-long-short-term-memory-lstm/advantages-of-lstms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练简单循环神经网络 (neural network)（RNN）存在一些重要困难，最显著的是梯度消失和梯度爆炸问题。这些问题源于在反向传播 (backpropagation)过程中，同一权重 (weight)矩阵和激活函数 (activation function)在不同时间步上的重复应用。因此，简单 RNN 难以捕获序列中相距较远元素之间的关系，限制了它们在需要长期上下文 (context)的任务上的表现。

长短期记忆（LSTM）网络是专门为解决这些局限而开发的。它们的主要优点在于其精巧的内部结构，以**单元状态**和**门控机制**为核心，这使得它们能够比简单 RNN 更有效地调节信息流。

以下是 LSTM 通常表现优于简单 RNN 的原因，特别是在处理长序列时：

1. **应对梯度消失和梯度爆炸：** LSTM 的核心创新点是单元状态 ($C_t$)，它通常被形象地比作一条贯穿整个 LSTM 单元链的传送带。信息可以通过受严格控制的门加入或移除单元状态。重要的是，单元状态更新涉及由遗忘门 ($f_t$) 和输入门 ($i_t$) 控制的按元素进行的加法和乘法运算。

   回顾单元状态更新公式：

   
   $$
   C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
   $$
   

   选择性遗忘的过去状态 ($f_t \odot C_{t-1}$) 与新的候选信息 ($i_t \odot \tilde{C}_t$) 之间的相加作用 ($+$) 意味着梯度流经单元状态时，相较于简单 RNN 中重复的矩阵乘法，更不容易消失或爆炸。门使用 Sigmoid 激活函数 ($\sigma$) 输出 0 到 1 之间的值，它们的作用如同控制器，决定遗忘多少旧状态信息以及添加多少新信息。这种结构为梯度在多个时间步上的传播提供了更稳定的路径。
2. **学习长期依赖关系：** 因为单元状态允许信息在未被门明确修改的情况下基本保持不变地流动，所以 LSTM 能够在较长时间内维持上下文信息。遗忘门学会识别并丢弃过去的不相关信息，而输入门学会识别并存储重要的新信息。这种选择性记忆能力对于需要理解多步之前上下文的任务是根本的。例如，在处理句子“我在法国长大……我说流利的法语”时，LSTM 能够跨越多个中间词携带“法国”的信息，以正确理解“法语”的上下文。简单 RNN 可能会因梯度衰减而丢失此上下文。
3. **对记忆的明确控制：** 与简单 RNN 不同，简单 RNN 中的隐藏状态试图通过单一变换隐式捕获所有相关的过去信息，LSTM 将记忆（单元状态 $C_t$）与输出计算（隐藏状态 $h_t$）明确分离。输出门 ($o_t$) 决定单元状态的哪些部分与当前预测或传递到下一个时间步的隐藏状态相关。这提供了对何时以及如何使用信息的更精细控制。

> 比较说明了简单 RNN 单元和 LSTM 单元之间信息流的不同之处。LSTM 引入了单独的单元状态路径，带有门控、加法更新，有助于随时间保持信息。

总之，LSTM 的架构改进，主要包括单元状态和门控机制（遗忘门、输入门、输出门），相较于简单 RNN 提供了重要的优势：

- 它们缓解了梯度消失和梯度爆炸问题，使得训练更加稳定。
- 它们具有在长序列中选择性记忆相关信息并遗忘不相关细节的能力。
- 它们提供了对记忆内容以及传递给输出的信息更精确的控制。

这些特点使 LSTM 成为一个强大的工具，适用于广泛的序列建模任务，从自然语言处理到时间序列分析，在这些任务中，捕获长期模式通常对于良好的表现非常必要。

## 参考资料

- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: The MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍LSTM架构的奠基性论文，详细阐述了其设计如何克服传统RNN在长期依赖和梯度问题方面的局限。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 404-417
  一本广泛使用的教科书，提供了循环神经网络的深入理论解释，其中包括专门介绍LSTM及其在序列建模中益处的章节。
- [CS224n: Natural Language Processing with Deep Learning, Lecture Notes: Recurrent Neural Networks (RNNs) and LSTMs](https://web.stanford.edu/class/cs224n/) — Tatsunori Hashimoto, Chris Manning (2023)
  Publisher: Stanford University
  来自知名大学课程的讲义，对RNN的局限性以及LSTM解决这些问题的机制提供了清晰易懂的学术解释。

---

[上一节](07-LSTM%E5%8D%95%E5%85%83%E4%B8%AD%E7%9A%84%E4%BF%A1%E6%81%AF%E6%B5%81%E5%8A%A8.md) · [下一节](../06-%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83%20%28GRUs%29/01-%E4%BB%8B%E7%BB%8DGRU%EF%BC%9A%E4%B8%80%E7%A7%8D%E6%9B%B4%E7%AE%80%E6%B4%81%E7%9A%84%E9%97%A8%E6%8E%A7%E6%9E%B6%E6%9E%84.md)
