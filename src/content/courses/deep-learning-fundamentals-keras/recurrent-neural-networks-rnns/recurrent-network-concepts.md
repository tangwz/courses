---
course: "deep-learning-fundamentals-keras"
chapter: "recurrent-neural-networks-rnns"
lesson: "recurrent-network-concepts"
sourceId: 4958
sourceUrl: "https://apxml.com/zh/courses/deep-learning-fundamentals-keras/chapter-5-recurrent-neural-networks-rnns/recurrent-network-concepts"
title: "循环神经网络要点"
description: "阐述循环神经网络的核心思想，包括隐藏状态和逐步处理序列。"
order: 2
plots: []
sourceHash: "72366adc07b6ba84fa00d9bab71cb3d749d253246a89054109c1ee96cf9976b9"
sourceCorrections: []
---

“正如本章引言中所述，许多问题都涉及序列数据，其中元素的顺序对理解信息起着决定性的作用。试想理解一个句子、预测某人将要输入的下一个词、分析随时间变化的股市走势，或者判读机器的传感器读数。标准的前馈神经网络 (neural network)，例如我们见过的全连接网络或卷积神经网络 (CNN)，都是独立处理输入的。它们在处理当前输入时，不会固有地保留过去输入的信息，这对于序列任务是一个显著的限制。”

在处理像文本、语音或时间序列这样的顺序数据时，标准神经网络的一个主要限制是它们无法保持对先前输入的信息。循环神经网络（RNN）正是为了解决这一限制而设计的。RNN的主要思想是**循环**：网络对序列中的每个元素执行相同的操作，但每个元素的输出不仅取决于当前输入，还取决于先前元素的结果。这是通过在网络结构中引入一个“循环”来实现的。

### 隐藏状态：网络中的记忆

想象一下逐词处理一个句子。为了把握“猫追老鼠，然后它跑了”这句话中“它”的含义，你需要记住“它”指的是猫还是老鼠。RNN通过一个内部的**隐藏状态**（通常表示为$h$）来完成这种上下文 (context)保留。

在序列中的每一步$t$（例如，处理第$t$个词或第$t$个时间点），RNN接收两个输入：

1. 当前输入元素$x_t$。
2. 来自上一步的隐藏状态$h_{t-1}$。

然后它计算新的隐藏状态$h_t$，并可选地计算一个输出$y_t$。重点在于，$h_t$的计算使用了$h_{t-1}$。这形成了一个依赖链，其中来自早期步骤的信息可以通过隐藏状态在序列中传递。这使得网络能够拥有某种“记忆”，保留来自过去元素的上下文信息。

时间步$t$时隐藏状态的更新规则可以表示为：


$$
h_t = f(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
$$


时间步$t$的输出（如果每一步都需要）可以表示为：


$$
y_t = g(W_{hy}h_t + b_y)
$$


此处：

- $x_t$是时间步$t$的输入。
- $h_{t-1}$是来自前一时间步$t-1$的隐藏状态。
- $h_t$是时间步$t$的新隐藏状态。
- $y_t$是时间步$t$的输出。
- $W_{xh}$、$W_{hh}$和$W_{hy}$是权重 (weight)矩阵（分别为输入到隐藏层、隐藏层到隐藏层以及隐藏层到输出层）。
- $b_h$和$b_y$是偏置 (bias)向量 (vector)。
- $f$和$g$是激活函数 (activation function)（例如，对于$f$可以是`tanh`或`relu`，对于$g$根据任务可以是`softmax`或`sigmoid`）。

在所有时间步中，使用*相同*的权重集（$W_{xh}$、$W_{hh}$、$W_{hy}$）和偏置（$b_h$、$b_y$）。这种参数 (parameter)共享使得RNN高效，并能够泛化序列中不同位置的模式。

### 展开RNN

为了更好地观察信息的流动，通常会将RNN循环按序列长度“展开”。想象序列有$T$个时间步。展开意味着创建网络的$T$个副本，每个时间步一个，并将一个时间步的隐藏状态输出连接到下一个时间步的隐藏状态输入。

> 一个随时间展开的RNN。每个`RNN 单元`块表示在不同时间步应用相同的一组权重 (weight)。隐藏状态`h`将信息从一步传递到下一步。输入`x`和输出`y`在每一步发生。

这种展开视图使得梯度在反向传播 (backpropagation)（通过时间的反向传播，即BPTT）过程中如何流动更加清晰，以及为什么捕捉长程依赖有时会比较困难，这会导致像梯度消失问题，我们将在后面讨论。

### 输入和输出情景

RNN在处理不同的序列输入/输出关系方面很灵活：

- **多对一：** 序列输入，单个输出（例如，句子的情感分类）。
- **一对多：** 单个输入，序列输出（例如，图像描述生成——输入图像，输出词序列）。
- **多对多（对齐 (alignment)）：** 序列输入，序列输出且输入/输出长度匹配（例如，句子中每个词的词性标注）。
- **多对多（延迟）：** 序列输入，序列输出，其中输出在一些延迟或处理完整个输入后开始（例如，机器翻译）。

核心RNN思想是这些不同模式的依据。在Keras中，`SimpleRNN`、`LSTM`和`GRU`等层实现了这种循环行为。我们将在下一节中说明如何使用这些层，从基本的`SimpleRNN`开始。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  全面、数学化地介绍了循环神经网络，涵盖了其架构、训练以及本节讨论的理论基础。
- [Recurrent layers](https://keras.io/api/layers/recurrent_layers/) — Keras team (2024)
  Keras官方文档，用于在Keras中实现循环神经网络（RNN）、LSTM和GRU，提供了与课程相关的实用代码示例和API细节。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter, Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆（LSTM）网络的开创性论文，该网络解决了传统RNN中的梯度消失问题，是序列建模领域的一项基础性进展。
