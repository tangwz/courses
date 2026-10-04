---
course: "introduction-to-neural-networks"
chapter: "forward-propagation"
lesson: "information-flow"
sourceId: 1946
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-3-forward-propagation/information-flow"
title: "网络中的信息流动"
description: "追踪数据流经前馈网络各层的路径。"
order: 1
plots: []
sourceHash: "427d337e1eee58641f2ead78a89dd0a45343e98babc81808edb09b50ce3ead2c"
sourceCorrections: []
---

把神经网络 (neural network)想象成一系列按层级组织的计算站点。前向传播过程就是你的输入数据通过这些站点所经历的路径，它从输入层开始，经过所有中间层（隐层），最终抵达输出层以产生一个结果。这是一个单向通道：信息严格地向前流动，从输入到输出，不会回溯（至少在我们当前讨论的前馈网络中是这样）。

### 层层计算

基本原则是一个层的输出会作为下一个层的输入。让我们追踪这条路径：

1. **输入层：** 这实际上不是一个计算层。它只是接收你的初始数据，这些数据通常表示为一个向量 (vector) $x$。向量 $x$ 也可以被认为是输入层的“激活值”，有时记作 $a^{[0]}$。
2. **第一个隐层：** 这个层（我们称之为第1层）的每个神经元都接收来自输入层的激活值 ($a^{[0]}$)。在第1层的每个神经元 $j$ 中，发生两件事：
   - 计算输入的加权和加上一个偏置 (bias)：$z_j^{[1]} = \sum_i w_{ji}^{[1]} a_i^{[0]} + b_j^{[1]}$。
   - 对这个和应用一个激活函数 (activation function) $g$：$a_j^{[1]} = g(z_j^{[1]})$。
     这个层中所有激活值 $a_j^{[1]}$ 的集合构成了激活向量 $a^{[1]}$，它成为下一个层的输入。
3. **后续隐层：** 对于任何额外的隐层（第2层、第3层等），这个过程都会重复。对于任意层 $l$，每个神经元都基于*前一个*层（即 $l-1$ 层）的激活值 $a^{[l-1]}$ 计算其加权和 $z_j^{[l]}$，然后应用激活函数 $g$ 来得到它自己的激活值 $a_j^{[l]}$。向量 $a^{[l]}$ 随后向前传递。
   $z_j^{[l]} = \sum_i w_{ji}^{[l]} a_i^{[l-1]} + b_j^{[l]}$
   $a_j^{[l]} = g(z_j^{[l]})$
4. **输出层：** 最终层执行相同的计算（加权和与激活），但其输出 $a^{[L]}$（其中 $L$ 是最后一层）代表了网络的最终预测结果，通常记作 $\hat{y}$。这个层中激活函数的选择通常取决于问题类型（例如，二分类问题用 Sigmoid，多分类问题用 Softmax，回归问题则不用激活函数）。

在神经网络 (neural network)中，前向传播是一种顺序处理，其中层 $l-1$ 的输出成为层 $l$ 的输入，并持续到最终输出的产生。

### 流动可视化

考虑一个包含输入层、一个隐层和一个输出层的简单网络。信息严格地从左向右流动。

> 一个简单的前馈网络呈现信息的流动。输入 $x_1, x_2$（激活值 $a^{[0]}$）由隐层处理以产生激活值 $a^{[1]}$，然后这些激活值再由输出层处理以产生最终预测 $\hat{y}$（激活值 $a^{[2]}$）。

图中的每根箭头都代表一个权重 (weight)，它连接着一个层中神经元的输出与下一个层中神经元的输入计算。整个过程包括计算从第1层到最终层 $L$ 的每一层 $l$ 的值 $z^{[l]}$ 和 $a^{[l]}$。结果 $a^{[L]}$ 就是网络针对给定输入 $x$ 所生成的预测 $\hat{y}$。

接下来的章节将详细说明每一层中线性变换 ($z$) 和激活函数 (activation function)应用 ($a$) 所涉及的具体计算，并逐步介绍如何使用矩阵运算来提高效率。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapters 6
  一本基础教材，涵盖神经网络的数学和概念基础，包括前馈架构中前向传播的详细解释。
- [Neural Networks and Deep Learning (Course 1 of the Deep Learning Specialization)](https://www.coursera.org/learn/neural-networks-deep-learning) — Andrew Ng, Kian Katanforoosh, Younes Bensouda Mourri (2017)
  Publisher: DeepLearning.AI
  一门入门级在线课程，对神经网络架构和前向传播过程提供了清晰的解释和实践见解。
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael A. Nielsen (2015)
  Publisher: Determination Press; Pages: Chapter 1
  一本免费在线书籍，为神经网络提供了温和而数学上严谨的介绍，其中包含前向传播的清晰步骤。
