---
course: "introduction-to-neural-networks"
chapter: "forward-propagation"
lesson: "final-output-prediction"
sourceId: 1957
sourceUrl: "https://apxml.com/zh/courses/introduction-to-neural-networks/chapter-3-forward-propagation/final-output-prediction"
title: "计算最终输出预测"
description: "从输出层获得网络的最终预测。"
order: 5
plots: []
sourceHash: "10287da920240b2895621c7c002e616618d318513fd848ae3bb3d0ba23f5c101"
sourceCorrections: []
---

在输出层，神经网络 (neural network)生成其最终预测结果 $\hat{y}$。这是输入数据处理的最后一个阶段，涉及线性变换和非线性激活。输出层内的操作与隐藏层中的操作相似，但最终输出会根据要解决的具体问题进行调整。

让我们将来自最后一个隐藏层（或者，如果没有隐藏层，则来自输入层）的激活表示为 $A^{[L-1]}$。为了得到输出预测 $A^{[L]}$，我们执行最后两步：

1. **计算输出层的加权和：** 就像在隐藏层中一样，我们使用输出层的权重 (weight) ($W^{[L]}$) 和偏置 (bias) ($b^{[L]}$) 计算此层输入 ($A^{[L-1]}$) 的线性组合。

   
   $$
   Z^{[L]} = W^{[L]} A^{[L-1]} + b^{[L]}
   $$
   

   这里，$Z^{[L]}$ 表示输出层神经元的预激活值。$W^{[L]}$、$b^{[L]}$ 以及因此 $Z^{[L]}$ 的维度取决于最后一个隐藏层中的神经元数量和输出层中的神经元数量。输出神经元的数量由任务决定：二分类或回归任务为一个神经元，具有 $C$ 个类别的多分类任务则为 $C$ 个神经元。
2. **应用输出层激活函数 (activation function)：** 输出层激活函数 $g^{[L]}$ 的选择十分重要，因为它将最终输出 $A^{[L]}$ 格式化为所需的预测形式。

   
   $$
   A^{[L]} = g^{[L]}(Z^{[L]})
   $$
   

   这个最终激活 $A^{[L]}$ 是网络的预测，我们通常将其表示为 $\hat{y}$。

   > 前向传播的最后一步是应用针对输出层的线性变换和激活函数，以生成网络的预测 $\hat{y}$。

### 选择正确的输出激活函数 (activation function)

与隐藏层通常选择 ReLU 不同，输出层的激活函数 ($g^{[L]}$) 直接取决于预测任务的性质：

- **回归：** 如果网络需要预测连续数值（例如预测房价），通常使用**线性激活**函数。这意味着输出就是简单的加权和：$A^{[L]} = Z^{[L]}$。输出范围可以从 $-\infty$ 到 $+\infty$。
- **二分类：** 对于输出是两个类别之一（例如，垃圾邮件或非垃圾邮件，0 或 1）的任务，使用 **Sigmoid** 函数。它将输出 $Z^{[L]}$ 压缩到 (0, 1) 范围，这可以解释为正类别的概率。
  
  $$
  \hat{y} = A^{[L]} = \sigma(Z^{[L]}) = \frac{1}{1 + e^{-Z^{[L]}}}
  $$
  
- **多分类：** 当在两个以上互斥类别之间分类时（例如，识别数字0-9），**Softmax** 函数是标准选择。它接受向量 (vector) $Z^{[L]}$（其中每个元素对应一个类别）并将其转换为概率分布。输出向量 $A^{[L]}$ 中的每个元素都在0到1之间，并且所有元素的总和为1。
  
  $$
  \hat{y}_i = A^{[L]}_i = \text{softmax}(Z^{[L]})_i = \frac{e^{Z^{[L]}_i}}{\sum_{j=1}^{C} e^{Z^{[L]}_j}} \quad \text{对于 } i=1, ..., C
  $$
  
  这里，$C$ 是类别数量，$\hat{y}_i$ 表示输入属于类别 $i$ 的预测概率。

### 理解预测 ($\hat{y}$)

最终输出向量 (vector) $A^{[L]}$，我们称之为 $\hat{y}$，包含网络基于输入 $X$ 的预测。

- 对于回归，$\hat{y}$ 是一个单个连续值（如果预测多个目标，则为多个值）。
- 对于二分类，$\hat{y}$ 是一个介于0和1之间的单个值，表示 $P(\text{类别}=1 | X)$。您可能会应用一个阈值（例如0.5）来将此概率转换为明确的类别标签（0或1）。
- 对于多分类，$\hat{y}$ 是一个概率向量，每个类别一个。概率最高的类别通常被选为最终预测标签。

随着 $\hat{y}$ 的计算完成，前向传播过程也已完成。网络接收输入 $X$，并通过其层传递信息，沿途应用线性变换和激活函数 (activation function)，从而生成预测 $\hat{y}$。现在可以使用损失函数 (loss function)将此预测与真实标签 $y$ 进行比较，这是训练过程的第一步（将在下一章中介绍）。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  涵盖神经网络的理论基础，包括前向传播、各种输出单元（线性、sigmoid、softmax）及其数学公式的详细解释。
- [Neural Networks and Deep Learning (Deep Learning Specialization, Course 1)](https://www.coursera.org/learn/neural-networks-deep-learning) — Andrew Ng, Kian Katanforoosh, Younes Bensouda Mourri (2017)
  Publisher: DeepLearning.AI
  提供神经网络基础知识的易懂介绍，清晰地解释了前向传播、激活函数以及不同输出层在实践中的作用。
