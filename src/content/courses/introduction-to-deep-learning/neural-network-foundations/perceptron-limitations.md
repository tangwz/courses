---
course: "introduction-to-deep-learning"
chapter: "neural-network-foundations"
lesson: "perceptron-limitations"
sourceId: 5022
sourceUrl: "https://apxml.com/zh/courses/introduction-to-deep-learning/chapter-1-neural-network-foundations/perceptron-limitations"
title: "单层感知器的局限性"
description: "了解为什么单层感知器无法解决像异或这样的非线性可分问题。"
order: 5
plots: ["plots/5022-0.json"]
sourceHash: "149deb9afb68d50fb288c15fc9f28068ed466c6c6fe2fde3b66ba10fd129cf13"
sourceCorrections: []
---

正如我们所见，感知器是一种简单而基本的**人工神经元模型**，能够学习将输入分为两类。它通过找到一个线性决策边界来实现这一点——本质上是一条线（在二维空间中）、一个平面（在三维空间中）或一个超平面（在更高维度空间中）——用于分离属于不同类别的数据点。

感知机对于线性可分的问题非常有效。如果一个数据集你可以画一条直线（或平面/超平面）来分割空间，使得一类的所有点位于边界的一侧，而另一类的所有点位于边界的另一侧，那么该数据集就是线性可分的。典型例子包括逻辑与和或函数。例如，你可以很容易地画一条线来分离与操作中结果为`True`的输入和结果为`False`的输入。

然而，单层感知器的能力到此为止。当数据不是线性可分时会发生什么？

### 异或问题：一个经典局限

说明感知器局限性最著名的例子是\*\*异或（XOR）\*\*问题。异或是一种逻辑运算，当输入不同时输出`True`（或1），当输入相同时输出`False`（或0）。

让我们看一下两个输入（$x_1, x_2$）的异或真值表：

| $x_1$ | $x_2$ | Output (y) |
| --- | --- | --- |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

现在，让我们在二维图上可视化这四个输入点，其中颜色或形状代表输出类别（0或1）：



![异或问题数据点](plots/5022-0.json)



> 在二维空间中绘制的异或输入点。红色点代表类别0，蓝色点代表类别1。

仔细观察该图。你能画一条*单一的直线*来分离红色点（输出0）和蓝色点（输出1）吗？无论你尝试在哪里画线，你总会发现至少有一个点位于错误的一侧。

### 感知器在异或问题上失败的原因

感知器通过一个阶跃函数处理输入的加权和来确定其输出：
$y = \text{step}(\sum_{i} w_i x_i + b)$
决策边界由阶跃函数自变量为零的点定义：
$\sum_{i} w_i x_i + b = 0$
对于我们的双输入异或问题，这个方程变为：
$w_1 x_1 + w_2 x_2 + b = 0$
这是在($x_1, x_2$)二维平面中的一条直线方程。正如我们直观看到的那样，没有一条单一的直线能够根据异或函数的要求正确地划分空间。

让我们尝试推理 (inference)一下：

1. 点(0, 0)应该产生输出0，这意味着 $w_1(0) + w_2(0) + b = b$ 应该小于阈值（通常为 $\le 0$）。因此，$b \le 0$。
2. 点(1, 1)也应该产生输出0，这意味着 $w_1(1) + w_2(1) + b = w_1 + w_2 + b$ 也应该 $\le 0$。
3. 点(0, 1)应该产生输出1，这意味着 $w_1(0) + w_2(1) + b = w_2 + b$ 必须大于阈值（或 $> 0$）。
4. 点(1, 0)应该产生输出1，这意味着 $w_1(1) + w_2(0) + b = w_1 + b$ 也必须 $> 0$。

从（3）和（4）中，我们知道 $w_1 + b > 0$ 和 $w_2 + b > 0$。如果我们把这两个不等式相加，我们得到 $(w_1 + b) + (w_2 + b) > 0$，这简化为 $w_1 + w_2 + 2b > 0$。

现在将其与条件（2）：$w_1 + w_2 + b \le 0$ 和条件（1）：$b \le 0$ 进行比较。
如果 $b \le 0$，那么 $2b \le b$。因此，$w_1 + w_2 + 2b \le w_1 + w_2 + b$。
但我们发现 $w_1 + w_2 + 2b$ 必须 $> 0$，而 $w_1 + w_2 + b$ 必须 $\le 0$。这产生了矛盾。使用单一线性决策边界，不可能找到同时满足异或函数所有条件的权重 (weight) $w_1, w_2$ 和偏置 (bias) $b$。

### 突破线性边界

单层感知器无法处理像异或这样的非线性可分问题是一个重要的发现。这表明，要处理数据中更复杂的模式和关系，我们需要更精巧的网络架构。这一根本局限性直接推动了\*\*多层感知器 (MLP)\*\*的发展，它在输入和输出之间引入了隐藏层。正如我们将在下一节中看到的那样，这些隐藏层允许网络学习复杂的非线性决策边界，从而克服了简单感知器的局限性。

## 参考资料

- [Perceptrons: An Introduction to Computational Geometry](https://mitpress.mit.edu/9780262631112/perceptrons/) — Marvin Minsky and Seymour Papert (1987)
  Publisher: MIT Press
  这本经典著作批判性地分析了单层感知器的局限性，特别是证明了它们无法解决XOR问题，这极大地影响了早期人工智能研究的方向。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 6
  一本内容全面的教材，现代视角地阐述了深度学习，清晰解释了感知器、其固有限制以及为克服这些问题而发展多层架构的动因。
- [CS229 Lecture Notes: Neural Networks](http://cs229.stanford.edu/notes/cs229-notes-deep_learning.pdf) — Tengyu Ma, Anand Avati, Kian Katanforoosh, Andrew Ng (2019)
  Publisher: Stanford University; Pages: Lecture 6
  来自一门备受推崇的机器学习课程的讲义，提供了对感知器模型、其线性可分性以及其无法解决XOR等问题的易懂学术解释。
