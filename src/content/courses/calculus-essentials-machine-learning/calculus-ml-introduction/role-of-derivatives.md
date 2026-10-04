---
course: "calculus-essentials-machine-learning"
chapter: "calculus-ml-introduction"
lesson: "role-of-derivatives"
sourceId: 1348
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-1-calculus-ml-introduction/role-of-derivatives"
title: "衡量变化：导数的作用"
description: "导数如何衡量变化并指导机器学习中优化过程的概览。"
order: 3
plots: ["plots/1348-0.json"]
sourceHash: "297e4a1da2f5997e4b858646a90051200635334b3418db53d0740e0c1d5c9ecc"
sourceCorrections: []
---

我们已经知道机器学习 (machine learning)模型常以函数形式表示，我们的目标通常是优化它们，找出能产生最佳结果（例如，最低误差）的输入（参数 (parameter)）。但我们如何系统地找出这些“最佳”参数呢？尤其当模型有数百万个参数时，随意尝试数值的效率很低。我们需要一个方法来知道应该*朝哪个方向*调整参数。

想象一下你正站在浓雾笼罩的山坡上，想要到达最低点。你无法看清整个区域，但能感受到你所处位置的地面坡度。如果地面向左下方倾斜，你就会向左迈一步。如果地面向前方倾斜，你就会向前迈一步。这种“坡度”告诉你最陡峭的下降方向。

在数学中，衡量函数在特定点的瞬时斜率或变化率的工具就是**导数**。

### 导数：衡量瞬时变化

设想一个函数 $f(x)$，它表示模型基于单个参数 (parameter) $x$ 的成本（模型表现有多差）。我们想要找出使 $f(x)$ 最小的 $x$ 值。 $f(x)$ 相对于 $x$ 的导数，通常写作 $f'(x)$ 或 $\frac{df}{dx}$，告诉我们当输入 $x$ 有很小的变化时，输出 $f(x)$ 会改变多少。

从几何角度看，导数 $f'(a)$ 给出的是函数 $y = f(x)$ 在 $x = a$ 处的切线斜率。

- **正导数 ($f'(a) > 0$)**: 函数在 $x=a$ 处递增。如果我们稍稍增加 $x$， $f(x)$ 将会增加。切线向上倾斜。
- **负导数 ($f'(a) < 0$)**: 函数在 $x=a$ 处递减。如果我们稍稍增加 $x$， $f(x)$ 将会减少。切线向下倾斜。
- **零导数 ($f'(a) = 0$)**: 函数在 $x=a$ 处暂时平坦。这可能是一个最小值点、最大值点或鞍点（一个既非最小值也非最大值的平坦点）。切线是水平的。

考虑函数 $f(x) = x^2 - 2x + 3$。让我们在不同点可视化它的斜率：



![函数与切线（斜率）](plots/1348-0.json)



> 图表显示了函数 $f(x) = x^2 - 2x + 3$。在 $x=-1$ 处，切线（红色虚线）向下倾斜（负导数），表明函数正在递减。在 $x=1$ 处，切线（黄色虚线）是水平的（零导数），表明可能存在一个最小值。在 $x=3$ 处，切线（绿色虚线）向上倾斜（正导数），表明函数正在递增。

### 导数为何对优化很重要

这种思路可以直接应用于机器学习 (machine learning)模型的优化。“成本函数”是我们希望最小化的函数。模型的参数 (parameter)是此函数的输入。

如果我们计算成本函数相对于某个参数的导数：

1. **负导数**告诉我们增加参数会*降低*成本。因此，为了最小化成本，我们应该*增加*参数。
2. **正导数**告诉我们增加参数会*提高*成本。为了最小化成本，我们应该*减少*参数。
3. **零导数**表明我们可能相对于该参数处于一个最小值点（或最大值点/鞍点）。

通过计算所有参数的导数，我们知道成本函数的“最陡峭上升方向”。为了最小化成本，我们只需向*相反*方向迈一小步。这是梯度下降 (gradient descent)背后的主要思想，梯度下降是机器学习中最常用的优化算法之一，我们后续会详细讨论。

目前，主要要点是导数提供了一种量化 (quantization)衡量方式，说明函数输出如何随输入变化。这种衡量方式对处理模型成本函数的复杂细节并逐步调整参数以找出最佳值非常重要。在后续章节中，我们将规范导数的定义，学习如何计算各种函数的导数，并将这些思想扩展到具有许多输入（参数）的函数，这在机器学习中是常见情况。

## 参考资料

- [Mathematics for Machine Learning](https://mml-book.com) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679930](https://doi.org/10.1017/9781108679930)
  这本书将导数和优化等基本数学概念直接与它们在机器学习算法中的应用联系起来。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 4
  该书关于数值计算的章节介绍了梯度下降等优化算法，并解释了导数在训练机器学习模型中的应用。
- [18.01SC Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — Massachusetts Institute of Technology (MIT OpenCourseWare) (2010)
  Publisher: MIT OpenCourseWare
  提供了全面的单变量微积分视频讲座和材料，为导数及其性质的学习奠定了坚实的基础。
