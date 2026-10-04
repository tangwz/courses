---
course: "calculus-fundamentals-machine-learning"
chapter: "calculus-action-optimization"
lesson: "learning-rate"
sourceId: 2315
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization/learning-rate"
title: "学习率参数"
description: "介绍学习率及其在控制步长中的作用。"
order: 6
plots: ["plots/2315-0.json"]
sourceHash: "e9877cddbe5316f677693c3ad6e8bd59cc1450ae99b2409ebbe19e5e2c0d4b9c"
sourceCorrections: []
---

让我们更好地理解梯度下降 (gradient descent)的步骤。我们已经知道会计算成本函数的梯度，这会告诉我们成本函数最陡峭的*上升*方向。由于我们的目标是*最小化*成本，因此我们希望朝着梯度的*相反*方向移动。

在梯度下降的每一步中，确定向相反方向移动的距离至关重要。**学习率**（通常用希腊字母阿尔法，$\alpha$ 表示）在此发挥作用。它是一个小的正数，用于决定每次迭代中迈出步子的大小。

回想一下我们用于参数 (parameter)（例如线性回归例子中的 $m$ 或 $b$）的更新规则。一般形式如下：

`parameter = parameter - learning_rate * gradient_of_cost_wrt_parameter`

对于我们的具体参数 $m$ 和 $b$，使用偏导数表示法表示梯度分量，更新公式为：


$$
b = b - \alpha \frac{\partial J}{\partial b}
$$


$$
m = m - \alpha \frac{\partial J}{\partial m}
$$


这里，$J$ 表示成本函数，$\frac{\partial J}{\partial b}$ 是成本对 $b$ 的偏导数，$\frac{\partial J}{\partial m}$ 是成本对 $m$ 的偏导数，$\alpha$ 是我们的学习率。

### 为什么需要学习率？

梯度告诉我们移动的方向，但没有告诉距离。梯度向量 (vector)的大小（量级）表明了坡度有多陡。如果我们每一步都直接从参数 (parameter)中减去整个梯度，那么在坡度陡峭时，我们可能会迈出很大的步子；而在坡度平坦时，步子又会很小。迈出巨大的步子，尤其是在远离最小值时，可能会导致问题。

想象一下你在有雾的天气下山。你能感觉到脚下的坡度（梯度），它告诉你最陡峭的下坡路。学习率就像是决定你步子长度的依据。

### 学习率大小的影响

选择合适的学习率（$\alpha$）对有效优化很重要。让我们考虑不同选择会发生什么：

- **如果学习率太小：** 你会迈出小步子下坡。你最终会到达底部（最小成本），但这可能需要很长时间和很多次迭代。进展会很慢。
- **如果学习率太大：** 你会迈出大步子下坡。你可能会越过谷底，到达另一侧，甚至可能比你开始的位置更高！成本可能会不稳定地波动，无法下降，甚至可能随着时间增加（发散）。
- **如果学习率“刚刚好”：** 你会迈出合理大小的步子，稳步向最小值前进，而不会过度越过。这通常能高效地找到好的解决方案。

让我们将此可视化。想象一个仅依赖于一个参数 (parameter)的简单成本函数，绘制为参数值的函数。我们想要找到最低点。下面的图表模拟了在不同学习率下，成本如何随着迭代次数变化。



![学习率对成本降低的影响](plots/2315-0.json)



> 模拟成本函数 $J(w) = w^2$，从 $w=5$ 开始，使用梯度下降 (gradient descent) $w = w - \alpha (2w)$。良好的学习率（绿色）会使成本稳步下降。小的学习率（黄色）使成本下降非常缓慢。大的学习率（橙色）可能会波动而没有改善。更大的学习率（红色）会导致成本增加（发散）。

### 选择值

那么，如何选择学习率呢？找到最优学习率通常需要一些实验。常见的起始值可能是 0.1、0.01、0.001 或 0.0001。你可以尝试几个不同的值，看看哪个能使成本函数在初始训练迭代中稳步且合理地快速下降。也有更高级的技术可以在训练期间自动调整学习率，但目前，你需要明白它是一个通常在优化过程开始前设置的参数 (parameter)。

总之，学习率 $\alpha$ 是一个虽小但重要的数值，它控制着梯度下降 (gradient descent)在每次迭代中迈出的步子大小。它需要仔细选择：太小会导致收敛缓慢，而太大会导致不稳定和发散。选择正确有助于算法高效找到使成本函数最小化的参数值。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 8
  一本全面的教科书，涵盖了梯度下降和学习率作为优化算法中的核心超参数。第8章“用于训练深度模型的优化”尤为有用。
- [CS229 Lecture Notes - Supervised Learning, Part I](http://cs229.stanford.edu/notes2022/cs229-notes1.pdf) — Andrew Ng, Tengyu Ma (2022)
  Publisher: Stanford University; Pages: 8-9
  斯坦福大学机器学习课程官方讲义的一部分，提供了梯度下降和在线性回归背景下学习率设置的基础解释。
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf) — Diederik P. Kingma and Jimmy Ba (2014)
  Journal: International Conference on Learning Representations (ICLR 2015); Pages: 1-13; DOI: [10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)
  这篇基础论文介绍了 Adam 优化器，这是一种广泛采用的自适应学习率算法。它提供了关于如何动态调整学习率的见解，这是作为高级主题提及的一个概念。
