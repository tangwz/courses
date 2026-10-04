---
course: "calculus-fundamentals-machine-learning"
chapter: "partial-derivatives-multiple-inputs"
lesson: "calculating-partial-derivatives"
sourceId: 2290
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-4-partial-derivatives-multiple-inputs/calculating-partial-derivatives"
title: "计算偏导数"
description: "学习使用常用规则计算偏导数的步骤。"
order: 3
plots: []
sourceHash: "a9978596fc1e9f9aee9c56b13dd447ce3377d3ba69b34f33168f549f23984407"
sourceCorrections: []
---

将其他变量视为常数是计算偏导数的主要思想。如果您对计算单变量函数（如 $f(x)$）的导数很熟悉，那么您已经具备了大部分所需的能力。

## 求导步骤：固定变量再求导

当您需要求一个函数对某个特定变量的偏导数时，请遵循以下步骤：

1. **确定目标变量：** 确定您要对其进行求导的变量。例如，如果您要计算 $\frac{\partial f}{\partial x}$，您的目标变量就是 $x$。
2. **将其他变量视为常数：** 在心里（或者实际地，如果这有帮助的话）将函数中所有其他变量替换为固定常数。把它们想象成 5、-2 或 $\pi$ 这样的数字。
3. **应用标准求导法则：** 现在，*仅*对您的目标变量，使用常用规则（如幂法则、常数法则、和法则）对函数进行求导。请记住，任何常数项的导数都为零。

让我们通过几个例子来使其更具体。

### 示例 1：一个简单多项式

考虑函数：
$f(x, y) = x^2 + y^3 + 4$

我们来找到它对 $x$ 和 $y$ 的偏导数。

**计算 $\frac{\partial f}{\partial x}$ (对 $x$ 的偏导数)：**

1. **目标变量：** $x$。
2. **将其他变量视为常数：** 将 $y$ 视为常数。这意味着 $y^3$ 也被视为常数。数字 $4$ 本身就是常数。
3. **求导：**

   - $x^2$ 对 $x$ 的导数是 $2x$。
   - $y^3$（被视为常数）对 $x$ 的导数是 $0$。
   - $4$（一个常数）对 $x$ 的导数是 $0$。

   使用和法则将其组合起来：
   $\frac{\partial f}{\partial x} = 2x + 0 + 0 = 2x$

**计算 $\frac{\partial f}{\partial y}$ (对 $y$ 的偏导数)：**

1. **目标变量：** $y$。
2. **将其他变量视为常数：** 将 $x$ 视为常数。这意味着 $x^2$ 也被视为常数。数字 $4$ 是一个常数。
3. **求导：**

   - $x^2$（被视为常数）对 $y$ 的导数是 $0$。
   - $y^3$ 对 $y$ 的导数是 $3y^2$。
   - $4$（一个常数）对 $y$ 的导数是 $0$。

   将其组合起来：
   $\frac{\partial f}{\partial y} = 0 + 3y^2 + 0 = 3y^2$

请注意，这个过程如何隔离开仅一个变量变化的影响。当我们计算 $\frac{\partial f}{\partial x}$ 时，$y^3$ 项消失了，因为从 $x$ 的角度来看，$y$ 没有变化。

### 示例 2：变量相乘的情况

我们来看一个在处理模型参数 (parameter)（如权重 (weight) ($w$) 和偏差 ($b$)）时常见的函数结构：

$g(w, b) = w^2 b + 5w - 2b + 7$

**计算 $\frac{\partial g}{\partial w}$ (对 $w$ 的偏导数)：**

1. **目标变量：** $w$。
2. **将其他变量视为常数：** 将 $b$ 视为常数。
3. **求导：**

   - 考虑项 $w^2 b$。由于 $b$ 被视为常数系数，$w^2 b$ 对 $w$ 的导数是 $(b) \times (2w) = 2wb$。（可以把它想象成求 $5x^2$ 的导数，结果是 $5 \times 2x = 10x$；这里 $b$ 扮演 $5$ 的角色。）
   - $5w$ 对 $w$ 的导数是 $5$。
   - $-2b$（被视为常数）对 $w$ 的导数是 $0$。
   - $7$（一个常数）对 $w$ 的导数是 $0$。

   将这些组合起来：
   $\frac{\partial g}{\partial w} = 2wb + 5 + 0 + 0 = 2wb + 5$

**计算 $\frac{\partial g}{\partial b}$ (对 $b$ 的偏导数)：**

1. **目标变量：** $b$。
2. **将其他变量视为常数：** 将 $w$ 视为常数。这意味着 $w^2$ 和 $5w$ 也被视为常数。
3. **求导：**

   - 考虑项 $w^2 b$。由于 $w^2$ 被视为常数系数，$w^2 b$ 对 $b$ 的导数是 $w^2 \times 1 = w^2$。（可以把它想象成求 $ax$ 对 $x$ 的导数，结果是 $a$；这里 $w^2$ 扮演 $a$ 的角色，$b$ 扮演 $x$ 的角色。）
   - $5w$（被视为常数）对 $b$ 的导数是 $0$。
   - $-2b$ 对 $b$ 的导数是 $-2$。
   - $7$（一个常数）对 $b$ 的导数是 $0$。

   将这些组合起来：
   $\frac{\partial g}{\partial b} = w^2 + 0 - 2 + 0 = w^2 - 2$

## 要点

计算偏导数沿用了您为单变量函数学习的求导规则。核心技巧是暂时“固定”除您正在求导的变量之外的所有变量，在计算过程中将它们视为常数。这使您能够确定当单个目标变量变化时，函数输出如何变化，同时保持其他所有不变。这个技巧对于理解如何使用梯度下降 (gradient descent)等方法调整机器学习 (machine learning)模型参数 (parameter)非常重要。

## 参考资料

- [Calculus: Early Transcendentals](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHeguSh9XN86oHYREs4nK0hgn7UY73C0ktPvGnlop4v-YXjTPdeMFCU5BdkYR9OSRDvFz9nN1mKmUNNMfgBVIFrpKbqgzSDcSu8UGDhUNprurtS-q0d313ERKZgA_LZXI1Q5kPTRWPzp_bRReop0UqO7GDJUzLBGWZ9Te5lgM86IgvbpndROlL5x96wfw==) — James Stewart (2015)
  Publisher: Cengage Learning; Pages: 1368
  一本内容全面、被广泛采用的教科书，清晰地介绍了多元微积分，包括偏导数的详细解释和示例。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  本书涵盖机器学习所需的数学基础，其中一章专门讨论多元微积分，直接在机器学习应用背景下解释偏导数。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  深度学习领域的标准教科书，频繁使用偏导数作为梯度下降和反向传播的基础，展示了它们在机器学习中的实际作用。
