---
course: "calculus-essentials-machine-learning"
chapter: "single-variable-calculus-ml"
lesson: "limits-foundation"
sourceId: 1351
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-2-single-variable-calculus-ml/limits-foundation"
title: "理解极限：根本原理"
description: "对极限理念的简要回顾，作为定义导数的根基。"
order: 1
plots: ["plots/1351-0.json"]
sourceHash: "29e46e52c0dc39b84457b403bf0dbf31ca1547d1327aacdcf5467d7cf951e139"
sourceCorrections: []
---

在讨论导数所表示的*瞬时*变化率之前，我们需要一种方法来描述函数值在其输入*非常*接近某个点（不一定达到该点）时的表现。这就是极限的理念。可以把它想成是无限放大图上的一个点，观察函数趋向何处。

设有一个函数 $f(x)$。我们想了解当 $x$ 趋近某个值 $a$ 时，$f(x)$ 的行为。我们关注的是当 $x$ 从两侧（小于 $a$ 的值和大于 $a$ 的值）越来越接近 $a$ 时，$f(x)$ 越来越接近的值，但 $x$ *不*一定等于 $a$ 本身。

## 形式化这个思想：极限

数学上，我们称 $f(x)$ 当 $x$ 趋近 $a$ 时的极限为 $L$，记作：

$\lim_{x \to a} f(x) = L$

这个表述意味着，我们可以通过选取足够接近 $a$ 但不等于 $a$ 的 $x$，使 $f(x)$ 的值任意接近 $L$。

即使函数在 $x=a$ 处没有定义，极限也可能存在。这一点很重要。极限描述的是点*附近*的趋势，而不是点*本身*的值。

我们来看一个例子。考虑函数：

$f(x) = \frac{x^2 - 1}{x - 1}$

当 $x$ 趋近 1 时会发生什么？如果我们尝试代入 $x=1$，我们会得到 $\frac{1^2 - 1}{1 - 1} = \frac{0}{0}$，这是未定义的。然而，对于 $x \neq 1$，我们可以简化函数：

$f(x) = \frac{(x-1)(x+1)}{x-1} = x+1 \quad (\text{对于 } x \neq 1)$

现在，当 $x$ 非常接近 1 时（例如 0.9, 0.99, 0.999 或 1.1, 1.01, 1.001），$f(x)$ 的值非常接近 $1+1=2$。即使 $f(1)$ 未定义，极限仍然存在：

$\lim_{x \to 1} \frac{x^2 - 1}{x - 1} = 2$



![函数 f(x) = (x^2 - 1) / (x - 1) 的图示](plots/1351-0.json)



> 函数 $f(x) = (x^2 - 1) / (x - 1)$ 的行为与 $f(x) = x+1$ 完全一致，除了在 $x=1$ 处未定义（由空心圆表示）。当 $x$ 趋近 1 时，极限是 2，这表示函数在“空心”点附近趋近的 y 值。

## 单侧极限

有时，我们可能只关注当 $x$ 从某个特定方向趋近 $a$ 时发生的情况：

- **左极限：** $x$ 通过小于 $a$ 的值趋近 $a$。记作 $\lim_{x \to a^-} f(x)$。
- **右极限：** $x$ 通过大于 $a$ 的值趋近 $a$。记作 $\lim_{x \to a^+} f(x)$。

为了使总极限 $\lim_{x \to a} f(x)$ 存在并等于 $L$，左极限和右极限都必须存在且等于 $L$。

## 为什么极限对导数很重要

极限的理念是构建导数定义的根基。如章节引言中所述，导数衡量的是瞬时变化率，本质上是一个函数在单个点的斜率。

我们如何找到单个点的斜率？我们首先找到连接函数曲线上两点（割线）的直线的斜率。然后，我们使用极限的理念使其中一个点无限接近另一个点。当点之间的距离趋近于零时，这些割线斜率的极限，就是该单个点处切线的斜率。这个取极限的过程正是导数的定义所在，我们将在下一节正式定义它。理解极限使我们能够将“区间上的平均变化率”转换为“点上的瞬时变化率。

## 参考资料

- [Calculus](https://www.cengage.com/c/calculus-8e-stewart/9781285740621/) — James Stewart (2015)
  Publisher: Cengage Learning
  一本基础教材，全面介绍极限，包括其定义、示例和性质。
- [Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — Prof. David Jerison (2010)
  Publisher: MIT OpenCourseWare
  麻省理工学院提供的一门优秀的免费在线课程，通过视频讲座和笔记涵盖了单变量微积分的基础概念，包括极限。
