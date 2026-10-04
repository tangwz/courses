---
course: "calculus-fundamentals-machine-learning"
chapter: "why-calculus-for-ml"
lesson: "limits-intuition"
sourceId: 2243
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-1-why-calculus-for-ml/limits-intuition"
title: "对极限的直观理解"
description: "通过简单的例子建立对极限的直观理解。"
order: 5
plots: ["plots/2243-0.json"]
sourceHash: "d728cf49a6d90ffc16e2f31154775c5a44a3970a412dc233115622a40b44cf4f"
sourceCorrections: []
---

我们已经见过极限的形式化定义：思考当函数的输入越来越接近某个点时，函数值会趋近于哪个数值。但这在实践中到底是什么含义呢？让我们建立一些直观认识。

想象一下，您正在观察一个函数的图表，例如简单的线性函数 $f(x) = 2x + 1$。极限可以理解为在问：“如果我用手指沿着图表描绘，当输入值（比如 $x=3$）极其接近某个点时，我的手指在垂直轴（即 $f(x)$ 值）上看起来会指向何处？”

让我们以 $f(x) = 2x + 1$ 为例，当 $x$ 趋近于 3 时，进行尝试。我们不是在问 $f(3)$ 究竟是多少（尽管在这个简单情形下，它是 $2(3) + 1 = 7$）。相反，我们是在查看 $x=3$ 周围的“邻域”。

考虑略微 *小于* 3 的 $x$ 值：

- 如果 $x = 2.9$，那么 $f(x) = 2(2.9) + 1 = 5.8 + 1 = 6.8$
- 如果 $x = 2.99$，那么 $f(x) = 2(2.99) + 1 = 5.98 + 1 = 6.98$
- 如果 $x = 2.999$，那么 $f(x) = 2(2.999) + 1 = 5.998 + 1 = 6.998$

现在，考虑略微 *大于* 3 的 $x$ 值：

- 如果 $x = 3.1$，那么 $f(x) = 2(3.1) + 1 = 6.2 + 1 = 7.2$
- 如果 $x = 3.01$，那么 $f(x) = 2(3.01) + 1 = 6.02 + 1 = 7.02$
- 如果 $x = 3.001$，那么 $f(x) = 2(3.001) + 1 = 6.002 + 1 = 7.002$

注意到规律了吗？当 $x$ 从任一侧越来越接近 3 时，输出值 $f(x)$ 越来越接近 7。这个值 7 就是函数 $f(x) = 2x + 1$ 当 $x$ 趋近于 3 时的极限。我们将其数学表示为：


$$
\lim_{x \to 3} (2x + 1) = 7
$$


下图有助于直观地显示这一点。当我们在直线上选取越来越接近 $x=3$ 的垂直线的点时，函数上对应的点会越来越接近高度 $y=7$。



![逼近极限：x=3 附近的 f(x) = 2x + 1](plots/2243-0.json)



> 当输入值 `x` 从左侧（如 2.9, 2.99）和右侧（如 3.1, 3.01）都越来越接近 3 时，输出值 `f(x)` 越来越接近 7。极限描述的就是这个目标值 7。

这个定义中重要的一点是“逼近一个值，*但不一定达到它*”。对于许多简单的函数，例如 $f(x) = 2x + 1$，当 $x$ 趋近于点 $a$ 时，极限就是函数在该点的值 $f(a)$。然而，极限的这个想法更具普适性。它使我们能够分析函数行为，即使在函数可能未定义的点上（比如图表上有一个“洞”，尽管我们在这里不会特别关注这些复杂情况）。极限关注的是当输入任意接近某个特定值时，函数输出的 *趋势*。

为什么要花时间在“逼近”这个想法上呢？因为它构成理解导数的根本，导数衡量的是 *瞬时* 变化率。我们需要无限接近一个点的想法，才能讨论 *在该点* 的斜率或变化。我们会在下一章清楚地看到这种联系。

## 参考资料

- [18.01SC Single Variable Calculus (Fall 2010)](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — Prof. David Jerison (2010)
  Publisher: MIT OpenCourseWare
  提供视频讲座和配套笔记，清晰地解释了极限的直观理解，使复杂概念易于掌握。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press
  包含微积分的基础回顾，包括极限的直观基础，对机器学习从业者有帮助。
