---
course: "calculus-fundamentals-machine-learning"
chapter: "derivatives-measuring-change"
lesson: "derivative-definition"
sourceId: 2253
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-2-derivatives-measuring-change/derivative-definition"
title: "导数：切线的斜率"
description: "从几何角度将导数定义为函数图像切线的斜率。"
order: 2
plots: ["plots/2253-0.json"]
sourceHash: "fd5f25f6510088cbd77fec4e9e0a7999af71ab276c60c2ea2ddb5d5f78aa3ebb"
sourceCorrections: []
---

在上一节中，我们讨论了函数如何变化，查看了平均变化率和瞬时变化率。在一个区间内计算的平均变化率提供了一个大致的了解，但我们通常需要知道函数在 *某一点* 到底变化多快。想想你的汽车里程表。它不会显示你过去一小时的平均速度；它显示的是你 *此刻* 的速度。我们要找的就是这种“此刻”的变化率，它引出了导数的想法。

如何形象地理解这种瞬时变化率？想象一下，在一个函数图像的某一点附近放大、再放大。如果函数是平滑的（没有尖角或断裂），曲线会变得越来越像一条直线。这条恰好触及我们感兴趣的单一点，并且在该点与曲线方向相同的直线，称为**切线**。

考虑**割线**，它是一条穿过曲线上 *两* 个点的直线。这条割线的斜率代表了这两个点之间的 *平均* 变化率。

现在，设想沿着曲线移动其中一个点，使其越来越接近另一个点。随着两点之间距离的缩小，割线会围绕着固定点旋转，越来越接近成为切线。割线的斜率会逼近切线的斜率。



![割线逼近切线](plots/2253-0.json)



> 函数 $f(x) = x^2$（蓝色）。随着割线上的第二个点（橙色、红色）向 $x=1$ 处的点靠近，割线的斜率会逼近 $x=1$ 处切线（绿色虚线）的斜率。

这引出了导数的几何定义：

**函数 $f(x)$ 在点 $x=a$ 处的导数，就是该点处函数图像的切线斜率。**

为什么斜率重要？记住，斜率衡量的是陡峭程度，即变化率。更陡的斜率意味着更快的变化。切线的斜率精确地告诉我们，函数输出 $y$ 相对于输入 $x$ *在那个精确瞬间* 的变化速度。

- 如果切线向上倾斜（正斜率），函数在该点增加。
- 如果切线向下倾斜（负斜率），函数在该点减少。
- 如果切线水平（零斜率），函数在该点暂时平坦；它在该点没有增加也没有减少。这通常发生在曲线的波峰和波谷。

数学上，我们使用极限来表达割线斜率逼近切线斜率的这个想法。如果割线上的两点是 $(x, f(x))$ 和 $(x+h, f(x+h))$，那么割线的斜率是：

$\text{割线斜率} = \frac{\Delta y}{\Delta x} = \frac{f(x+h) - f(x)}{(x+h) - x} = \frac{f(x+h) - f(x)}{h}$

为了求得切线的斜率，我们取这个表达式在两点距离 $h$ 趋近于零时的极限：

$\text{切线斜率} = f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$

这个极限，如果存在的话，*就是* 函数 $f$ 在点 $x$ 处的导数。现在不必过于担心直接计算极限。重要的收获是这种形象的联系：导数衡量的是瞬时变化率，在几何上，这对应于函数图像在该特定点切线的斜率。这种关联对于理解我们如何在机器学习 (machine learning)中利用导数来求得最佳解是很重要的。

## 参考资料

- [Calculus: Early Transcendentals](https://www.cengage.com/c/calculus-early-transcendentals-9e-stewart/9781337613927/) — James Stewart, Daniel K. Clegg, Saleem Watson (2020)
  Publisher: Cengage Learning
  一本广泛使用的微积分入门教材，对导数、切线和极限从基本原理进行了全面解释。
- [Thomas' Calculus](https://www.pearson.com/en-us/subject-catalog/p/thomas-calculus/P200000003507/9780134438986) — Joel R. Hass, Christopher E. Heil, Maurice D. Weir (2017)
  Publisher: Pearson
  一本经典的微积分教材，以清晰地讲解微分学而闻名，包括导数的几何解释。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; Pages: Chapter 5; DOI: [10.1017/9781108679902](https://doi.org/10.1017/9781108679902)
  一本将机器学习建立在数学基础之上的书，其中有一个专门关于微分学的章节，介绍了导数及其作用。
