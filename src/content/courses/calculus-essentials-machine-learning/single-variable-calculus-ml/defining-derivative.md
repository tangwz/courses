---
course: "calculus-essentials-machine-learning"
chapter: "single-variable-calculus-ml"
lesson: "defining-derivative"
sourceId: 1354
sourceUrl: "https://apxml.com/zh/courses/calculus-essentials-machine-learning/chapter-2-single-variable-calculus-ml/defining-derivative"
title: "导数的定义"
description: "导数的正式定义，即瞬时变化率或切线斜率。"
order: 2
plots: ["plots/1354-0.json"]
sourceHash: "a13da46e2b00a0f1bf1e0e25d851a889132694cea1326bef110f5b9756d602da"
sourceCorrections: []
---

在上一节中，我们回顾了极限，它为精确地定义瞬时变化率提供了数学工具。现在，我们来使用极限正式定义导数。

设想有一个函数，比如$f(x)$，它代表了你想要理解或优化的某种事物，比如机器学习 (machine learning)模型基于单个参数 (parameter)$x$的成本。你关心的是当输入$x$发生微小变动时，输出$f(x)$如何变化。

### 从平均变化率到瞬时变化率

考虑函数$y = f(x)$图上的两个点：一个在$x$处，另一个在附近的$x+h$处。对应的y值分别为$f(x)$和$f(x+h)$。x的变化量是$(x+h) - x = h$，y的变化量是$f(x+h) - f(x)$。

这两个点之间的*平均变化率*是连接它们的直线的斜率（称为割线）：

$\text{平均变化率} = \frac{\text{y的变化量}}{\text{x的变化量}} = \frac{f(x+h) - f(x)}{h}$

这表明在从$x$到$x+h$的区间内，$f$平均每个单位$x$变化量对应改变了多少。

那么，如果我们想知道在$x$点*精确地*变化率是多少呢？我们可以通过让区间$h$变得越来越小来找到这个*瞬时变化率*，这样第二个点就越来越接近第一个点。这就是极限的作用。我们取平均变化率当$h$趋近于零时的极限。

### 导数的定义

函数$f(x)$关于$x$的**导数**，记作$f'(x)$（读作“f撇x”）或$\frac{dy}{dx}$（莱布尼茨记号，读作“dee y dee x”），定义为函数在$x$点的瞬时变化率。形式上，它是这个极限：

$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$

如果此极限存在，我们就说函数$f$在$x$处是*可微的*。$f'(a)$的值给出了函数$f(x)$在特定点$x=a$处变化的精确速率。

### 几何意义：切线的斜率

从几何上看，导数$f'(a)$表示函数$y=f(x)$在点$(a, f(a))$处*切线*的斜率。切线是恰好“接触”曲线在该点，并且在该点与曲线方向相同的直线。

当$h \to 0$时，连接$(x, f(x))$和$(x+h, f(x+h))$的割线会旋转并接近在$(x, f(x))$处的切线。这条极限直线的斜率就是导数。



![割线趋近于切线](plots/1354-0.json)



> 随着第二个点（由$h$控制）越来越接近曲线$f(x)=x^2$上的第一个点$(1, 1)$，割线的斜率趋近于在$x=1$处切线的斜率。切线的斜率就是导数$f'(1)$。对于$f(x)=x^2$，有$f'(x)=2x$，所以$f'(1)=2$。

### 为什么这个定义对机器学习 (machine learning)有益处？

在机器学习中，我们经常使用成本函数（或损失函数 (loss function)）来衡量模型表现得有多差。假设$J(w)$是成本函数，$w$表示模型中的一个参数 (parameter)（比如权重 (weight)）。我们希望调整$w$来最小化$J(w)$。

导数$\frac{dJ}{dw}$告诉我们成本$J$对参数$w$的微小变化有多敏感。

- 如果$\frac{dJ}{dw}$是正数，稍微增加$w$会使成本增加。要降低成本，我们应该减小$w$。
- 如果$\frac{dJ}{dw}$是负数，稍微增加$w$会使成本降低。要降低成本，我们应该增加$w$。
- $\frac{dJ}{dw}$的*大小*告诉我们成本函数在该点有多*陡峭*。更大的大小表明$w$微小变化会带来更大的成本变化。

这种关于变化方向和大小的信息，正是梯度下降 (gradient descent)等优化算法迭代更新模型参数并找到最小成本所使用的。因此，理解导数对于理解机器学习模型如何训练是十分必要的。

在接下来的部分中，我们将学习计算导数的实用规则，而无需每次都回到极限定义。我们还将研究如何使用导数来寻找函数的最小值点和最大值点。

## 参考资料

- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: Chapter 4
  这本深度学习基础教材介绍了数学工具，包括导数和基于梯度的优化，这些工具对于理解和开发机器学习模型至关重要。
- [Single Variable Calculus (18.01SC)](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — Prof. David Jerison (2010)
  Publisher: MIT OpenCourseWare
  一个优秀的在线课程，提供了结构化的大学水平单变量微积分教学，涵盖了导数的定义、其极限定义和几何解释。
