---
course: "calculus-fundamentals-machine-learning"
chapter: "derivatives-measuring-change"
lesson: "higher-order-derivatives"
sourceId: 2262
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-2-derivatives-measuring-change/higher-order-derivatives"
title: "高阶导数介绍"
description: "简要介绍二阶导数及更高阶导数的原理。"
order: 6
plots: ["plots/2262-0.json"]
sourceHash: "600eb9d20ace8b224616644c5805950b3173edc3ee59da1fd7db97a02f5ec99b"
sourceCorrections: []
---

导数 $f'(x)$ 表明了函数 $f(x)$ 的瞬时变化率。它衡量了函数输出相对于其输入变化的快慢，在几何上表示为切线的斜率。

但是，如果我们想知道*变化率本身*是如何变化的呢？正如我们可以求一个函数的变化率一样，我们也可以求其导数的变化率。这就引出了**高阶导数**的原理。

### 二阶导数

你最常用到的高阶导数，特别是在与机器学习 (machine learning)相关的优化问题中，就是**二阶导数**。

简单来说，二阶导数就是*一阶*导数的导数。如果你有一个函数 $f(x)$，你先求它的一阶导数 $f'(x)$。然后，你再次对 $f'(x)$ 求导，以获得二阶导数。

**记号：**

就像一阶导数有多种记号一样，二阶导数也有多种：

- **拉格朗日记号：** 如果一阶导数是 $f'(x)$，二阶导数表示为 $f''(x)$（读作“f double prime of x”）。
- **莱布尼茨记号：** 如果一阶导数是 $\frac{dy}{dx}$，二阶导数表示为 $\frac{d^2y}{dx^2}$（读作“d squared y by dx squared”）。请注意'2'的位置。它表示对 $x$ 连续应用两次导数运算符 $\frac{d}{dx}$。

**计算：**

计算二阶导数需要应用你已经学过的求导法则，只是再应用一次。

我们来看一个例子：
考虑函数：
$f(x) = x^3 + 2x^2 - 5x + 1$

首先，使用幂法则、常数倍法则和和法则求一阶导数：
$f'(x) = \frac{d}{dx}(x^3) + \frac{d}{dx}(2x^2) - \frac{d}{dx}(5x) + \frac{d}{dx}(1)$
$f'(x) = 3x^{3-1} + 2(2x^{2-1}) - 5(1x^{1-1}) + 0$
$f'(x) = 3x^2 + 4x - 5$

现在，为了求二阶导数 $f''(x)$，我们对 $f'(x)$ 关于 $x$ 求导：
$f''(x) = \frac{d}{dx}(f'(x)) = \frac{d}{dx}(3x^2 + 4x - 5)$
$f''(x) = \frac{d}{dx}(3x^2) + \frac{d}{dx}(4x) - \frac{d}{dx}(5)$
$f''(x) = 3(2x^{2-1}) + 4(1x^{1-1}) - 0$
$f''(x) = 6x + 4$

因此，对于函数 $f(x) = x^3 + 2x^2 - 5x + 1$，一阶导数是 $f'(x) = 3x^2 + 4x - 5$，二阶导数是 $f''(x) = 6x + 4$。

### 二阶导数有何作用？

一阶导数 $f'(x)$ 表示函数的斜率。二阶导数 $f''(x)$ 则表明了斜率是如何变化的。

- **如果在某一点 $x$ 处 $f''(x) > 0$**，表示斜率 $f'(x)$ 正在*增加*。函数的图形向上弯曲，就像一个盛水的杯子。这称为**向上凹**。
- **如果在某一点 $x$ 处 $f''(x) < 0$**，表示斜率 $f'(x)$ 正在*减小*。函数的图形向下弯曲，就像一个倒扣的杯子。这称为**向下凹**。
- **如果在某一点 $x$ 处 $f''(x) = 0$**，在该点处凹凸性可能发生变化（即拐点），或者仅凭二阶导数测试无法确定。

可以联想一下速度和加速度。如果 $f(x)$ 表示物体的位移，那么 $f'(x)$ 表示其速度（位移的变化率）。而 $f''(x)$ 则表示其加速度（速度的变化率）。正加速度意味着速度增加；负加速度（减速）意味着速度减小。



![凹凸性与二阶导数](plots/2262-0.json)



> 图中显示了 $y=x^2$，其总是向上凹 ($f''(x)=2$)，以及 $y=-x^2$，其总是向下凹 ($f''(x)=-2$)。

### 二阶导数之后

我们可以继续求导吗？是的。我们可以求二阶导数的导数，这称为**三阶导数**，记作 $f'''(x)$ 或 $\frac{d^3y}{dx^3}$。我们可以继续求四阶导数、五阶导数，以此类推。这些统称为高阶导数。

尽管这些导数都存在，但一阶导数和二阶导数是机器学习 (machine learning)基本原理中最常见且实际有用的导数，特别是对于优化问题。二阶导数，我们将在后面提到，在确认用一阶导数找到的点是否确实是最小值（比如成本函数中谷底）还是最大值方面有作用。

目前来看，重要的一点是我们可以多次对函数求导，而二阶导数提供了关于函数的曲率或凹凸性的信息。

## 参考资料

- [Single Variable Calculus](https://ocw.mit.edu/courses/18-01sc-single-variable-calculus-fall-2010/) — David Jerison, Arthur Mattuck, Haynes Miller, Benjamin Brubaker, Jeremy Orloff, Heidi Burgiel, Christine Breiner, David Jordan, Joel Lewis (2010)
  Publisher: MIT OpenCourseWare
  麻省理工学院提供的一门易于理解的在线课程，为单变量微积分奠定了坚实的基础，涵盖了导数、高阶导数及其几何和物理解释。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  这本权威教科书介绍了深度学习的数学基础，包括导数和二阶导数在机器学习相关优化算法中的应用。
