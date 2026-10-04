---
course: "introduction-to-machine-learning"
chapter: "supervised-learning-regression"
lesson: "gradient-descent-finding-best-fit"
sourceId: 1484
sourceUrl: "https://apxml.com/zh/courses/introduction-to-machine-learning/chapter-3-supervised-learning-regression/gradient-descent-finding-best-fit"
title: "梯度下降：寻找最佳拟合"
description: "介绍梯度下降算法，用于最小化成本函数和寻找模型参数。"
order: 5
plots: ["plots/1484-0.json"]
sourceHash: "8275f52467fa79b82a087505f77ffbe116b558a83a89ad7d1382caae720046fb"
sourceCorrections: []
---

成本函数，例如均方误差 (MSE)，用于衡量直线与数据的拟合程度。成本越低，拟合效果越好。目标是找到斜率 ($m$) 和 y 轴截距 ($b$) 的具体值，使得成本*尽可能低*。**梯度下降 (gradient descent)**算法正是用于系统地找出这些最佳值的方法。

可以把成本函数想象成一个形状，也许是一个山谷或一个碗。成本函数的值（我们的误差）对应于这个形状中任意给定点的海拔高度。这个形状中的坐标是我们参数 (parameter) $m$ 和 $b$ 的值。我们的目标是找到这个山谷的最低点，即成本最小的点。

梯度下降是一个迭代算法，它帮助我们在这个成本函数上“下坡行走”，直到我们到达底部，或者至少一个很低的点。

### 类比：蒙着眼睛下山

想象一下你正站在一个雾蒙蒙的山坡上，你的目标是到达山谷底部。你无法看到整个区域，但你可以感觉到你所站位置的坡度陡峭程度和方向。

1. **感知坡度：** 你检查脚下的地面，以确定从当前位置开始哪个方向是下坡最陡峭的路径。这种“陡峭程度和方向”与成本函数的**梯度**类似。梯度指向成本*增加最快*的方向。
2. **迈出一步：** 既然你想下坡（降低成本），你就朝着最陡峭上升方向的精确反方向迈出一小步。
3. **重复：** 从你的新位置，你再次感知坡度，确定下坡最陡峭的方向，然后迈出又一小步。

你不断重复这个过程。每一步都会把你带到山坡上一个稍微低一点的位置。最终，如果你持续小步下坡，你就会到达山谷底部，即最低海拔点（最小成本）。

### 梯度下降 (gradient descent)的数学工作方式

在线性回归的背景下，梯度下降通过以下方式工作：

1. **初始化：** 从 $m$ 和 $b$ 的一些初始猜测值开始。这些值可以是任何数，通常是 0。
2. **计算梯度：** 在 $m$ 和 $b$ 的当前值处计算成本函数 $J(m, b)$ 的梯度。梯度是一对值，它们告诉我们如果轻微改变 $m$ 或 $b$，成本函数会改变多少。具体来说，它告诉我们成本*增加最快*的方向。

   - $\frac{\partial J}{\partial m}$: 成本如何随斜率 $m$ 的微小变化而变化。
   - $\frac{\partial J}{\partial b}$: 成本如何随截距 $b$ 的微小变化而变化。
3. **更新参数 (parameter)：** 通过沿着梯度的*相反*方向移动少量来调整 $m$ 和 $b$。这意味着从当前参数值中减去梯度的一部分。这个“步长”的大小由一个称为**学习率**的参数控制（通常用希腊字母 alpha $\alpha$ 表示）。

   更新规则如下所示：
   $m := m - \alpha \frac{\partial J}{\partial m}$
   $b := b - \alpha \frac{\partial J}{\partial b}$
   `:=` 符号表示“更新为”。我们根据使用 $m$ 和 $b$ 的*当前*值计算出的梯度，同时更新 $m$ 和 $b$。
4. **迭代：** 重复步骤 2 和 3 很多次。每次迭代， $m$ 和 $b$ 的值都应该更接近使成本函数 $J(m, b)$ 最小化的值，并且成本本身也应该降低。

### 学习率 ($\alpha$)

学习率 $\alpha$ 是一个小的正数（例如 0.01、0.001），它控制着我们在每次迭代中下坡的步长大小。它是一个重要的参数 (parameter)：

- **如果 $\alpha$ 太小：** 梯度下降 (gradient descent)会起作用，但可能需要很长时间才能达到最小值，因为步长太小了。想象一下你非常缓慢地挪动着下山。
- **如果 $\alpha$ 太大：** 我们可能会越过最小值。想象一下你下坡迈的步子太大，以至于直接越过了谷底，最终跑到了山的另一边更高的地方。更糟的情况下，算法甚至可能**发散**，这意味着成本随着每一步的进行而越来越高，永远无法找到最小值。

选择一个好的学习率通常需要一些尝试。我们希望它足够大，以便合理地快速收敛，但又足够小，以避免越界或发散。

### 梯度下降 (gradient descent)的可视化

让我们将成本函数 $J(m,b)$ 可视化为等高线图。每条等高线代表成本相同的点 $(m, b)$。等高线的中心代表最小成本。梯度下降从某个点 $(m_0, b_0)$ 开始，沿着垂直于等高线的方向迈步，向中心移动。



![成本函数等高线上的梯度下降路径](plots/1484-0.json)



> 一个等高线图，显示了基于斜率 ($m$) 和截距 ($b$) 不同值的成本函数。红线展示了梯度下降可能走的路径，它从一个初始猜测开始，并迭代地移向最小成本点（中心）。

### 我们何时停止？

当满足以下条件之一时，我们通常会停止迭代过程：

1. **收敛：** 成本函数在一次迭代中只减少了很小的量，这表明我们已经非常接近最小值了。
2. **最大迭代次数：** 我们让算法运行预定义的最大步数。

梯度下降 (gradient descent)是优化引擎，它使我们的线性回归模型能够从数据中“学习”。通过迭代地调整斜率 ($m$) 和截距 ($b$) 以最小化成本函数，它找到了一条根据 MSE 指标最适合我们训练数据的直线。

## 参考资料

- [Machine Learning (Course Materials)](https://www.coursera.org/specializations/machine-learning-introduction) — Andrew Ng (2022)
  Publisher: DeepLearning.AI / Stanford University
  提供了梯度下降的直观和基础性解释。
- [Deep Learning](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 271-272
  为梯度下降及其变体提供了全面的数学背景，特别是关于数值计算的第四章。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  包含了梯度下降在现代机器学习应用中的实际案例和清晰说明。
