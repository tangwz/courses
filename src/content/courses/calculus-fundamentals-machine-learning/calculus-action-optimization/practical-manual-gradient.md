---
course: "calculus-fundamentals-machine-learning"
chapter: "calculus-action-optimization"
lesson: "practical-manual-gradient"
sourceId: 2320
sourceUrl: "https://apxml.com/zh/courses/calculus-fundamentals-machine-learning/chapter-5-calculus-action-optimization/practical-manual-gradient"
title: "实践操作：手动梯度计算"
description: "为一个非常简单的数据集和线性模型进行梯度的手动计算。"
order: 8
plots: ["plots/2320-0.json", "plots/2320-1.json"]
sourceHash: "b01f860652d058a14bc6dab5a4c9320b6dc18f9961d5fd3ef579574e54b87a93"
sourceCorrections: []
---

让我们将理论付诸实践。我们已经讨论了成本函数、梯度以及梯度下降 (gradient descent)的基本思想。现在，我们将使用一个小型数据集，手动计算一个简单线性回归模型的梯度。这项练习将巩固您对导数如何推动优化过程的理解。

### 场景设定：一个简单问题

假设我们有一个仅包含三个点 $(x, y)$ 的小型数据集：$(1, 2)$、$(2, 3)$ 和 $(3, 5)$。我们的目标是为这些点找到形式为 $y = mx + b$ 的最佳拟合直线。



![简单数据集和初始模型](plots/2320-0.json)



> 我们的包含三个点的小型数据集，以及我们直线的初始猜测：$y = 0x + 0$。

我们需要一种方法来衡量我们直线的“好坏”。我们将使用我们之前讨论过的**均方误差 (MSE)** 作为成本函数。对于我们的三个数据点 $(x_1, y_1), (x_2, y_2), (x_3, y_3)$，MSE 为：


$$
J(m, b) = \frac{1}{3} \sum_{i=1}^{3} (y_i - \hat{y}_i)^2 = \frac{1}{3} \sum_{i=1}^{3} (y_i - (mx_i + b))^2
$$


让我们从参数 (parameter)的初始猜测开始：$m = 0$ 和 $b = 0$。我们的直线初始为 $y = 0$。

### 计算初始成本

首先，让我们计算初始猜测 ($m=0, b=0$) 的成本。
预测值 ($\hat{y}_i = mx_i + b$) 如下：

- $\hat{y}_1 = 0(1) + 0 = 0$
- $\hat{y}_2 = 0(2) + 0 = 0$
- $\hat{y}_3 = 0(3) + 0 = 0$

平方误差为：

- $(y_1 - \hat{y}_1)^2 = (2 - 0)^2 = 4$
- $(y_2 - \hat{y}_2)^2 = (3 - 0)^2 = 9$
- $(y_3 - \hat{y}_3)^2 = (5 - 0)^2 = 25$

均方误差为：


$$
J(0, 0) = \frac{1}{3} (4 + 9 + 25) = \frac{38}{3} \approx 12.67
$$


这是我们的初始成本。我们的目标是通过调整 $m$ 和 $b$ 来降低这个值。

### 计算梯度

现在进入核心微积分部分：找到成本函数 $J(m, b)$ 的梯度。梯度是一个包含关于每个参数 (parameter)的偏导数的向量 (vector)：$\nabla J = \left[ \frac{\partial J}{\partial m}, \frac{\partial J}{\partial b} \right]$。这些导数表示当我们微小改变 $m$ 或 $b$ 时，成本如何变化。

让我们找到 $\frac{\partial J}{\partial m}$ 和 $\frac{\partial J}{\partial b}$。请记住，在对一个变量（如 $m$）取偏导数时，我们将其他变量（如 $b$）视为常量。

#### 关于 $m$ 的偏导数 ($\frac{\partial J}{\partial m}$)

我们从成本函数开始：


$$
J(m, b) = \frac{1}{3} \left[ (y_1 - (mx_1 + b))^2 + (y_2 - (mx_2 + b))^2 + (y_3 - (mx_3 + b))^2 \right]
$$


我们逐项对 $m$ 求导。让我们关注一个项：$(y_i - (mx_i + b))^2$。我们使用链式法则。设 $u = y_i - mx_i - b$。那么该项是 $u^2$。
$u^2$ 关于 $m$ 的导数是 $2u \cdot \frac{\partial u}{\partial m}$。
现在，我们找到 $\frac{\partial u}{\partial m} = \frac{\partial}{\partial m} (y_i - mx_i - b)$。由于在对 $m$ 求导时 $y_i$、$x_i$ 和 $b$ 被视为常量，这简化为 $\frac{\partial}{\partial m} (-mx_i) = -x_i$。

因此，$(y_i - (mx_i + b))^2$ 关于 $m$ 的导数是 $2(y_i - mx_i - b)(-x_i)$。

将此应用于我们的成本函数 $J(m, b)$：


$$
\frac{\partial J}{\partial m} = \frac{1}{3} \sum_{i=1}^{3} 2(y_i - (mx_i + b))(-x_i)
$$


$$
\frac{\partial J}{\partial m} = -\frac{2}{3} \sum_{i=1}^{3} x_i(y_i - (mx_i + b))
$$


#### 关于 $b$ 的偏导数 ($\frac{\partial J}{\partial b}$)

类似地，我们对 $J(m, b)$ 关于 $b$ 求导。同样，考虑一个项 $(y_i - (mx_i + b))^2$。设 $u = y_i - mx_i - b$。
关于 $b$ 的导数是 $2u \cdot \frac{\partial u}{\partial b}$。
现在，$\frac{\partial u}{\partial b} = \frac{\partial}{\partial b} (y_i - mx_i - b)$。将 $y_i$、$m$ 和 $x_i$ 视为常量，这得到 $\frac{\partial}{\partial b} (-b) = -1$。

因此，$(y_i - (mx_i + b))^2$ 关于 $b$ 的导数是 $2(y_i - mx_i - b)(-1)$。

将此应用于成本函数：


$$
\frac{\partial J}{\partial b} = \frac{1}{3} \sum_{i=1}^{3} 2(y_i - (mx_i + b))(-1)
$$


$$
\frac{\partial J}{\partial b} = -\frac{2}{3} \sum_{i=1}^{3} (y_i - (mx_i + b))
$$


### 数值梯度计算

现在我们将数据点 $(1, 2), (2, 3), (3, 5)$ 和当前参数 (parameter) $m=0, b=0$ 代入这些公式中。请记住我们的预测值为 $\hat{y}_1=0, \hat{y}_2=0, \hat{y}_3=0$。

**计算 $\frac{\partial J}{\partial m}$：**


$$
\frac{\partial J}{\partial m} = -\frac{2}{3} \left[ x_1(y_1 - \hat{y}_1) + x_2(y_2 - \hat{y}_2) + x_3(y_3 - \hat{y}_3) \right]
$$


$$
\frac{\partial J}{\partial m} = -\frac{2}{3} \left[ 1(2 - 0) + 2(3 - 0) + 3(5 - 0) \right]
$$


$$
\frac{\partial J}{\partial m} = -\frac{2}{3} \left[ 1(2) + 2(3) + 3(5) \right]
$$


$$
\frac{\partial J}{\partial m} = -\frac{2}{3} \left[ 2 + 6 + 15 \right] = -\frac{2}{3} [23] = -\frac{46}{3} \approx -15.33
$$


**计算 $\frac{\partial J}{\partial b}$：**


$$
\frac{\partial J}{\partial b} = -\frac{2}{3} \left[ (y_1 - \hat{y}_1) + (y_2 - \hat{y}_2) + (y_3 - \hat{y}_3) \right]
$$


$$
\frac{\partial J}{\partial b} = -\frac{2}{3} \left[ (2 - 0) + (3 - 0) + (5 - 0) \right]
$$


$$
\frac{\partial J}{\partial b} = -\frac{2}{3} \left[ 2 + 3 + 5 \right] = -\frac{2}{3} [10] = -\frac{20}{3} \approx -6.67
$$


因此，在 $m=0, b=0$ 时，梯度为 $\nabla J \approx [-15.33, -6.67]$。

### 解释梯度

这些数字告诉我们什么？

- $\frac{\partial J}{\partial m} \approx -15.33$：这是负数，意味着如果我们稍稍*增加* $m$，成本 $J$ 将会*降低*。大小 (15.33) 表明成本对 $m$ 的变化非常敏感。
- $\frac{\partial J}{\partial b} \approx -6.67$：这也是负数。稍稍增加 $b$ 也会降低成本 $J$。在这一点上，成本对 $b$ 的变化不如对 $m$ 的变化敏感。

梯度 $[-15.33, -6.67]$ 指向成本*最陡峭上升*的方向。为了*降低*成本（这是我们在优化中的目标），我们需要沿着梯度的*相反*方向移动。这意味着我们应该增加 $m$ 和 $b$。

### 迈出小步（梯度下降 (gradient descent)）

让我们执行一步梯度下降。我们需要一个学习率 $\alpha$。我们选择一个较小的值，例如 $\alpha = 0.01$。

更新规则如下：

- $m_{new} = m_{old} - \alpha \frac{\partial J}{\partial m}$
- $b_{new} = b_{old} - \alpha \frac{\partial J}{\partial b}$

代入我们的值 ($m_{old}=0, b_{old}=0$， $\alpha=0.01$， $\frac{\partial J}{\partial m} = -46/3$， $\frac{\partial J}{\partial b} = -20/3$)：

- $m_{new} = 0 - (0.01) \left(-\frac{46}{3}\right) = 0.01 \times \frac{46}{3} = \frac{0.46}{3} \approx 0.153$
- $b_{new} = 0 - (0.01) \left(-\frac{20}{3}\right) = 0.01 \times \frac{20}{3} = \frac{0.20}{3} \approx 0.067$

我们的新参数 (parameter)大约是 $m \approx 0.153$ 和 $b \approx 0.067$。

让我们可视化新直线 $y \approx 0.153x + 0.067$：



![一步后的数据集和直线](plots/2320-1.json)



> 我们的数据点，以及初始直线（虚线红色）和梯度下降一步后的直线（实线绿色）。新直线稍微更接近数据点。

如果我们计算成本 $J(0.153, 0.067)$，我们会发现它低于我们 $12.67$ 的初始成本（如在思考过程中所计算的，大约是 10.03）。通过重复这个过程——计算梯度和更新参数——梯度下降会迭代地找到 $m$ 和 $b$ 的更优值，从而最小化成本函数。

这次手动计算展示了其机制：导数（梯度）告诉我们调整参数（$m$ 和 $b$）的方向，以通过降低成本函数（MSE）来改进我们的模型。尽管库可以自动化此过程，但理解底层计算对于掌握机器学习 (machine learning)模型如何学习至关重要。

## 参考资料

- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — Andrew Ng, Geoff Ladwig, Aarti Bagul, Eddy Shyu (2022)
  Publisher: DeepLearning.AI, Stanford Online
  一门广泛可访问的在线课程，通过实际解释介绍了线性回归、成本函数（如MSE）和梯度下降算法。与初始模型设置和优化步骤相关。
- [Mathematics for Machine Learning](https://mml-book.github.io/) — Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong (2020)
  Publisher: Cambridge University Press; DOI: [10.1017/9781108679902](https://doi.org/10.1017/9781108679902)
  这本教材为机器学习建立了坚实的数学基础，其中关于多元微积分、偏导数和链式法则的部分对于理解手动梯度计算至关重要。
- [Pattern Recognition and Machine Learning](https://www.springer.com/gp/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer
  一本经典教材，详细阐述了用于回归的线性模型、各种成本函数以及梯度下降等优化技术的理论基础。
