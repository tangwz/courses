---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-foundations-revisited"
lesson: "gradient-descent-fundamentals"
sourceId: 1923
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-1-gradient-boosting-foundations-revisited/gradient-descent-fundamentals"
title: "梯度下降基本原理"
description: "回顾与提升（boosting）算法相关的梯度下降优化思想。"
order: 4
plots: ["plots/1923-0.json"]
sourceHash: "f09f6f1e6e07794d498c224c4ec670baf4641a72c34c1d977ae39956c4d9e6a2"
sourceCorrections: []
---

为了叠加地构建模型，使每个新组件都能纠正集成模型截至目前的误差，我们需要一个系统方法来确定进行*何种*修正。这就是优化技术发挥作用的地方，其中最重要的一种是**梯度下降 (gradient descent)**。尽管您可能在通过调整权重 (weight)参数 (parameter)训练线性回归或神经网络 (neural network)等模型时接触过梯度下降，但它在梯度提升中的应用略有不同但相互关联。在看到它如何应用于提升算法之前，理解其核心机制很重要。

### 目标：最小化损失

监督学习 (supervised learning)的核心在于最小化**损失函数 (loss function)**，通常表示为 $L(y, F(x))$。此函数量化 (quantization)了真实目标值 ($y$) 与我们模型 ($F(x)$) 所做预测之间的差异。常见例子包括回归中的均方误差（MSE）或分类中的对数损失。我们的目标是找到一个模型 $F(x)$，使其在整个训练数据上的损失值尽可能小。

### 直观理解：沿着坡度下降

把损失函数 (loss function)想象成一片有山丘和山谷的地形，其中任意点的高度代表给定模型配置的损失值。我们想找到这片地形中的最低点，即损失函数的最小值。梯度下降 (gradient descent)提供了一个简单的迭代策略：

1. 从某个点（初始模型）开始。
2. 确定从当前点出发最陡峭的*下降*方向。
3. 沿着该方向迈出一小步。
4. 重复步骤2和3，直到无法再向下移动（我们已达到一个最小值，希望是全局最小值）。

如何找到最陡峭的下降方向？微积分告诉我们，损失函数的**梯度**，表示为 $\nabla L$，指向最陡峭的*上升*方向。因此，要向下移动，我们只需沿着梯度的*反方向*移动，即沿着 $-\nabla L$。

### 更新规则

在优化模型参数 (parameter) $\theta$ 时，第 $t$ 次迭代的基本梯度下降 (gradient descent)更新规则是：


$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
$$


让我们分解理解一下：

- $\theta_t$: 当前迭代中的模型参数集合。
- $\nabla L(\theta_t)$: 损失函数 (loss function)关于参数 $\theta$ 的梯度，在当前值 $\theta_t$ 处求得。该向量 (vector)指示了改变参数以*最快速度增加*损失的方向。
- $\eta$: **学习率**，一个小的正标量。这个超参数 (hyperparameter)控制着我们在负梯度方向上迈出步长的大小。
- $\theta_{t+1}$: 下一次迭代的更新参数集合。

该过程重复进行，迭代调整参数以逐步降低损失。

### 学习率 ($\eta$)

学习率 $\eta$ 的选择对算法性能很重要：

- **如果 $\eta$ 过大：** 我们可能会越过最小值，可能导致损失剧烈波动甚至增加，从而导致发散。
- **如果 $\eta$ 过小：** 向最小值收敛会非常缓慢，需要多次迭代和大量计算时间。

找到合适的学习率通常需要实验和调优。在梯度提升中，这个学习率通常被称为**收缩率**，它在控制学习过程和充当一种正则化 (regularization)形式方面发挥双重作用，我们稍后会进行说明。



![梯度下降中学习率的影响](plots/1923-0.json)



> 一个简化的1D示例 ($L=x^2$) 展示了不同的学习率 ($\eta$) 如何影响梯度下降 (gradient descent)从 $x=4$ 开始所走的步长。目标是达到 $x=0$。合适的学习率能有效收敛，较小的学习率收敛缓慢，较大的学习率会越过最小值，而过大的学习率会导致发散。

尽管基本更新规则使用在整个数据集上计算的梯度（批量梯度下降），但存在变体。**随机梯度下降（SGD）** 基于单个数据点计算梯度，而**小批量梯度下降**使用数据的一个小部分。这些方法的计算成本可能更低，有时有助于逃离浅层局部最小值，尽管标准的GBM通常在整个数据集上计算梯度，除非明确使用了随机子采样。

### 函数空间中的梯度下降 (gradient descent)

那么，这与叠加地构建树的集成模型有何关联呢？梯度提升不是优化一组固定的参数 (parameter) $\theta$，而是在**函数空间**中进行优化。我们从一个初始简单模型开始（例如，回归中目标值的均值）。然后在每次迭代 $m$ 中，我们希望找到一个新函数 $h_m(x)$（我们的基本学习器，通常是决策树）将其添加到当前集成 $F_{m-1}(x)$ 中，以便整体损失降低：


$$
F_m(x) = F_{m-1}(x) + \eta h_m(x)
$$


梯度提升在此处使用了梯度下降的思想：它计算损失函数 (loss function) $L(y, F(x))$ 相对于*当前模型预测* $F_{m-1}(x)$ 的负梯度，对每个训练实例 $i$ 进行评估：


$$
r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right]_{F(x) = F_{m-1}(x)}
$$


这些负梯度 $r_{im}$ 通常被称为**伪残差**，代表了在函数空间中，给定当前集成模型的预测，损失对于每个数据点下降最快的“方向”。算法随后拟合新的基本学习器 $h_m(x)$ 以近似这些伪残差。本质上，我们正在使用梯度下降来指导集成模型的序列构建，告诉我们下一个树应该侧重于纠正哪些误差（由损失的负梯度定义）。

将提升算法视为函数空间中的梯度下降，这是贯穿本课程所讨论算法的核心思想。我们将在下一章中推导通用的梯度提升机算法时，进一步使其规范化。目前，重要的收获是梯度下降提供了优化机制，通过添加基本学习器来修正剩余误差，正如损失函数的梯度所指明的那样，从而迭代改进集成模型。

## 参考资料

- [Pattern Recognition and Machine Learning](https://link.springer.com/book/9780387310732) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 234-239
  这本基础教科书清晰地介绍了梯度下降作为机器学习中的一种优化技术，尤其是在参数更新方面。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; Pages: 337-360
  对梯度提升进行了全面介绍，详细阐述了其作为函数梯度下降的解释以及潜在的损失最小化过程。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍梯度提升机的原始论文，将提升作为函数梯度下降的概念进行了形式化。
- [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — Stephen Boyd and Lieven Vandenberghe (2004)
  Publisher: Cambridge University Press; Pages: 457-463
  为梯度下降方法及其在优化问题中的特性提供了严格的数学基础。
