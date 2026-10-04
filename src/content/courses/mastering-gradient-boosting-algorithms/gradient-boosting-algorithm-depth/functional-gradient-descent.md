---
course: "mastering-gradient-boosting-algorithms"
chapter: "gradient-boosting-algorithm-depth"
lesson: "functional-gradient-descent"
sourceId: 1929
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/functional-gradient-descent"
title: "函数梯度下降"
description: "将提升法视为函数空间中的梯度下降。"
order: 1
plots: []
sourceHash: "13a78ed3fb51690da2dd4603a9289b7081cb324666cb6ae91f20ec1eb77d1d9f"
sourceCorrections: []
---

标准机器学习 (machine learning)算法，如线性回归或具有固定结构的神经网络 (neural network)，其优化通常涉及为预定义模型结构 $f(x; \theta)$ 寻找最佳的 *参数 (parameter)* $\theta$ 集合。梯度下降 (gradient descent)等方法被用来迭代更新这些参数，以最小化在训练数据上平均的损失函数 (loss function) $L(y, f(x; \theta))$。更新规则通常是 $\theta_{new} = \theta_{old} - \eta \nabla_{\theta} L$，其中 $\nabla_{\theta} L$ 是损失函数关于参数的梯度。

梯度提升机（GBM）从不同的角度进行优化。GBM 不在固定的模型结构中优化参数，而是迭代地构建模型*本身*。我们优化的“参数”是表示集成模型的整个函数 $F(x)$。我们要在可能的函数空间中寻找一个函数，使其能最大程度最小化训练数据上的总损失。这种优化过程可以被视为在*函数空间*中执行梯度下降。

### 函数空间中的优化

设想一个高维空间 (high-dimensional space)，其中每个点都对应一个特定函数 $F$。我们的目标是在此空间中找到函数 $F^*$，使其最小化数据集 $(x_i, y_i)$ 在 $i=1, \dots, N$ 上的总损失：


$$
L_{total}(F) = \sum_{i=1}^{N} L(y_i, F(x_i))
$$


就像在参数 (parameter)空间中一样，我们希望迭代地改进当前对最优函数的估计。设 $F_{m-1}(x)$ 是经过 $m-1$ 次提升迭代后的当前集成模型。我们希望找到一个新的函数 $h_m(x)$（我们的基学习器，通常是决策树），将其添加到当前模型能使我们更接近最小损失。也就是说，我们希望：


$$
F_m(x) = F_{m-1}(x) + \eta h_m(x)
$$


其中 $\eta$ 是步长（学习率），且 $h_m(x)$ 指向一个能降低总损失 $L_{total}$ 的方向。

### 梯度方向

在标准梯度下降 (gradient descent)中，负梯度 $-\nabla_{\theta} L$ 指示了在参数 (parameter)空间中能够带来损失最陡峭下降的方向。那么在函数空间中，与之对应的是什么呢？

我们可以将每个训练样本 $i$ 的预测值 $F(x_i)$ 视为定义当前在函数空间中位置的“坐标”。总损失 $L_{total}$ 关于这些坐标的梯度是一个向量 (vector)，其中第 $i$ 个分量是单个损失项 $L(y_i, F(x_i))$ 关于模型预测值 $F(x_i)$ 的偏导数，在当前模型 $F_{m-1}$ 处进行评估：


$$
g_{im} = \left[ \frac{\partial L(y_i, F(x))}{\partial F(x)} \right]_{F(x) = F_{m-1}(x_i)}
$$


这个向量 $(g_{1m}, g_{2m}, \dots, g_{Nm})$ 表示在由 $F_{m-1}$ 定义的函数空间点上，总损失最陡峭*上升*的方向。

### 伪残差：基学习器的目标

为了最小化损失，我们需要沿着与梯度相反的方向移动。我们定义*负梯度*分量，通常被称为**伪残差**，用于迭代 $m$ 中的每个样本 $i$：


$$
r_{im} = - g_{im} = - \left[ \frac{\partial L(y_i, F(x))}{\partial F(x)} \right]_{F(x) = F_{m-1}(x_i)}
$$


这些伪残差 $r_{im}$ 表示我们的下一个基学习器 $h_m(x)$ 应该去近似的目标值。为什么叫“伪残差”？

考虑使用平方误差损失的回归情况：$L(y, F(x)) = \frac{1}{2}(y - F(x))^2$。
梯度分量为：


$$
\frac{\partial L(y_i, F(x))}{\partial F(x)} = \frac{\partial}{\partial F(x)} \left( \frac{1}{2}(y_i - F(x))^2 \right) = -(y_i - F(x))
$$


在 $F_{m-1}(x_i)$ 处评估此式得到 $-(y_i - F_{m-1}(x_i))$。
因此，负梯度为：


$$
r_{im} = - [-(y_i - F_{m-1}(x_i))] = y_i - F_{m-1}(x_i)
$$


在这种特定情况下，伪残差恰好是普通的残差（真实值与当前模型预测值之间的差异）。对于其他损失函数 (loss function)，$r_{im}$ 并非简单的残差，但它仍表示当前模型 $F_{m-1}$ 最需要改进的方向（在函数空间中逐点评估）。

### 执行步进

我们不能直接将伪残差向量 (vector) $(r_{1m}, \dots, r_{Nm})$ “添加”到我们的函数 $F_{m-1}(x)$。相反，我们拟合一个基学习器 $h_m(x)$，以根据输入特征 $x_i$ 来预测这些伪残差：


$$
h_m = \arg \min_{h} \sum_{i=1}^{N} (r_{im} - h(x_i))^2
$$


这个基学习器 $h_m(x)$ 提供了在整个输入域上所需的负梯度步长的近似，不仅限于训练点。

最后，我们通过添加这个新的学习器并按学习率 $\eta$ 进行缩放来更新集成模型：


$$
F_m(x) = F_{m-1}(x) + \eta h_m(x)
$$


这种更新在函数空间中执行近似的梯度下降 (gradient descent)步骤。通过基于当前集成模型的误差迭代计算伪残差，并用新的基学习器拟合这些伪残差，GBM 算法逐步最小化总损失函数 (loss function)。这种函数梯度视角提供了一个有力的统一框架，有助于理解梯度提升如何在不同损失函数和基学习器下工作。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-Function-Approximation-A-Gradient-Boosting-Machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍了梯度提升机（GBM）框架，形式化了函数梯度下降视角及其在各种损失函数中的应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 337-387; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本经典的统计学习教科书，提供全面的统计学习视角，其中包含关于提升及其函数梯度解释的详细章节（第10章）。
- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD '16); Publisher: ACM; Pages: 785–794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  详细介绍了高度可伸缩的梯度提升系统的实现，进一步阐述了带有二阶梯度和正则化的函数梯度方法。
- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](http://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) — Guolin Ke, Qi Meng, Thomas Finley, Taifeng Wang, Wei Chen, Weidong Ma, Qiwei Ye, Tie-Yan Liu (2017)
  Journal: Advances in Neural Information Processing Systems 30 (NIPS 2017); Publisher: Curran Associates, Inc.; Pages: 3149-3157
  介绍了LightGBM，这是另一个高效的梯度提升框架，通过新颖的技术提升了函数梯度方法的速度和内存效率。
