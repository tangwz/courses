---
course: "building-ml-recommendation-system"
chapter: "model-based-collaborative-filtering"
lesson: "regularization-to-prevent-overfitting"
sourceId: 7474
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-4-model-based-collaborative-filtering/regularization-to-prevent-overfitting"
title: "防止过拟合的正则化"
description: "将正则化技术应用于矩阵分解模型，以提高泛化能力并防止过拟合。"
order: 5
plots: ["plots/7474-0.json"]
sourceHash: "2b073ace784f66dc1b6e7e2806b238484af2ede59cfdb3438797a0663eb5a36b"
sourceCorrections: []
---

在使用随机梯度下降 (gradient descent)（SGD）训练矩阵分解模型时，主要目标是尽量减少训练数据上的预测误差。虽然这听起来很合理，但过于追求这一目标会导致一个严重的问题：过拟合 (overfitting)。过拟合的模型会过度学习训练数据中的细节和噪声，以至于无法很好地推导至新的、未见过的数据。

在矩阵分解的场景中，过拟合意味着模型生成的效用向量 (vector)（用户向量 $p_u$ 和物品向量 $q_i$）过度拟合了训练集中的评分。这些向量往往包含极大或极小的数值，虽然这能让模型完美地还原已知评分，但对于未知评分，往往会产生糟糕且极端的预测。此时，模型实际上是记住了训练样本，而不是学习用户品味的内在模式。

### 惩罚模型复杂度

为了应对过拟合 (overfitting)，我们可以引入一种称为正则化 (regularization)的技术。其思路是修改代价函数，对模型的复杂度进行惩罚。对于矩阵分解而言，复杂度与隐含特征向量 (vector)中数值的大小有关。通过对较大的特征值施加惩罚，我们鼓励模型寻找更简单的解，在解释数据的同时避免拟合噪声。

最常用的方法是 L2 正则化。我们在原始的平方误差代价函数中添加一个惩罚项，该项与用户矩阵和物品矩阵中所有元素平方和成正比。

更新后的代价函数如下：


$$
\text{代价} = \sum_{(u,i) \in K} (r_{ui} - p_u^T q_i)^2 + \lambda \left( \sum_{u} ||p_u||^2 + \sum_{i} ||q_i||^2 \right)
$$


让我们分析一下这个新公式：

- 第一项 $\sum_{(u,i) \in K} (r_{ui} - p_u^T q_i)^2$ 与我们在上一节中尝试最小化的平方误差和相同。它促使模型做出准确的预测。
- 第二项 $\lambda \left( \sum_{u} ||p_u||^2 + \sum_{i} ||q_i||^2 \right)$ 是 L2 正则化惩罚项。
  - $||p_u||^2$ 是用户隐含向量的 L2 范数平方，计算方式为 $\sum_{k=1}^{K} p_{uk}^2$。即其各元素平方之和。
  - $||q_i||^2$ 是物品隐含向量的 L2 范数平方，计算方式为 $\sum_{k=1}^{K} q_{ik}^2$。
  - $\lambda$ (lambda) 是一个超参数 (parameter) (hyperparameter)，用于控制正则化的强度。$\lambda$ 值越高，惩罚越重，迫使隐含特征值变得更小，从而产生更简单的模型。如果 $\lambda$ 为 0，则完全取消正则化。

现在的优化过程必须平衡两个目标：最小化预测误差和保持隐含特征向量的数值较小。

### 修改 SGD 更新规则

代价函数的这种变化也改变了我们在 SGD 更新规则中使用的梯度。当我们计算关于 $p_{uk}$ 和 $q_{ik}$ 的偏导数时，正则化 (regularization)项增加了一个新组件。由此产生的更新规则变为：

对于训练集中的每个评分 $r_{ui}$：

1. 计算预测误差：$e_{ui} = r_{ui} - p_u^T q_i$
2. 更新用户向量 (vector) $p_u$：
   $p_u \leftarrow p_u + \eta \cdot (e_{ui} \cdot q_i - \lambda \cdot p_u)$
3. 更新物品向量 $q_i$：
   $q_i \leftarrow q_i + \eta \cdot (e_{ui} \cdot p_u - \lambda \cdot q_i)$

注意新增的部分：$- \lambda \cdot p_u$ 和 $- \lambda \cdot q_i$。在每次更新中，除了沿着误差梯度的方向移动向量外，我们还按 $\lambda$ 的比例稍微缩小它们。这可以防止隐含向量中的任何单个元素增长得过大并在点积计算中占据主导地位，从而使模型更稳定，泛化能力更强。

### 寻找 Lambda 的平衡点

$\lambda$ 的选择对模型性能非常重要。它控制着拟合训练数据与保持模型简单之间的权衡。

- **如果 $\lambda$ 过低：** 正则化 (regularization)效果微弱，模型可能仍然会过拟合 (overfitting)。
- **如果 $\lambda$ 过高：** 对大向量 (vector)值的惩罚过于严重。模型会产生非常小的隐含特征，可能会变得过于简单而无法捕捉数据中的潜在模式，这种情况称为欠拟合 (underfitting)。

$\lambda$ 的理想值通常通过实验找到。你可以使用不同的 $\lambda$ 值训练模型，并在独立的验证数据集（不用于训练的数据部分）上评估它们的表现。选择在验证集上表现最好的那个值。



![正则化对模型误差的影响](plots/7474-0.json)



> 正则化强度、训练误差和验证误差之间的关系。随着 $\lambda$ 的增加，训练误差上升，因为模型受到了更多限制。验证误差通常会下降到一个最低点（即最佳 $\lambda$），然后由于欠拟合而再次上升。

在实践中，大多数推荐系统库（包括我们即将使用的库）都提供了参数 (parameter)来轻松控制正则化，因此你可以专注于调整其强度，而无需从头实现更新规则。

## 参考资料

- [Matrix Factorization Techniques for Recommender Systems](https://www.computer.org/csdl/magazine/co/2009/08/mco2009080030/13rruB7x4gA) — Yehuda Koren, Robert Bell, and Chris Volinsky (2009)
  Journal: Computer; Publisher: IEEE Computer Society; Volume: 42; Pages: 30-37; DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)
  协同过滤中矩阵分解的开创性论文，详细阐述了L2正则化在防止过拟合和提高泛化能力方面的应用。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本内容全面的教科书，涵盖了L2正则化（岭回归）的理论基础及其在控制模型复杂性方面的作用。
- [Recommender Systems: The Textbook](https://link.springer.com/book/10.1007/978-3-319-29659-3) — Charu C. Aggarwal (2016)
  Publisher: Springer; DOI: [10.1007/978-3-319-29659-3](https://doi.org/10.1007/978-3-319-29659-3)
  提供了推荐系统中矩阵分解模型的详细概述，包括正则化技术的实际应用和重要性。
