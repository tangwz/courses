---
course: "building-ml-recommendation-system"
chapter: "model-based-collaborative-filtering"
lesson: "optimization-with-sgd"
sourceId: 7473
sourceUrl: "https://apxml.com/zh/courses/building-ml-recommendation-system/chapter-4-model-based-collaborative-filtering/optimization-with-sgd"
title: "使用随机梯度下降 (SGD) 进行优化"
description: "介绍如何使用随机梯度下降 (SGD) 寻找最优隐类因子矩阵。"
order: 4
plots: []
sourceHash: "7bb50daa019cc5722fef0432e7f4121b80b5bd7a7c4cae93c4239438e84f67e4"
sourceCorrections: []
---

虽然奇异值分解 (SVD) 提供了一种强大的矩阵分解方法，但在推荐系统中直接将其经典形式应用于稀疏的用户-物品交互矩阵往往不切实际。大量缺失值的存在使得直接计算变得困难且低效。我们可以通过迭代优化过程来学习用户和物品的因子矩阵 $P$ 和 $Q$，而不是寻找精确的分解。随机梯度下降 (gradient descent) (SGD) 是一种针对此任务非常有效且扩展性良好的算法。

### 目标：最小化预测误差

我们的目标是找到能够产生最佳预测结果的因子矩阵 $P$ 和 $Q$。我们将“最佳”定义为最小化预测评分 ($\hat{r}_{ui}$) 与实际已知评分 ($r_{ui}$) 之间的差异。衡量这种差异的一种常用方法是平方误差。

为了防止模型仅仅死记硬背训练数据（过拟合 (overfitting)），我们添加了一个正则化 (regularization)项。该项会对因子向量 (vector)中的较大数值进行惩罚，从而鼓励模型发现更简单、更具泛化性的模式。将这些结合起来就得到了我们的目标函数，也就是我们旨在最小化的函数：


$$
E = \sum_{(u,i) \in K} (r_{ui} - p_u \cdot q_i)^2 + \lambda (\sum_{u} ||p_u||^2 + \sum_{i} ||q_i||^2)
$$


这里：

- $K$ 是我们拥有评分的所有用户-物品对的集合。
- $p_u$ 是用户 $u$ 的隐类因子向量。
- $q_i$ 是物品 $i$ 的隐类因子向量。
- $\lambda$ (lambda) 是正则化参数 (parameter)，用于控制惩罚力度。

该方程代表了总的正则化平方误差。我们的任务是找到所有 $p_u$ 和 $q_i$ 的值，使 $E$ 尽可能小。

### 使用梯度下降 (gradient descent)寻找最小值

梯度下降是一种用于寻找函数最小值的迭代优化算法。其核心思路是先对参数 (parameter)（$P$ 和 $Q$ 的元素）设定一个初始猜测值，然后朝着误差减小最快的方向反复调整。我们通过计算误差函数相对于每个参数的梯度（即偏导数）来确定这个方向。

然而，在每一步都使用所有已知评分来计算梯度（批量梯度下降）对于大型数据集来说会非常缓慢。这就是 SGD 的“随机”部分发挥作用的地方。SGD 不使用整个数据集，而是通过一次只考虑一个训练样本来更新参数。对于我们的推荐模型，一个训练样本就是一个已知的评分 $(u, i, r_{ui})$。

对于每个评分，我们执行以下步骤：

1. 计算该特定评分的预测误差。
2. 计算误差相对于用户因子向量 (vector) ($p_u$) 和物品因子向量 ($q_i$) 的梯度。
3. 沿着梯度的反方向迈出一小步，从而更新这两个向量。

由目标函数的偏导数推导出的更新规则非常直观。首先，我们计算预测误差：


$$
e_{ui} = r_{ui} - \hat{r}_{ui} = r_{ui} - p_u \cdot q_i
$$


然后，我们利用这个误差来调整用户 $u$ 和物品 $i$ 的因子向量：


$$
p_u \leftarrow p_u + \alpha (e_{ui} \cdot q_i - \lambda \cdot p_u)
$$


$$
q_i \leftarrow q_i + \alpha (e_{ui} \cdot p_u - \lambda \cdot q_i)
$$


参数 $\alpha$ (alpha) 是学习率，它控制我们调整的步长。较小的学习率会导致收敛缓慢但稳定，而较大的学习率虽然能加快学习速度，但存在越过最小值的风险。

### 矩阵分解的 SGD 算法

完整的算法包括初始化因子矩阵，然后多次遍历数据集，为每个已知评分更新因子。

1. **初始化：** 创建用户因子矩阵 $P$ 和物品因子矩阵 $Q$。用较小的随机数填充它们。这种随机启动对于打破对称性并允许因子学习不同的特征非常。
2. **迭代：** 按照设定的轮数（epochs）循环遍历训练数据。一轮是指对整个训练数据集进行一次完整处理。
3. **打乱与处理：** 在每一轮中，打乱评分的顺序是一个好的做法。然后，遍历每个已知评分 $(u, i, r_{ui})$。
4. **更新：** 对于每个评分，计算预测误差 $e_{ui}$ 并应用更新规则来调整向量 (vector) $p_u$ 和 $q_i$。

通过重复此过程，$P$ 和 $Q$ 中的向量会逐渐从随机初始值转变为能够编码有意义的隐类特征的数值，从而最小化训练数据上的整体预测误差。

> 针对单个用户-物品评分的 SGD 迭代更新过程。预测评分与实际评分之间的误差被用来调整用户和物品的隐类因子向量。

### 调整超参数 (parameter) (hyperparameter)

通过 SGD 训练的模型性能很大程度上由其超参数决定。我们模型中最重要的两个超参数是：

- **学习率 ($\alpha$):** 该参数决定了算法在每次更新期间迈出的步长。如果 $\alpha$ 太大，算法可能会变得不稳定并产生发散，导致误差不减反增。如果太小，训练会非常缓慢。一种通用的方法是从 0.005 之类的值开始，然后根据模型在验证集上的表现进行调整。
- **正则化 (regularization)参数 ($\lambda$):** 该参数控制在拟合训练数据与保持因子向量 (vector)较小（以避免过拟合 (overfitting)）之间的权衡。较大的 $\lambda$ 会对较大的因子值施加更强的惩罚，从而产生一个可能泛化效果更好但可能对训练数据拟合不足的简化模型。较小的 $\lambda$ 给予模型更多拟合训练数据的自由，但会增加过拟合的风险。典型值通常在 0.01 到 0.1 之间。

找到这些超参数的正确组合通常需要通过实验以及网格搜索或随机搜索等技术，我们将在评估模型时研究这些技术。通过掌握 SGD，你可以训练出既具有扩展性又能有效发现用户隐藏偏好的矩阵分解模型。

## 参考资料

- [Matrix Factorization Techniques for Recommender Systems](https://doi.org/10.1109/MC.2009.263) — Yehuda Koren, Robert Bell, Chris Volinsky (2009)
  Journal: Computer; Publisher: IEEE Computer Society; Volume: 42; Pages: 30-37; DOI: [10.1109/MC.2009.263](https://doi.org/10.1109/MC.2009.263)
  本文介绍了并推广了用于推荐系统的带有SGD的矩阵分解技术，尤其是在Netflix Prize中的成功之后。它描述了目标函数和迭代优化过程。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, Aaron Courville (2016)
  Publisher: MIT Press
  第8章介绍了优化算法，包括SGD和多种正则化方法。这些内容对于理解迭代模型训练的基本机制至关重要。
- [Matrix Factorization Methods for Recommender Systems](https://doi.org/10.1007/978-1-4939-2708-2_5) — Yehuda Koren, Robert Bell (2015)
  Publisher: Springer; Pages: 143-176; DOI: [10.1007/978-1-4939-2708-2_5](https://doi.org/10.1007/978-1-4939-2708-2_5)
  此章节来自《推荐系统手册》（第二版），详细介绍了推荐系统中的矩阵分解，解释了多种模型和优化策略，包括SGD的应用。
- [Lecture Notes on Gradient Descent, CS229](http://cs229.stanford.edu/notes2022fall/main_notes.pdf) — Andrew Ng (2022)
  这些备受推崇的讲义清晰地介绍了梯度下降，为矩阵分解中SGD优化提供了基础。
