---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "the-gradient-boosting-machine"
lesson: "gbm-regression-loss-functions"
sourceId: 7581
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-2-the-gradient-boosting-machine/gbm-regression-loss-functions"
title: "回归任务的损失函数"
description: "了解梯度提升中用于回归问题的常用损失函数，包括均方误差 (MSE) 和平均绝对误差 (MAE)。"
order: 4
plots: ["plots/7581-0.json"]
sourceHash: "ad6a27c84064f8deb614d0e1dd1dae224d0f3fcbe8a0647be7907bbbf423ad4a"
sourceCorrections: []
---

梯度提升机 (GBM) 的灵活性源于其能够优化任何可微分的损失函数 (loss function)。对于回归问题，这意味着我们可以选择一个最能契合我们性能目标和数据误差特点的函数。损失函数通过界定每个后续树应尝试修正的“错误”，直接指引算法。

### 均方误差 (L2 损失)

回归问题中最普遍的选择，也常是许多库中的默认设置，是**均方误差 (MSE)**，亦称 L2 损失。它衡量误差平方的平均值。对于单个预测，损失计算如下：


$$
L(y_i, \hat{y}_i) = \frac{1}{2}(y_i - \hat{y}_i)^2
$$


引入 $\frac{1}{2}$ 这个因子是为了数学上的便利，因为它简化了导数。

MSE 的主要特点是它对较大误差的惩罚远重于较小误差。一个 4 的误差受到的惩罚是 1 的误差的 16 倍。这一特性使得模型集中精力减少大误差，这通常是人们乐见的。

当我们求得该损失函数 (loss function)相对于预测 $\hat{y}_i$ 的负梯度时，与 GBM 核心算法的关联变得明确：


$$
- \frac{\partial L}{\partial \hat{y}_i} = - (-(y_i - \hat{y}_i)) = y_i - \hat{y}_i
$$


这展现了一个非常直观的结果：使用 MSE 时，负梯度正好是**残差**。这就是为什么标准的 GBM 回归算法常被描述为每个新树都被训练来预测前一个模型预测的残差。

然而，MSE 对大误差的敏感性也可能是一个弊端。如果您的数据集包含明显异常值，模型可能会将其过多能力用于修正这几个异常点，可能以牺牲其在其余数据上的表现为代价。

### 平均绝对误差 (L1 损失)

作为一种替代方案，我们可以使用**平均绝对误差 (MAE)**，或 L1 损失。该函数衡量真实值与预测值之间绝对差的平均值。


$$
L(y_i, \hat{y}_i) = |y_i - \hat{y}_i|
$$


与 MSE 不同，MAE 线性地惩罚误差。一个 4 的误差受到的惩罚仅是 1 的误差的 4 倍。这使得用 MAE 训练的模型对异常值更有抵抗力，因为单个大误差不会主导梯度计算。

MAE 的负梯度是残差的符号：


$$
- \frac{\partial L}{\partial \hat{y}_i} = \text{符号}(y_i - \hat{y}_i)
$$


梯度为 +1 或 -1（取决于预测值是过低还是过高），表明误差的方向而非其大小。这阻止异常值在后续弱学习器的训练中产生过大的影响。

### Huber 损失

Huber 损失在 MSE 的敏感性与 MAE 的鲁棒性之间提供了一种折衷。它对于小误差呈二次方行为（如 MSE），对于大误差呈线性行为（如 MAE）。这使其对异常值不那么敏感，同时仍能精细调整接近真实值的预测。

该函数分段定义如下：


$$
L_{\delta}(y_i, \hat{y}_i) =
\begin{cases}
\frac{1}{2}(y_i - \hat{y}_i)^2 & \text{当 } |y_i - \hat{y}_i| \le \delta \\
\delta(|y_i - \hat{y}_i| - \frac{1}{2}\delta) & \text{否则}
\end{cases}
$$


参数 (parameter) $\delta$ (delta) 是您可以调整的阈值。它界定了损失函数 (loss function)从二次方转变为线性的点。小于 $\delta$ 的误差用平方项来最小化，而较大误差则用绝对项来最小化。



![回归损失函数的比较](plots/7581-0.json)



> 随着预测误差的增加，不同损失函数的行为表现。MSE 呈二次方增长，严厉惩罚大误差。MAE 呈线性增长，使其对异常值不那么敏感。Huber 损失结合了这些特性，对小误差呈二次方行为，对大误差呈线性行为。

### 您应该选择哪种损失函数 (loss function)？

损失函数的选择应由您的数据集和建模目标来引导。

- **使用均方误差 (MSE)** 当您认为您的目标变量相对没有极端异常值，或者当您希望模型密切关注这些异常值时。它是标准选择，通常表现优异。
- **使用平均绝对误差 (MAE)** 当您的数据集包含可能损害模型的明显异常值时。如果您的目标变量分布有厚尾，这是一个不错的选择。
- **使用 Huber 损失** 当您希望在两者之间取得平衡时。它为大误差提供了 MAE 的鲁棒性，同时保持了 MSE 对小误差的敏感性，使其成为一个多功能且有效的选项。

最终，损失函数是您可用的另一种工具，用于构建一个与您的数据具体特征相符的模型。尝试不同的损失函数有时能明显提升模型表现。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本标准教科书，涵盖多种统计学习方法，包括梯度提升和相关损失函数的完整处理。
- [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  引入梯度提升机算法的原始研究论文，解释了其数学基础和可微分损失函数的使用。
- [Robust Statistics](https://doi.org/10.1002/9780470434259) — Peter J. Huber, Elvezio M. Ronchetti (2009)
  Publisher: Wiley; DOI: [10.1002/9780470434259](https://doi.org/10.1002/9780470434259)
  本书对稳健统计方法进行了权威论述，包括Huber损失的理论背景和特点。
