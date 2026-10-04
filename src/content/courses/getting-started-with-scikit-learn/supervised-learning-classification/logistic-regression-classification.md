---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-classification"
lesson: "logistic-regression-classification"
sourceId: 1855
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-3-supervised-learning-classification/logistic-regression-classification"
title: "逻辑回归用于分类"
description: "使用逻辑回归作为线性模型进行二元和多类别分类任务。"
order: 2
plots: ["plots/1855-0.json", "plots/1855-1.json"]
sourceHash: "5af7a008bd5e832873bc64f13871f05998799b1825ae4a75fc655f99a6f5a153"
sourceCorrections: []
---

尽管“逻辑回归”这个名字包含“回归”，但它实际上是用于**分类**任务最基本且广泛使用的算法之一。在我们对上一章线性模型的理解之上，逻辑回归调整了线性方法来预测类别结果的概率。它不是预测一个连续值，而是估计一个实例属于某个特定类别的概率。

### 从线性回归到概率：Sigmoid 函数

标准线性回归模型使用输入特征 $x$ 的线性组合来预测输出 $y$：

$\hat{y} = w^T x + b$

输出 $\hat{y}$ 的范围可以从 $-\infty$ 到 $+\infty$。这适用于回归问题，但对于分类问题，我们需要一个表示概率的输出，它被限制在 0 和 1 之间。我们如何将线性方程潜在的无界输出映射到这个范围呢？

这就是 **Sigmoid 函数**（也称为**逻辑函数**）发挥作用的地方。它接受任何实数值 $z$，并将其压缩到 (0, 1) 的范围内。Sigmoid 函数的定义如下：

$g(z) = \frac{1}{1 + e^{-z}}$

这里，$z$ 通常是模型线性部分的输出，所以 $z = w^T x + b$。函数 $g(z)$ 给出了输出 $y$ 为 1 的估计概率 $P(y=1 | x; w, b)$，这是给定输入特征 $x$ 和学得参数 (parameter) $w$ (权重 (weight)) 以及 $b$ (偏置 (bias)) 的情况下。

让我们查看 Sigmoid 函数：



![Sigmoid (逻辑) 函数](plots/1855-0.json)



> Sigmoid 函数将任何输入值 $z$ 映射到 0 到 1 之间的输出。当 $z$ 变得非常大的正数时，$g(z)$ 接近 1。当 $z$ 变得非常大的负数时，$g(z)$ 接近 0。当 $z=0$ 时，$g(z)=0.5$。

### 做出预测：决策边界

Sigmoid 函数的输出给出了一个概率。为了做出具体的类别预测（例如，二分类中的 0 或 1），我们需要一个**决策阈值**。一个常用选择是 0.5。

- 如果 $g(z) \ge 0.5$，预测为类别 1。
- 如果 $g(z) < 0.5$，预测为类别 0。

从 Sigmoid 函数的图上看，我们看到 $g(z) \ge 0.5$ 发生在 $z \ge 0$ 的时候。由于 $z = w^T x + b$，我们的决策规则变为：

- 如果 $w^T x + b \ge 0$，预测为 1
- 如果 $w^T x + b < 0$，预测为 0

方程 $w^T x + b = 0$ 定义了**决策边界**。对于使用这种线性形式 $z$ 的逻辑回归，决策边界是一条线（在二维空间中）、一个平面（在三维空间中）或一个超平面（在更高维度空间中），它将两个类别分开。



![线性决策边界示例](plots/1855-1.json)



> 一个散点图示例，显示了两类数据点和逻辑回归学得的线性决策边界。一侧的点被分类为类别 0，另一侧的点被分类为类别 1。

### 逻辑回归如何训练：损失函数 (loss function)

就像线性回归需要一个损失函数来衡量直线拟合数据的程度一样，逻辑回归也需要一个。然而，将线性回归中的均方误差（MSE）与 Sigmoid 函数一起使用，会导致一个具有许多局部最小值的非凸损失函数。这使得梯度下降 (gradient descent)等优化算法难以可靠地找到全局最小值。

取而代之，逻辑回归使用一种称为**对数损失**（也称为二元交叉熵）的损失函数。对于单个训练样本 ($x^{(i)}, y^{(i)}$)，其中 $y^{(i)}$ 是真实标签（0 或 1），且 $h(x^{(i)}) = g(w^T x^{(i)} + b)$ 是类别 1 的预测概率，损失为：

$\text{损失}(h(x^{(i)}), y^{(i)}) = -[y^{(i)} \log(h(x^{(i)})) + (1 - y^{(i)}) \log(1 - h(x^{(i)}))]$

所有 $m$ 个训练样本的总损失函数 $J(w, b)$ 是这些单独损失的平均值：

$J(w, b) = \frac{1}{m} \sum_{i=1}^{m} \text{损失}(h(x^{(i)}), y^{(i)})$
$J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [y^{(i)} \log(h(x^{(i)})) + (1 - y^{(i)}) \log(1 - h(x^{(i)}))]$

让我们分析一下这个损失：

- 如果真实标签 $y^{(i)}=1$，损失是 $-\log(h(x^{(i)}))$。当预测概率 $h(x^{(i)})$ 接近 1 时，此损失接近 0；当 $h(x^{(i)})$ 接近 0 时，此损失接近 $\infty$。当实际类别为 1 时，模型因预测低概率而受到严重惩罚。
- 如果真实标签 $y^{(i)}=0$，损失是 $-\log(1 - h(x^{(i)}))$。当 $h(x^{(i)})$ 接近 0 时（意味着 $1-h(x^{(i)})$ 接近 1），此损失接近 0；当 $h(x^{(i)})$ 接近 1 时，此损失接近 $\infty$。当实际类别为 0 时，模型因预测高概率而受到严重惩罚。

这个对数损失函数是凸的，使其适用于梯度下降等优化算法来找到最优参数 (parameter) $w$ 和 $b$。

### Scikit-learn 中的逻辑回归

Scikit-learn 在 `sklearn.linear_model.LogisticRegression` 类中提供了逻辑回归的直接实现。

这是一个使用它的基本例子：

```python
# 导入所需的库
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification # 用于生成样本数据

# 生成一些合成分类数据
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42, flip_y=0.1)

# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 1. 实例化 LogisticRegression 模型
#    常用参数：
#    - C: 正则化强度的倒数（C 越小，正则化强度越大）。默认值为 1.0。
#    - penalty: 正则化类型（'l1'，'l2'，'elasticnet'，'none'）。默认值为 'l2'。
#    - solver: 用于优化的算法。默认通常是 'lbfgs'。
#             不同的求解器支持不同的惩罚类型。
log_reg = LogisticRegression(solver='liblinear', random_state=42) # liblinear 适用于较小的数据集

# 2. 在训练数据上训练（拟合）模型
log_reg.fit(X_train, y_train)

# 3. 对测试数据进行预测
y_pred = log_reg.predict(X_test)

# 4. 对测试数据预测概率
#    返回形状为 (n_samples, n_classes) 的数组
#    每行总和为 1。列对应于按数值排序的类别。
y_pred_proba = log_reg.predict_proba(X_test)

# 评估模型（我们将在后面详细讲解指标）
accuracy = accuracy_score(y_test, y_pred)

print(f"模型系数 (w): {log_reg.coef_}")
print(f"模型截距 (b): {log_reg.intercept_}")
print(f"样本预测概率（前 5 个）：\n{y_pred_proba[:5]}")
print(f"样本预测（前 5 个）：{y_pred[:5]}")
print(f"测试准确率: {accuracy:.4f}")
```

关于 Scikit-learn 实现的重要事项：

- **正则化 (regularization)：** 默认情况下，`LogisticRegression` 应用 L2 正则化（由 `C` 参数 (parameter)控制）以防止过拟合 (overfitting)，尤其是在处理许多特征时。较小的 `C` 值会增加正则化强度。
- **求解器：** 有多种优化算法（`solver`）可用（`'liblinear'`、`'lbfgs'`、`'sag'`、`'saga'`、`'newton-cg'`）。有些求解器对于大型数据集更快，而其他求解器则支持不同的正则化类型。
- **`predict_proba`：** 当您需要实际的概率估计而不仅仅是最终的类别预测时，此方法很有用。它返回一个数组，其中每行对应一个样本，每列对应一个类别，包含该样本属于该类别的概率。

### 处理多类别分类

如果您有多个类别（例如，{猫，狗，鸟}）怎么办？逻辑回归可以通过两种主要策略来处理多类别问题：

1. **一对多 (OvR)**：也称为一对余 (OvA)。这是 Scikit-learn 的 `LogisticRegression` 默认使用的策略。在这种方法中，对于一个有 $K$ 个类别的问题，会训练 $K$ 个独立的二元逻辑回归分类器。第一个分类器预测类别 1 与 {所有其他类别}，第二个预测类别 2 与 {所有其他类别}，以此类推。当对新实例进行预测时，所有 $K$ 个分类器都会提供一个概率，然后选择对应于概率最高的分类器的类别。
2. **多项逻辑回归 (Softmax 回归)**：这种方法不是训练多个二元分类器，而是将逻辑回归直接推广到处理多个类别。它使用 **Softmax 函数**（Sigmoid 函数的推广）来输出所有 $K$ 个类别的概率分布。您可以在 Scikit-learn 中通过设置 `multi_class='multinomial'` 并使用兼容的求解器（如 `'lbfgs'` 或 `'newton-cg'`）来启用此功能。

对于许多实际应用，默认的 OvR 策略通常运行良好。

### 优点和缺点

**优点：**

- **可解释性：** 类似于线性回归，系数 ($w$) 可以提供对每个特征在概率结果（对数几率）上的重要性和影响方向的理解。
- **效率：** 它的训练和预测计算成本低，使其适用于大型数据集。
- **概率输出：** 提供校准良好的概率，这对于分类中的决策制定很有用。
- **基准模型：** 通常可作为与更复杂算法进行比较的强基准模型。

**缺点：**

- **线性假设：** 假定特征与结果的*对数几率*之间存在线性关系。在没有特征工程（例如，创建多项式特征）的情况下，它可能无法有效捕捉复杂、非线性的模式。
- **对异常值的敏感性：** 像线性回归一样，它可能对特征空间中的异常值敏感。
- **在非线性问题上的表现：** 当决策边界高度非线性时，与更复杂的模型（如带核的 SVM 或基于树的方法）相比，它的表现可能不佳。

逻辑回归是一种强大而简单的分类算法，构成了许多机器学习 (machine learning)应用的根本。了解其工作原理，包括 Sigmoid 函数、决策边界和损失函数 (loss function)，在转向 K-近邻和支持向量 (vector)机等其他分类技术之前，提供了扎实的知识储备，我们将在接下来讨论这些技术。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 119-122 (Chapter 4, Section 4.4); DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  这本经典教材为逻辑回归提供了全面的统计学基础，涵盖了S型函数、最大似然估计（引出了对数损失）、决策边界及其作为广义线性模型的角色。
- [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-and-machine-learning/) — Christopher M. Bishop (2006)
  Publisher: Springer; Pages: 205-212 (Chapter 4, Section 4.3.2); DOI: [10.1007/978-0-387-44958-3](https://doi.org/10.1007/978-0-387-44958-3)
  一本备受推崇的教材，从概率角度探讨了逻辑回归，详细阐述了似然函数、交叉熵误差的推导和优化方法。
- [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) — Scikit-learn developers (2023)
  Scikit-learn官方文档，详细介绍了其逻辑回归类的参数（如C、penalty、solver）、`predict_proba`等方法，以及多分类问题的处理。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQEqtvnaaGjJTTc3e7IquRFXWIOFa0ld_lTCg0fQL78KYoCwPaQNiws-z0A0-KTqVPEyjY28UOmryhkCtEBe7G7h7n2g30Lb3oss8uAfl4Jh0Ji6ciPZ6TWcZvg8yOHmIulhX9046495GMmiaJNnc6h-H69_Db1BT0CEd6NzF1NZabxyqUxI26uE8KzyNVgbf3MdMd7-r2Yg0YEt_1E=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 153-172 (Chapter 4, Training Linear Models)
  一本实用指南，将逻辑回归的理论概念与Scikit-learn中的实现有效结合，包括正则化和超参数调优的解释。
