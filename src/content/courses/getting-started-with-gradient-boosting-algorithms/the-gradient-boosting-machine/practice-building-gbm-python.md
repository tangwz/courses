---
course: "getting-started-with-gradient-boosting-algorithms"
chapter: "the-gradient-boosting-machine"
lesson: "practice-building-gbm-python"
sourceId: 7583
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-gradient-boosting-algorithms/chapter-2-the-gradient-boosting-machine/practice-building-gbm-python"
title: "动手实践：使用 Python 构建 GBM"
description: "一次实践编程环节，您将使用 Python 和 NumPy 从头实现一个简化的梯度提升机，以巩固您的认识。"
order: 6
plots: ["plots/7583-0.json"]
sourceHash: "44f3da729e30aa9e83d89bc2c847af4ad60b6b8312bd4d06cb6864c96141fd23"
sourceCorrections: []
---

理论提供地图，但只有亲手实践才能真正领会其奥秘。构建梯度提升机需要将其算法逻辑转化为可运行的 Python 代码。这项练习旨在巩固您对 GBM 如何迭代学习的认识。我们不会构建一个生产级别的库；相反，我们将为回归任务构建一个简化的 GBM，以观察其运行机制。

我们的弱学习器将是浅层决策树，具体来说是 Scikit-Learn 中的 `DecisionTreeRegressor`。通过专注于提升过程本身，您将准确地看到这些简单模型如何结合起来形成一个强大而准确的预测器。

### 环境设置

首先，我们来准备工作环境。我们需要 `numpy` 进行数值计算，以及 `matplotlib` 来可视化结果。最重要的是，我们将导入 `DecisionTreeRegressor` 作为我们的弱学习器。

我们将生成一个基于正弦波的简单非线性数据集。这为我们提供了一个清晰的目标函数，以便观察模型学习的效果。

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# 生成合成数据集
np.random.seed(42)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.random.normal(0, 0.2, 100)

# 绘制数据，了解其特点
plt.figure(figsize=(10, 6))
plt.scatter(X, y, c='#495057', s=20, label='数据点')
plt.plot(X, np.sin(X), color='#f03e3e', linewidth=2, label='真实函数 (sin(x))')
plt.title('合成回归数据集')
plt.xlabel('特征 (x)')
plt.ylabel('目标 (y)')
plt.legend()
plt.show()
```

### 从头开始构建梯度提升算法

我们将为回归任务实现 GBM 的主要逻辑，使用均方误差 (MSE) 作为损失函数 (loss function)。正如我们所学，MSE 损失函数 $L(y, F) = \frac{1}{2}(y - F)^2$ 的负梯度就是残差 $y - F$。

#### 步骤 1：模型初始化

第一步是创建一个初始预测。对于 MSE，最小化损失的最佳常数预测是目标变量的均值。这将是我们的起点 $F_0(x)$。

```python
# 初始预测是目标变量的均值
initial_prediction = np.mean(y)
```

#### 步骤 2：迭代构建树

现在我们进入算法的主循环。每次迭代，我们执行三个操作：

1. 计算伪残差（我们的下一棵树需要纠正的“误差”）。
2. 将一个弱学习器（一个浅层决策树）拟合到这些残差上。
3. 通过添加这棵新树的贡献，并按学习率进行缩放，来更新我们整体模型的预测。

让我们定义模型的超参数 (parameter) (hyperparameter)。

```python
# 超参数
n_estimators = 100
learning_rate = 0.1
max_depth = 1 # 浅层树是弱学习器

# 存储树和当前预测
trees = []
F = np.full(y.shape, initial_prediction) # F 代表我们集成模型的预测

for _ in range(n_estimators):
    # 1. 计算残差
    residuals = y - F

    # 2. 将弱学习器拟合到残差上
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    tree.fit(X, residuals)

    # 3. 更新集成模型的预测
    prediction_from_tree = tree.predict(X)
    F += learning_rate * prediction_from_tree

    # 存储训练好的树
    trees.append(tree)
```

在此循环中，`F` 代表集成模型在每个阶段的累积预测。请注意，每棵新树不是在 `y` 上训练，而是在 `residuals`（残差）上训练。它学习预测当前集成模型的误差，然后我们将它预测的一小部分加回到我们的主要预测 `F` 中。

### 进行预测

对新数据进行预测时，我们遵循相同的过程。我们从初始预测（均值）开始，然后按顺序添加来自集成模型中每棵树的缩放预测。

```python
def predict(X_new):
    # 从初始常数预测开始
    prediction = np.full(X_new.shape[0], initial_prediction)

    # 添加每棵树的预测
    for tree in trees:
        prediction += learning_rate * tree.predict(X_new)

    return prediction

# 在原始数据上生成预测，查看效果
y_pred = predict(X)
```

### 结果可视化

理解我们所构建模型的最佳方式是将其输出可视化。下面的图表展示了原始数据点、我们试图建模的真实函数、我们简单的初始预测，以及我们自定义 GBM 的最终、更精细的预测。



![梯度提升模型拟合效果](plots/7583-0.json)



> 模型从一个简单的平均值开始，然后迭代地改进其预测。每一步都修正上一步的误差，逐渐从有噪声的数据点中学习潜在的正弦模式。

如您所见，我们的模型从一条朴素的水平线变成了一条精细的曲线，紧密贴合真实函数。它通过将 100 棵非常简单的决策树（在本例中是决策桩）串联起来实现这一点，每棵树都修正了前一棵树遗留的误差。

您现在已经构建了一个梯度提升机。尽管像 Scikit-Learn 和 XGBoost 这样的库提供了高度优化、功能丰富的实现，其主要原理正是您刚刚编写的代码所体现的。这种实践经验为您打下了良好的基础，以便我们在下一章中继续学习如何使用和调整这些功能强大的预构建库。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://www.jstor.org/stable/2676766) — Jerome H. Friedman (2001)
  Journal: Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍梯度提升机算法的原始学术论文，为后续发展奠定了基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本内容详尽的教科书，提供了梯度提升和各种集成方法的统计与算法描述。
- [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html) — scikit-learn developers (2024)
  scikit-learn库中决策树回归器的官方文档，在本次实践中用作弱学习器。
