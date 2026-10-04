---
course: "getting-started-with-scikit-learn"
chapter: "supervised-learning-regression"
lesson: "regression-evaluation-metrics"
sourceId: 1848
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-scikit-learn/chapter-2-supervised-learning-regression/regression-evaluation-metrics"
title: "回归评估指标"
description: "了解评估回归模型的常用指标：MAE、MSE、RMSE和R平方。"
order: 5
plots: []
sourceHash: "7f9eb6a06130a0d539bbc2ac0a566c4105f09a64a38b35223a8938e8bc80d9a4"
sourceCorrections: []
---

使用Scikit-learn训练线性回归模型时，检查像`model.coef_`和`model.intercept_`这样的系数可以提供初步信息。然而，一个基本问题出现了：模型的表现到底有多准确？仅仅拟合一个模型是不够的；客观地衡量其表现必不可少。这就是评估指标的作用。

对于回归任务，我们预测连续值，评估指标衡量了我们模型的预测值($\hat{y}$)与实际目标值($y$)有多接近。实际值与预测值之间的差($y_i - \hat{y}_i$)被称为该数据点的*残差*或*误差*。回归指标通常基于汇总这些残差，覆盖整个数据集（或特定子集，如测试集）。

我们来看看一些最常用的回归模型评估指标。

### 平均绝对误差 (MAE)

平均绝对误差 (MAE) 是最简单、最直观的指标之一。它衡量了预测值与实际值之间的平均绝对差。

MAE的公式是：
$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
这里：

- $n$ 是数据点的数量。
- $y_i$ 是第 $i$ 个数据点的实际值。
- $\hat{y}_i$ 是第 $i$ 个数据点的预测值。
- $|...|$ 表示绝对值。

**解释：**

- **单位：** MAE与目标变量的单位相同（例如，如果您预测以美元计的房价，MAE也将以美元为单位）。
- **范围：** MAE的范围从0到$\infty$。值越接近0表示模型表现越好，这意味着平均预测误差小。
- **对异常值的敏感性：** 因为它使用绝对差，与对误差进行平方的指标相比，MAE对大误差（异常值）的敏感性较低。它根据误差的大小线性地处理所有误差。

MAE给出了一个直观的平均误差大小。如果您的模型MAE为10，这意味着平均而言，其预测值与实际值相差10个单位。

### 均方误差 (MSE)

均方误差 (MSE) 是另一个广泛使用的指标。它不使用绝对差，而是计算预测值与实际值之间*平方*差的平均值。

MSE的公式是：
$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$

**解释：**

- **单位：** MSE的单位是目标变量单位的*平方*（例如，房价的美元平方）。这使得它在原始问题的背景下更难直接解释。
- **范围：** MSE的范围从0到$\infty$，值越接近0越好。
- **对异常值的敏感性：** 对误差进行平方会给更大的误差赋予更高的权重 (weight)。一个相差10个单位的预测会对总和贡献100，而一个相差2个单位的预测只贡献4。这意味着MSE对犯大错的模型给予重罚。
- **数学特性：** 平方项使得MSE可微分，这对用于训练某些模型的特定数学优化技术有利。

虽然由于其平方单位而直观性较差，但如果那些误差代价特别高或非常重要需要避免，那么MSE对大误差的敏感性可能是有益的。

### 均方根误差 (RMSE)

均方根误差 (RMSE) 简单来说就是MSE的平方根。它解决了MSE主要的解释问题，将指标恢复到目标变量的原始单位。

RMSE的公式是：
$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

**解释：**

- **单位：** RMSE与目标变量的单位相同，类似于MAE。
- **范围：** RMSE的范围从0到$\infty$，值越低表示拟合越好。
- **对异常值的敏感性：** 像MSE一样，RMSE对大误差的惩罚比MAE更重，因为在开方之前进行了平方运算。然而，其解释通常比MSE更容易，因为单位与目标变量匹配。

RMSE可以说是最常用的回归指标。它提供了误差典型大小的衡量标准，以原始单位表示，同时对较大的偏差仍然敏感。如果您的RMSE为15，这表明模型的预测值通常与实际值相差约15个单位，并且与MAE相比，更大的误差对最终指标产生了不成比例的更大影响。

### R平方 ($R^2$) 分数 (决定系数)

与之前衡量误差大小的指标不同，R平方 ($R^2$) 分数，也称为决定系数，衡量了模型特征解释的目标变量方差的比例。

它将模型的表现（使用平方误差和SSE，即 $n \times \text{MSE}$）与一个只预测所有输入的目标变量均值($\bar{y}$)的基准模型（其误差是总平方和SST）进行比较。

$R^2$ 的公式是：
$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2} = 1 - \frac{\text{SSE}}{\text{SST}}$
这里 $\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$ 是实际值的平均值。

**解释：**

- **单位：** $R^2$ 是一个无单位的指标，表示一个比例。
- **范围：**
  - $R^2$ 为1表示模型完美预测了目标变量（解释了100%的方差）。
  - $R^2$ 为0表示模型的表现不优于始终预测目标值平均值的基准模型。
  - $R^2$ 分数可以是负数。当模型表现*差于*基准平均值模型时，就会发生这种情况，这意味着与仅仅猜测平均值相比，模型的预测甚至不如简单地猜测平均值。
- **背景：** $R^2$ 提供了一个相对拟合度衡量标准。“好”的 $R^2$ 值取决于具体背景。在一些领域（如物理学），期望有较高的 $R^2$ 值（例如 > 0.95）。而在其他领域（如社会科学），$R^2$ 为0.3可能就被认为是有用的。它不告诉您平均误差大小，只表明您的模型相对于总方差，比简单地预测平均值好多少。

### 选择合适的指标

评估指标的选择取决于您的特定目标和数据的特点：

- 如果您想要一个在原始单位下易于解释的指标，并且异常值不是主要问题或不应受到不成比例的惩罚，请使用 **MAE**。
- 如果您想要一个在原始单位下的指标，但需要对大误差施加更显著的惩罚，请使用 **RMSE**。它通常是一个不错的默认选择。
- **MSE** 主要在其数学特性有利时使用（由于单位解释困难，在最终评估中较不常用）。
- 当您需要了解模型解释的方差比例，并提供相对的拟合优度衡量时，请使用 **$R^2$**。它对于比较模型的解释能力很有用，但不直接表示典型的预测误差大小。

通常，查看多个指标是有益的，以更全面地了解模型的表现。例如，高 $R^2$ 表示您的模型很好地捕捉了趋势，而低MAE或RMSE证实了典型的预测误差很小。

在下一节中，我们将看到如何使用Scikit-learn提供的函数轻松计算这些指标。

## 参考资料

- [sklearn.metrics: Metrics, distances, and kernel approximations](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — The scikit-learn developers (2023)
  提供Scikit-learn库中回归评估指标的官方文档，包括示例和用法。
- [An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Apurva Gundecha, Shani Salhotra, Eunseob Lee (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38914-1](https://doi.org/10.1007/978-3-031-38914-1)
  一本现代教材，解释统计学习概念，详细讨论线性回归和各种模型评估指标，并附有Python示例。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本高级教材，涵盖统计学习方法的理论基础，全面介绍回归分析和模型评估技术。
