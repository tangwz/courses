# 使用 Scikit-learn 实现 GBM

来源：[原文](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/implementing-sklearn-gbm)

[返回章节目录](README.md) · [返回课程目录](../README.md)

梯度提升机 (GBM) 算法的理论基础包括函数梯度下降 (gradient descent)、损失函数 (loss function)、收缩和子抽样。它在实践中可使用 Python 主要机器学习 (machine learning)库之一 Scikit-learn 来有效实现。Scikit-learn 通过其 `GradientBoostingRegressor` 和 `GradientBoostingClassifier` 类别提供了良好集成的实现，在熟悉的 API 中封装了 GBM 的核心逻辑。

### Scikit-learn 的梯度提升估计器

Scikit-learn 提供了两个主要类别用于梯度提升：

- `GradientBoostingRegressor`: 用于回归任务。
- `GradientBoostingClassifier`: 用于分类任务（二元和多类别）。

这些类别遵循标准的 Scikit-learn 估计器 API，这意味着它们拥有 `fit`、`predict` 和 `predict_proba`（用于分类器）等方法，以及其他辅助功能。这种一致性简化了它们在现有机器学习 (machine learning)流程中的集成。

### 理论与参数 (parameter)的对应关系

本章前面讨论的理论构成部分直接对应于这些 Scikit-learn 类别中的超参数 (hyperparameter)。了解这种对应关系对于有效的模型配置来说是必要的：

- **损失函数 (loss function) (`loss`)**: 此参数确定要优化的损失函数。
  - 对于 `GradientBoostingRegressor`：常见选项包括 `'ls'`（最小二乘回归，相当于平方误差）、`'lad'`（最小绝对偏差）、`'huber'`（LS 和 LAD 的组合）和 `'quantile'`（用于分位数回归）。默认值为 `'ls'`。
  - 对于 `GradientBoostingClassifier`：主要选项为 `'deviance'`（对数损失，适用于概率估计，用于二元和多类别分类）和 `'exponential'`（这实际上重现了 AdaBoost 算法）。默认值为 `'deviance'`。
- **估计器数量 (`n_estimators`)**: 这控制了要构建的提升阶段或序列树的数量（在我们之前的符号中为 $M$）。数量越多通常会导致模型越复杂，如果与其他正则化 (regularization)技术不平衡，可能会过拟合 (overfitting)。默认值为 100。
- **学习率 (`learning_rate`)**: 这对应于收缩参数 ($\nu$)。它调整每棵树的贡献。较低的值在达到类似性能时需要更多的估计器，但通常会提高泛化能力。它通过减小单棵树的影响来作为一种正则化技术。默认值为 0.1。存在一个典型的权衡：较小的 `learning_rate` 通常需要较大的 `n_estimators`。
- **子抽样 (`subsample`)**: 此参数通过指定用于拟合每个独立基础学习器（树）的样本比例来启用随机梯度提升。如果小于 1.0，它会引入随机性，降低方差，并可以提高泛化能力，通常会略微增加偏差。它还能加快计算速度。默认值为 1.0（无子抽样）。
- **树特有参数**: 由于 GBM 使用决策树作为基础学习器，您可以控制其结构进行正则化：
  - `max_depth`: 单个回归估计器的最大深度。限制深度可限制模型复杂度。默认值为 3。
  - `min_samples_split`: 分割内部节点所需的最小样本数。默认值为 2。
  - `min_samples_leaf`: 叶节点所需的最小样本数。默认值为 1。
  - `max_features`: 在寻找最佳分割时要考虑的特征数量/比例。引入了类似于随机森林的随机性，并提供了列子抽样。默认值为 `None`（考虑所有特征）。
- **初始化 (`init`)**: 允许为初始预测 $F_0(x)$ 指定一个初始估计器。默认情况下，它使用基于训练数据的简单估计器（例如，回归的均值，分类的对数几率）。

### 实现 GBM 回归器

让我们演示如何使用 `GradientBoostingRegressor`。我们将使用一个简单的合成数据集。

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go

# 1. 生成合成数据
rng = np.random.RandomState(0)
X = rng.rand(100, 1) * 10
y = np.sin(X).ravel() + rng.normal(0, 0.5, X.shape[0]) # 带有噪声的目标

# 2. 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 初始化并训练 GBM 回归器
gbr = GradientBoostingRegressor(
    n_estimators=100,      # 树的数量
    learning_rate=0.1,     # 收缩
    max_depth=3,           # 每棵树的最大深度
    subsample=0.8,         # 每棵树使用 80% 的数据
    loss='ls',             # 最小二乘损失
    random_state=42
)

gbr.fit(X_train, y_train)

# 4. 进行预测
y_pred = gbr.predict(X_test)

# 5. 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f"测试均方误差: {mse:.4f}")

# 创建排序后的 X_test 以便更平滑地绘图
X_test_sorted_indices = np.argsort(X_test.ravel())
X_test_sorted = X_test[X_test_sorted_indices]
y_test_sorted = y_test[X_test_sorted_indices]
y_pred_sorted = gbr.predict(X_test_sorted) # 在排序后的 X_test 上进行预测

# 6. 可视化结果（可选）
fig = go.Figure()
fig.add_trace(go.Scatter(x=X_train.ravel(), y=y_train, mode='markers', name='训练数据', marker=dict(color='#a5d8ff', size=8)))
fig.add_trace(go.Scatter(x=X_test_sorted.ravel(), y=y_test_sorted, mode='markers', name='测试数据（实际）', marker=dict(color='#ffc9c9', size=8)))
fig.add_trace(go.Scatter(x=X_test_sorted.ravel(), y=y_pred_sorted, mode='lines', name='GBM 预测', line=dict(color='#f03e3e', width=2)))

fig.update_layout(
    title='梯度提升回归器拟合',
    xaxis_title='特征 X',
    yaxis_title='目标 y',
    legend_title='数据',
    template='plotly_white',
    width=700,
    height=400
)
# 在 Jupyter 等环境中显示图表：fig.show()
# 转换为 JSON 以便在网页中显示（如果需要）：
# print(fig.to_json())
```



[交互图表：梯度提升回归器拟合](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/implementing-sklearn-gbm#plot-17tq8if)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "梯度提升回归器拟合"
    },
    "xaxis": {
      "title": {
        "text": "特征 X"
      }
    },
    "yaxis": {
      "title": {
        "text": "目标 y"
      }
    },
    "legend": {
      "title": {
        "text": "数据"
      }
    },
    "template": "plotly_white",
    "width": 700,
    "height": 400,
    "shapes": [],
    "annotations": [],
    "images": []
  },
  "data": [
    {
      "type": "scatter",
      "x": [
        9.50714306,
        7.31993942,
        5.98658484,
        1.5601864,
        1.5599452,
        0.58083612,
        8.66176146,
        6.01115012,
        7.08072578,
        0.20584494,
        9.69909852,
        8.32442641,
        2.12339111,
        1.81824967,
        1.8340451,
        3.04242243,
        5.24756432,
        4.31945019,
        2.9122914,
        6.11852895,
        1.39493861,
        7.78156751,
        8.60015705,
        4.93182761,
        9.41002685,
        5.14234438,
        4.33701175,
        0.8541496,
        2.50455188,
        4.83034269,
        9.85559796,
        5.19485115,
        6.12894532,
        1.2062868,
        3.80379443,
        3.96840149,
        5.8119254,
        2.08876748,
        0.94988801,
        7.4916576,
        7.95662476,
        3.68916694,
        1.84848969,
        9.3719915,
        8.09111666,
        1.45312536,
        9.20161087,
        6.36981675,
        0.95562617,
        2.4053381,
        8.46706969,
        3.72010145,
        6.30427923,
        2.14969973,
        5.97709935,
        2.1321059,
        4.81603096,
        7.07921996,
        6.84233027,
        0.53316528,
        7.91725038,
        9.96569932,
        3.47961916,
        1.97316968,
        1.76824173,
        3.02332674,
        9.61715905,
        6.57663794,
        8.20993226,
        7.2904971
      ],
      "y": [
        -0.22573727,
        -0.29882757,
        -0.28241782,
        1.5968085,
        0.7969219,
        0.54004673,
        -0.67793312,
        -0.20542986,
        0.15615468,
        0.02179867,
        -0.30159877,
        0.40695378,
        0.78324431,
        0.97059612,
        1.63095939,
        0.02690656,
        -0.88746359,
        -0.78223537,
        0.19646415,
        -0.0367384,
        0.93812857,
        0.17865225,
        -0.24349891,
        -0.95446461,
        -0.34025506,
        -0.67708831,
        -0.82480078,
        1.18607754,
        0.33621094,
        -0.98202184,
        -0.34491507,
        -0.90715297,
        -0.18533261,
        0.88719761,
        -0.55129095,
        -0.51274511,
        -0.50770306,
        1.04653605,
        0.76939997,
        -0.15964327,
        0.36762509,
        -0.64182211,
        1.63936637,
        -0.40137446,
        0.10726755,
        1.01185844,
        -0.39905272,
        -0.02097956,
        0.64171331,
        0.73487457,
        0.03069402,
        -0.57714571,
        0.01583104,
        0.9650141,
        -0.36386301,
        0.9750833,
        -1.16004055,
        0.12830288,
        0.49956526,
        0.61287197,
        0.25583644,
        -0.21738712,
        -0.35960213,
        1.00214899,
        1.28893082,
        0.04277157,
        -0.30516669,
        -0.20068934,
        0.1632535,
        -0.03163679
      ],
      "mode": "markers",
      "marker": {
        "color": "#a5d8ff",
        "size": 8
      },
      "name": "训练数据"
    },
    {
      "type": "scatter",
      "x": [
        0.80831981,
        2.65389416,
        2.72441592,
        3.11795893,
        3.6172176,
        3.83441519,
        4.0759469,
        4.44931107,
        4.49537461,
        5.0733336,
        5.15994008,
        5.36316317,
        5.48447936,
        5.6772502,
        5.86895096,
        6.07544852,
        6.37776661,
        6.94368762,
        7.1039056,
        7.12077893,
        7.18058481,
        7.1829079,
        7.27012156,
        7.71270331,
        7.88723351,
        8.05185208,
        8.83289166,
        9.0932043,
        9.18645815,
        9.97555493
      ],
      "y": [
        0.87704881,
        0.1934718,
        0.38239352,
        0.0743875,
        -0.13280356,
        -0.63039888,
        -0.8188885,
        -0.8852083,
        -1.01074229,
        -0.89694204,
        -1.07913596,
        -0.82024838,
        -0.8311913,
        -0.62528818,
        -0.40625302,
        -0.3141473,
        -0.12004357,
        0.50829503,
        0.6324774,
        0.63353484,
        0.54594957,
        0.47140218,
        0.00404736,
        0.34498013,
        0.24743373,
        0.08537333,
        -0.48415074,
        -0.57446324,
        -0.38930668,
        -0.11201503
      ],
      "mode": "markers",
      "marker": {
        "color": "#ffc9c9",
        "size": 8
      },
      "name": "测试数据（实际）"
    },
    {
      "type": "scatter",
      "x": [
        0.80831981,
        2.65389416,
        2.72441592,
        3.11795893,
        3.6172176,
        3.83441519,
        4.0759469,
        4.44931107,
        4.49537461,
        5.0733336,
        5.15994008,
        5.36316317,
        5.48447936,
        5.6772502,
        5.86895096,
        6.07544852,
        6.37776661,
        6.94368762,
        7.1039056,
        7.12077893,
        7.18058481,
        7.1829079,
        7.27012156,
        7.71270331,
        7.88723351,
        8.05185208,
        8.83289166,
        9.0932043,
        9.18645815,
        9.97555493
      ],
      "y": [
        0.81077412,
        0.46663044,
        0.33075875,
        0.08680574,
        -0.44157432,
        -0.58149643,
        -0.7002805,
        -0.86340934,
        -0.86340934,
        -0.89695514,
        -0.89695514,
        -0.76729925,
        -0.76729925,
        -0.62852131,
        -0.40623742,
        -0.20071312,
        -0.03685368,
        0.39235828,
        0.4229661,
        0.4229661,
        0.4229661,
        0.4229661,
        0.21944514,
        0.21944514,
        0.21944514,
        0.08100831,
        -0.44264076,
        -0.44264076,
        -0.44264076,
        -0.12895893
      ],
      "mode": "lines",
      "line": {
        "color": "#f03e3e",
        "width": 2
      },
      "name": "GBM 预测"
    }
  ]
}
```

</details>



> 训练好的梯度提升回归器模型的预测结果与实际测试数据点和原始训练数据进行比较。

### 实现 GBM 分类器

分类过程类似，使用 `GradientBoostingClassifier`。

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss
import plotly.graph_objects as go
import pandas as pd

# 1. 生成合成分类数据
X, y = make_classification(n_samples=200, n_features=2, n_informative=2, n_redundant=0,
                           n_clusters_per_class=1, random_state=42, class_sep=1.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 初始化并训练 GBM 分类器
gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=2,          # 较浅的树通常适用于分类
    subsample=0.8,
    loss='deviance',      # 用于概率输出的对数损失
    random_state=42
)

gbc.fit(X_train, y_train)

# 3. 进行预测
y_pred = gbc.predict(X_test)
y_pred_proba = gbc.predict_proba(X_test)[:, 1] # 正类的概率

# 4. 评估模型
accuracy = accuracy_score(y_test, y_pred)
logloss = log_loss(y_test, y_pred_proba)
print(f"测试准确度: {accuracy:.4f}")
print(f"测试对数损失: {logloss:.4f}")

# 5. 特征重要性
importances = gbc.feature_importances_
feature_names = [f'特征 {i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({'特征': feature_names, '重要性': importances})
importance_df = importance_df.sort_values(by='重要性', ascending=False)

print("\n特征重要性:")
print(importance_df)

# 创建特征重要性的 Plotly 条形图
fig_imp = go.Figure(go.Bar(
    x=importance_df['重要性'],
    y=importance_df['特征'],
    orientation='h',
    marker_color='#3bc9db'
))
fig_imp.update_layout(
    title='GBM 特征重要性',
    xaxis_title='重要性得分',
    yaxis_title='特征',
    yaxis={'categoryorder':'total ascending'}, # 将最重要的显示在顶部
    template='plotly_white',
    width=600,
    height=300
)
# 显示图表：fig_imp.show()
# print(fig_imp.to_json())
```



[交互图表：GBM 特征重要性](https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-2-gradient-boosting-algorithm-depth/implementing-sklearn-gbm#plot-na3hl0)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": {
      "text": "GBM 特征重要性"
    },
    "xaxis": {
      "title": {
        "text": "重要性得分"
      }
    },
    "yaxis": {
      "title": {
        "text": "特征"
      },
      "categoryorder": "total ascending"
    },
    "template": "plotly_white",
    "width": 600,
    "height": 300,
    "shapes": [],
    "annotations": [],
    "images": []
  },
  "data": [
    {
      "type": "bar",
      "x": [
        0.60439907,
        0.39560093
      ],
      "y": [
        "特征 1",
        "特征 0"
      ],
      "orientation": "h",
      "marker": {
        "color": "#3bc9db"
      }
    }
  ]
}
```

</details>



> 训练好的梯度提升分类器得出的特征重要性，表明了每个特征根据杂质减少对模型预测的相对贡献。

### Scikit-learn GBM 中的特征重要性

如演示所示，训练好的 GBM 模型提供了 `feature_importances_` 属性。这些重要性通常是根据该特征在集成中所有树上的分割所带来的损失函数 (loss function)（或像 Friedman MSE 这样的杂质准则）的总减少量计算得出的，并根据受影响的样本数量进行加权。虽然这对于快速评估特征相关性有用，但请记住，这些重要性分数有时可能会产生误导，特别是在特征相关联或比较不同类型或尺度的特征时。后续章节将介绍更先进的解释方法，如 SHAP 值。

### 展望

Scikit-learn 的 `GradientBoostingRegressor` 和 `GradientBoostingClassifier` 为本章讨论的算法提供了扎实、基础的实现。它们是解决许多问题的出色工具，也是理解更复杂提升库的起步。然而，对于要求更高性能、速度优化或高级处理分类数据或缺失值等特殊功能的任务，XGBoost、LightGBM 和 CatBoost 等库（我们将在后续章节中研究）通常提供显著优势。掌握了 GBM 的核心机制及其 Scikit-learn 实现后，您已做好准备来理解这些专门库带来的改进。

## 参考资料

- [Greedy Function Approximation: A Gradient Boosting Machine](https://www.projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full) — Jerome H. Friedman (2001)
  Journal: The Annals of Statistics; Publisher: Institute of Mathematical Statistics; Volume: 29; Pages: 1189-1232; DOI: [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451)
  介绍梯度提升机算法及其理论框架的原始研究论文。
- [Stochastic Gradient Boosting](https://doi.org/10.1016/S0167-9473%2801%2900065-2) — Jerome H. Friedman (2002)
  Journal: Computational Statistics & Data Analysis; Publisher: Elsevier BV; Volume: 38(4); Pages: 367-378; DOI: [10.1016/S0167-9473(01)00065-2](https://doi.org/10.1016/S0167-9473%2801%2900065-2)
  介绍子采样（随机梯度提升）作为GBM正则化技术的论文，与`subsample`参数直接相关。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  一本经典教材，对梯度提升及相关集成方法提供了全面的统计学解释（第2版）。
- [sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) — Scikit-learn Developers (2024)
  Scikit-learn的GradientBoostingRegressor官方文档，详细说明其回归任务的参数和用法。
- [sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) — Scikit-learn Developers (2024)
  Scikit-learn的GradientBoostingClassifier官方文档，详细说明其分类任务的参数和用法。

---

[上一节](06-%E6%8A%BD%E6%A0%B7%E6%96%B9%E6%B3%95%20%28%E9%9A%8F%E6%9C%BA%E6%A2%AF%E5%BA%A6%E6%8F%90%E5%8D%87%29.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%9E%84%E5%BB%BA%E5%9F%BA%E7%A1%80GBM%E6%A8%A1%E5%9E%8B.md)
