---
course: "mastering-gradient-boosting-algorithms"
chapter: "boosting-advanced-topics-customization"
lesson: "practice-custom-objectives-shap"
sourceId: 2047
sourceUrl: "https://apxml.com/zh/courses/mastering-gradient-boosting-algorithms/chapter-7-boosting-advanced-topics-customization/practice-custom-objectives-shap"
title: "实践：自定义目标与 SHAP"
description: "动手实现自定义损失函数与 SHAP 解释。"
order: 8
plots: ["plots/2047-0.json"]
sourceHash: "8cd78f125f3015fd19eabfba1da8eb9b432f9428e175d9ba5b0da8c6efbc383d"
sourceCorrections: []
---

本实践练习着重于将梯度提升模型适配特定需求并了解其内部机制。它展示了如何在 XGBoost 等提升框架中实现自定义目标函数，然后使用 SHAP 来分析所得模型的表现，并与使用标准目标训练的模型进行比较。

### 场景：非对称成本回归

设想一个情景，其中低估某个值的成本远高于高估它的成本。例如，低估产品需求可能导致销售损失和客户不满意（高成本），而高估则可能导致库存过剩（较低成本）。标准均方误差（MSE）对两种误差同等对待。我们可以定义一个自定义目标函数，以更严厉地惩罚低估。

### 定义自定义非对称平方误差目标

我们的目标是创建一个损失函数 (loss function)，当预测值 ($\hat{y}$) 小于真实值 ($y$) 时，应用更高的权重 (weight)。

让我们将单个预测的损失定义为：


$$
L(y, \hat{y}) =
\begin{cases}
\alpha (y - \hat{y})^2 & \text{如果 } y > \hat{y} \quad \text{(低估)} \\
(y - \hat{y})^2 & \text{如果 } y \le \hat{y} \quad \text{(高估或精确)}
\end{cases}
$$


这里，$\alpha > 1$ 是我们更严厉地惩罚低估的因子。

为了在 XGBoost 或 LightGBM 中使用此函数，我们需要损失函数对预测值 $\hat{y}$ 的一阶导数（梯度，$g$）和二阶导数（海森矩阵，$h$）。

**梯度 ($g$):**


$$
g = \frac{\partial L}{\partial \hat{y}} =
\begin{cases}
\frac{\partial}{\partial \hat{y}} [\alpha (y - \hat{y})^2] = \alpha \cdot 2 (y - \hat{y}) \cdot (-1) = -2 \alpha (y - \hat{y}) & \text{如果 } y > \hat{y} \\
\frac{\partial}{\partial \hat{y}} [(y - \hat{y})^2] = 2 (y - \hat{y}) \cdot (-1) = -2 (y - \hat{y}) & \text{如果 } y \le \hat{y}
\end{cases}
$$


我们可以将此简化为 $g = -2 w (y - \hat{y})$，其中当 $y > \hat{y}$ 时 $w = \alpha$，当 $y \le \hat{y}$ 时 $w = 1$。

**海森矩阵 ($h$):**


$$
h = \frac{\partial^2 L}{\partial \hat{y}^2} = \frac{\partial g}{\partial \hat{y}} =
\begin{cases}
\frac{\partial}{\partial \hat{y}} [-2 \alpha (y - \hat{y})] = -2 \alpha (-1) = 2 \alpha & \text{如果 } y > \hat{y} \\
\frac{\partial}{\partial \hat{y}} [-2 (y - \hat{y})] = -2 (-1) = 2 & \text{如果 } y \le \hat{y}
\end{cases}
$$


同样地，$h = 2 w$，其中 $w$ 的定义如上。

### 在 Python 中实现自定义目标

我们现在可以编写一个 Python 函数，用于计算梯度和海森矩阵，适用于 XGBoost 的 API。

```python
import numpy as np

def asymmetric_mse_objective(alpha):
    """
    非对称 MSE 的自定义目标函数。
    对低估（y_true > y_pred）施加 alpha 倍的惩罚。
    """
    def objective_function(y_true, y_pred):
        """
        计算非对称 MSE 的梯度和海森矩阵。
        """
        # 确保输入是 numpy 数组
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        residual = y_true - y_pred
        grad_weight = np.where(residual > 0, alpha, 1.0)
        hess_weight = np.where(residual > 0, alpha, 1.0)

        # 梯度: -2 * 权重 * (y_true - y_pred)
        grad = -2.0 * grad_weight * residual

        # 海森矩阵: 2 * 权重
        hess = 2.0 * hess_weight

        return grad, hess
    return objective_function

# 示例用法：创建一个目标函数，其中低估的成本是高估的 3 倍
custom_objective = asymmetric_mse_objective(alpha=3.0)
```

### 训练模型：标准目标与自定义目标对比

让我们生成一些合成数据，并训练两个 XGBoost 模型：一个使用标准 `reg:squarederror` 目标，另一个使用我们自定义的 `asymmetric_mse_objective`。

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import pandas as pd

# 生成合成回归数据
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])

# 为目标添加一些非线性
y = y + 5 * np.sin(X['feature_0'])**2 + np.random.normal(0, 10, size=y.shape[0])
y = np.maximum(0, y) # 确保目标值为非负，以符合实际情况

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# --- 模型 1：标准 MSE 目标 ---
xgb_std = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_std.fit(X_train, y_train)
y_pred_std = xgb_std.predict(X_test)
mse_std = mean_squared_error(y_test, y_pred_std)
print(f"标准模型 MSE: {mse_std:.4f}")

# --- 模型 2：自定义非对称 MSE 目标 ---
# 注意：XGBoost 会最小化目标函数。我们的定义可以直接使用。
xgb_custom = xgb.XGBRegressor(
    # 传递自定义目标函数
    objective=custom_objective, # 使用之前创建的函数 (alpha=3.0)
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
# 确保 y_train 是 NumPy 数组或与目标函数兼容的类型
xgb_custom.fit(X_train, y_train)
y_pred_custom = xgb_custom.predict(X_test)
mse_custom = mean_squared_error(y_test, y_pred_custom)
print(f"自定义目标模型 MSE: {mse_custom:.4f}") # MSE 可能更高，但低估的预测应该更少

# 分析预测误差
errors_std = y_test - y_pred_std
errors_custom = y_test - y_pred_custom

underprediction_cost_std = np.sum(np.maximum(0, errors_std)**2 * 3.0) + np.sum(np.maximum(0, -errors_std)**2)
underprediction_cost_custom = np.sum(np.maximum(0, errors_custom)**2 * 3.0) + np.sum(np.maximum(0, -errors_custom)**2)

print(f"标准模型非对称成本: {underprediction_cost_std:.2f}")
print(f"自定义模型非对称成本: {underprediction_cost_custom:.2f}")
```

你应该会看到，虽然 `xgb_std` 模型的标准 MSE 可能略低，但 `xgb_custom` 模型可能产生更低的*非对称成本*，表明它成功地按照我们自定义目标的目的，减少了低估带来的影响。

### 使用 SHAP 解释模型

现在，让我们使用 SHAP 来了解自定义目标如何影响模型的预测和特征贡献。

```python
import shap
import matplotlib.pyplot as plt

# 解释标准模型
explainer_std = shap.TreeExplainer(xgb_std)
shap_values_std = explainer_std.shap_values(X_test)

# 解释自定义目标模型
explainer_custom = shap.TreeExplainer(xgb_custom)
shap_values_custom = explainer_custom.shap_values(X_test)

# --- 比较全局特征重要性（汇总图） ---
print("\nSHAP 汇总图（标准模型）：")
shap.summary_plot(shap_values_std, X_test, show=False)
plt.title("SHAP 特征重要性（标准 MSE）")
plt.show()

print("\nSHAP 汇总图（自定义目标模型）：")
shap.summary_plot(shap_values_custom, X_test, show=False)
plt.title("SHAP 特征重要性（非对称 MSE，alpha=3）")
plt.show()
```

观察 SHAP 汇总图。两个模型之间特征重要性的排名或大小是否有所改变？有时，针对不同目标进行优化可以细微地改变模型对各种特征的依赖方式。

让我们查看特定特征的依赖图。

```python
# --- 比较特定特征（例如 feature_0）的依赖图 ---
print("\nSHAP feature_0 依赖图（标准模型）：")
shap.dependence_plot("feature_0", shap_values_std, X_test, interaction_index=None, show=False)
plt.title("依赖图：feature_0（标准 MSE）")
plt.show()

print("\nSHAP feature_0 依赖图（自定义目标模型）：")
shap.dependence_plot("feature_0", shap_values_custom, X_test, interaction_index=None, show=False)
plt.title("依赖图：feature_0（非对称 MSE，alpha=3）")
plt.show()
```

依赖图显示了当单个特征值变化时，预测结果如何变化，其中考虑了平均交互作用。比较这些图：自定义目标模型是否显示 `feature_0` 与预测之间有不同的关系，例如在某些范围将预测值向上移动以避免低估？

最后，让我们检查单个预测。

```python
# --- 比较单个预测（力图） ---
# 选择一个实例，例如第一个测试实例
instance_index = 0

print(f"\n分析实例 {instance_index}：")
print(f"  实际值: {y_test.iloc[instance_index]:.2f}")
print(f"  标准模型预测: {y_pred_std[instance_index]:.2f}")
print(f"  自定义模型预测: {y_pred_custom[instance_index]:.2f}")

print("\n力图（标准模型）：")
shap.force_plot(explainer_std.expected_value, shap_values_std[instance_index,:], X_test.iloc[instance_index,:], matplotlib=True, show=False)
plt.title(f"力图实例 {instance_index}（标准）")
# 如果图表拥挤，可能需要调整布局
# plt.tight_layout()
plt.show()

print("\n力图（自定义模型）：")
shap.force_plot(explainer_custom.expected_value, shap_values_custom[instance_index,:], X_test.iloc[instance_index,:], matplotlib=True, show=False)
plt.title(f"力图实例 {instance_index}（自定义）")
# plt.tight_layout()
plt.show()
```

比较同一实例的力图。基准值（期望值）可能略有不同。更重要的是，观察特征贡献（红色表示推高，蓝色表示推低）在模型之间有何不同。如果标准模型低估了该实例，自定义模型是否显示特征将预测推高了？

### 预测误差可视化

比较实际值与预测值的散点图可以视觉上突出非对称目标的效果。我们期望自定义模型在 y=x 线下方显著更少的点（低估）。

````python
# Using Plotly for interactive visualization
import plotly.graph_objects as go

fig = go.Figure()

# Scatter for Standard Model Predictions
fig.add_trace(go.Scatter(
    x=y_test, y=y_pred_std, mode='markers', name='标准 MSE',
    marker=dict(color='#339af0', opacity=0.6) # Blue
))

# Scatter for Custom Model Predictions
fig.add_trace(go.Scatter(
    x=y_test, y=y_pred_custom, mode='markers', name='非对称 MSE (alpha=3)',
    marker=dict(color='#f76707', opacity=0.6) # Orange
))

# Add y=x line for reference
fig.add_trace(go.Scatter(
    x=[min(y_test.min(), y_pred_std.min(), y_pred_custom.min()), max(y_test.max(), y_pred_std.max(), y_pred_custom.max())],
    y=[min(y_test.min(), y_pred_std.min(), y_pred_custom.min()), max(y_test.max(), y_pred_std.max(), y_pred_custom.max())],
    mode='lines', name='实际值 = 预测值',
    line=dict(color='#495057', dash='dash') # Gray
))

fig.update_layout(
    title='实际值与预测值对比：标准目标与自定义目标',
    xaxis_title='实际值',
    yaxis_title='预测值',
    legend_title='模型',
    hovermode='closest'
)

# Show Plotly chart JSON
print("```plotly")
print(fig.to_json(pretty=False))
print("```")
````



![实际值与预测值对比：标准目标与自定义目标](plots/2047-0.json)



> 使用标准 MSE（蓝色）和自定义非对称 MSE（橙色）训练的模型的预测值与实际值对比。非对称模型的预测值相对于标准模型有向上偏移的趋势，减少了严重的低估（虚线下方远处的点）。

### 结论

本练习展示了在 XGBoost 中定义、实现和评估自定义目标函数的过程。通过更严厉地惩罚低估，我们影响了模型做出与我们非对称成本结构更一致的预测。使用 SHAP，我们获得了关于自定义目标如何改变模型行为的认识，包括全局（特征重要性）和局部（单个预测）两方面。将自定义目标与 SHAP 等可解释性工具结合，使得构建的模型不仅能针对特定要求进行优化，而且可理解且值得信任。

## 参考资料

- [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785) — Tianqi Chen, Carlos Guestrin (2016)
  Journal: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining; Publisher: ACM; Pages: 785–794; DOI: [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785)
  介绍XGBoost的原始论文，详细阐述了其算法，包括使用一阶和二阶导数自定义目标函数的数学公式。
- [Custom Objective and Evaluation Metric](https://xgboost.readthedocs.io/en/stable/tutorials/custom_objective.html) — XGBoost Contributors (2023)
  Publisher: XGBoost Project
  XGBoost框架中实现自定义目标函数的官方指南，提供了实用的代码示例。
- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) — Scott M. Lundberg and Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems 30; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4768-4777; DOI: [10.48550/arXiv.1705.07874](https://doi.org/10.48550/arXiv.1705.07874)
  介绍了SHAP（SHapley Additive exPlanations），这是一种基于博弈论的方法，用于解释任何机器学习模型的预测结果。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  涵盖机器学习算法理论背景的教科书，包括梯度提升和损失函数的作用。
