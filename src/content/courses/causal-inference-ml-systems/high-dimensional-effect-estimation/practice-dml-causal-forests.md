---
course: "causal-inference-ml-systems"
chapter: "high-dimensional-effect-estimation"
lesson: "practice-dml-causal-forests"
sourceId: 4367
sourceUrl: "https://apxml.com/zh/courses/causal-inference-ml-systems/chapter-3-high-dimensional-effect-estimation/practice-dml-causal-forests"
title: "实践：DML和因果森林的实现"
description: "双重机器学习和因果森林在效应估计中的动手实现。"
order: 7
plots: ["plots/4367-0.json"]
sourceHash: "b5b7c1ce6be3e765315fa205415325db0940be98506554d1a78d19d5409cc1b5"
sourceCorrections: []
---

双重机器学习 (machine learning) (DML) 和因果森林是在高维环境中估计效应的技术。使用常用Python库实现这些估计量，以使它们能够应用于实际数据集。我们将使用已知真实情况的模拟数据，从而可以验证实现的效果。

我们的目标是使用DML估计平均处理效应 (ATE)，并使用因果森林估计条件平均处理效应 (CATE)。

### 设置与数据模拟

首先，请确保您已安装必要的库。我们将主要使用 `econml`、`scikit-learn`、`pandas` 和 `numpy`。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from econml.dml import LinearDML, CausalForestDML
from econml.cate_interpreter import SingleTreeCateInterpreter
import matplotlib.pyplot as plt # 仅用于以下绘图示例
```

让我们模拟具有高维混杂因素 ($X$)、二元处理 ($T$) 和结果 ($Y$) 的数据。我们将设计模拟，使处理效应具有异质性，取决于某个混杂因素。

```python
# 模拟参数
n_samples = 5000  # 样本数量
n_features = 20   # 混杂因素数量
true_ate = 1.5    # 基本平均处理效应
heterogeneity_slope = 0.5 # 基于X0的异质性斜率

# 生成混杂因素
np.random.seed(42)
X = np.random.normal(0, 1, size=(n_samples, n_features))
X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(n_features)])

# 生成处理分配（倾向得分取决于X）
# 简单的倾向得分逻辑模型
propensity_coeffs = np.random.uniform(-0.5, 0.5, size=n_features)
propensity_logit = X @ propensity_coeffs + np.random.normal(0, 0.1, size=n_samples)
propensity = 1 / (1 + np.exp(-propensity_logit))
T = np.random.binomial(1, propensity, size=n_samples)

# 生成结果（取决于X、T并包含异质性）
# Y = X的线性效应 + 处理效应 * T + 噪声
outcome_coeffs = np.random.uniform(0, 1, size=n_features)
# 真实CATE = true_ate + heterogeneity_slope * X[:, 0]
true_cate = true_ate + heterogeneity_slope * X[:, 0]
Y = X @ outcome_coeffs + true_cate * T + np.random.normal(0, 0.5, size=n_samples)

print(f"模拟数据形状:")
print(f"X: {X.shape}, T: {T.shape}, Y: {Y.shape}")
print(f"（基于模拟的）真实平均处理效应: {np.mean(true_cate):.4f}")
```

此设置模拟了一个常见情况，即许多特征可能混杂处理-结果关系，并且处理的有效性根据个体特征（此处具体指 $X_0$）而异。

### 实现用于ATE的双重机器学习 (machine learning) (DML)

DML通过使用机器学习模型从处理 $T$ 和结果 $Y$ 中部分去除混杂因素 $X$ 的效应来估计ATE。这包括拟合两个辅助模型：

1. 结果模型：$E[Y | X]$
2. 处理模型：$E[T | X]$（如果T是二元，则为倾向得分模型）

然后，我们使用这些模型的残差来估计效应。`econml` 库简化了此过程。我们将使用 `LinearDML`，它假设最终阶段是线性的，用于估计恒定的ATE。

```python
# 定义辅助模型
# 对于结果模型 E[Y|X]
model_y = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
# 对于处理模型 E[T|X]（倾向得分）
model_t = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)

# 实例化LinearDML估计量
# 由于T是二元的，我们使用discrete_treatment=True
dml_estimator = LinearDML(model_y=model_y,
                          model_t=model_t,
                          discrete_treatment=True,
                          random_state=123)

# 拟合估计量
# 我们提供结果Y、处理T、混杂因素X，以及可选的W（此处无效应修正因子）
dml_estimator.fit(Y, T, X=X)

# 获取ATE估计值和置信区间
ate_estimate = dml_estimator.effect(T=1) # 从T=0到T=1的效应
ate_ci = dml_estimator.effect_interval(T=1, alpha=0.05) # 95% 置信区间

print(f"DML估计的ATE: {ate_estimate[0]:.4f}")
print(f"95% 置信区间: [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")
# 与之前计算的真实ATE进行比较
print(f"真实平均处理效应: {np.mean(true_cate):.4f}")
```

`LinearDML` 在内部处理交叉拟合过程，以防止过拟合 (overfitting)并为推断提供标准误差。将估计的ATE及其置信区间与我们模拟的真实平均效应进行比较。它们应该合理接近，表明DML即使在高维混杂情况下也能恢复平均效应。

### 实现用于CATE的因果森林

DML提供的是*平均*效应的估计，而因果森林则旨在表明处理效应中的*异质性*。它们调整了随机森林算法来估计CATE，$E[Y(1) - Y(0) | X=x]$。`econml` 提供 `CausalForestDML`，它将DML残差化方法整合到森林结构中。

```python
# 实例化CausalForestDML估计量
# 它使用DML原则在森林分裂内进行正交化
# 我们可以指定辅助模型或使用默认设置（通常是梯度提升）
cf_estimator = CausalForestDML(model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
                               model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
                               discrete_treatment=True,
                               n_estimators=1000, # 更多的树通常对森林效果更佳
                               min_samples_leaf=10,
                               max_depth=10,
                               random_state=123)

# 拟合因果森林
cf_estimator.fit(Y, T, X=X)

# 估计数据集中所有样本的CATE
cate_estimates = cf_estimator.effect(X=X)

print(f"CATE估计的形状: {cate_estimates.shape}")
print(f"CATE估计示例（前5个）: {cate_estimates[:5].round(4)}")
```

`cf_estimator.effect(X=X)` 返回一个CATE估计数组，每个样本根据其特征 $X$ 对应一个估计值。

### CATE异质性可视化

为了理解处理效应如何变化，我们可以将估计的CATE与导致异质性的特征（在我们的模拟中是 $X_0$）进行可视化。

```python
# 创建估计CATE与X0的散点图
# 为了可视化，我们抽取一些点以避免重叠
sample_indices = np.random.choice(n_samples, 500, replace=False)
x0_sample = X[sample_indices, 0]
cate_sample = cate_estimates[sample_indices]

# 定义Plotly图表数据和布局
plotly_fig = {
    "data": [
        {
            "type": "scatter",
            "mode": "markers",
            "x": x0_sample.tolist(), # 使用抽样数据
            "y": cate_sample.tolist(), # 使用抽样数据
            "marker": {
                "color": "#228be6", # 蓝色
                "size": 6,
                "opacity": 0.6
            },
            "name": "估计CATE"
        },
        # 添加显示真实关系的线条以供参考
        {
            "type": "scatter",
            "mode": "lines",
            "x": sorted(x0_sample),
            "y": (true_ate + heterogeneity_slope * np.sort(x0_sample)).tolist(),
            "line": {
                "color": "#f03e3e", # 红色
                "width": 2,
                "dash": "dash"
            },
            "name": "真实CATE"
        }
    ],
    "layout": {
        "title": "估计CATE与特征X0对比",
        "xaxis": {"title": "特征X0"},
        "yaxis": {"title": "估计CATE"},
        "showlegend": True,
        "legend": {"x": 0.01, "y": 0.99},
        "width": 700,
        "height": 450,
        "template": "plotly_white"
    }
}
```



![估计CATE与特征X0对比](plots/4367-0.json)



> 对部分数据的特征X0值绘制的估计条件平均处理效应 (CATE)。虚线红线表明模拟中定义的真实CATE关系 ($CATE = 1.5 + 0.5 * X_0$)。

该图应显示因果森林的CATE估计值通常遵循真实的向上斜率，这表明模型成功地捕捉到了处理效应随着 $X_0$ 值升高而增加的趋势。散点图表示个体CATE估计值，它们自然会在真实线周围存在一些偏差。

您可以进一步解释CATE估计值，使用 `econml.cate_interpreter.SingleTreeCateInterpreter` 等工具，以理解哪些特征对引起异质性最重要。

```python
# 用简单树解释CATE模型
intrp = SingleTreeCateInterpreter(include_model_uncertainty=False, max_depth=2)
intrp.interpret(cf_estimator, X)

# 绘制解释树（需要graphviz）
# intrp.plot(feature_names=X_df.columns, fontsize=12)
# 如果模型运行良好，该图将主要显示基于X0的分裂。
print("CATE解释树结构:")
print(intrp.text_summary(feature_names=X_df.columns))
```

此总结提供了一个简化的树结构，近似于因果森林的CATE函数，通常突出显示最具影响力的特征（如我们案例中的 $X_0$）。

### 总结与后续步骤

本次实践环节演示了如何使用 `econml` 在高维环境中实现双重机器学习 (machine learning)进行ATE估计，以及因果森林进行CATE估计。我们看到了DML如何通过辅助模型处理混杂因素，从而有效地恢复平均效应，以及因果森林如何表明处理效应中的异质性。

要点：

- DML依赖于辅助函数 ($E[Y|X]$ 和 $E[T|X]$) 的准确建模。为这些任务选择机器学习模型很重要。
- 因果森林建立在DML原则之上，以估计效应如何随个体特征 $X$ 变化。
- 使用模拟数据可以验证这些方法，通过将估计值与已知真实情况进行比较。

接下来，您可以在DML和因果森林中尝试不同的模型（例如 `LassoCV` 与 `GradientBoosting`），调整超参数 (parameter) (hyperparameter)，并将这些技术应用于您自己的数据集。请记住仔细考虑这些方法的基本假设，特别是给定观测协变量 $X$ 的无混杂性假设（条件可忽略性）。下一章将研究由于未观测到的混杂因素导致此假设被违反的情况。

## 参考资料

- [Double/Debiased Machine Learning for Treatment and Causal Parameters](https://arxiv.org/abs/1608.00060) — Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins (2018)
  Journal: Econometrics Journal; Publisher: Wiley; Volume: 21; Pages: C1-C68; DOI: [10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097)
  介绍双重机器学习理论和渐近特性的基础论文，有助于理解其在高维设置下的统计保证。
- [Microsoft EconML Library Documentation](https://econml.azurewebsites.net/) — The EconML Team (2024)
  Python EconML 库的官方文档，提供实现 DML 和因果森林的指南和 API 参考。
- [Causal Inference: The Mixtape](https://mixtape.scot/) — Scott Cunningham (2021)
  Publisher: Yale University Press; Pages: 584
  一本现代且易于理解的教科书，涵盖了广泛的因果推断方法，包括 DML 和因果森林的详细解释。
