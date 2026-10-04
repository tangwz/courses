---
course: "model-interpretability-explainability"
chapter: "shap-additive-explanations"
lesson: "shap-python-implementation"
sourceId: 4086
sourceUrl: "https://apxml.com/zh/courses/model-interpretability-explainability/chapter-3-shap-additive-explanations/shap-python-implementation"
title: "SHAP 在 Python 中的实现"
description: "通过 Python 中 SHAP 库的示例来计算和可视化解释。"
order: 8
plots: ["plots/4086-0.json", "plots/4086-1.json", "plots/4086-2.json"]
sourceHash: "e3ef6c8546a25db3a498e46a651b26ac8ebfc91621034c81d60d682819c4f49d"
sourceCorrections: []
---

SHapley Additive exPlanations (SHAP) 的实际操作利用了广受欢迎的 `shap` Python 库。这个库提供了各种 SHAP 算法（包括 KernelSHAP 和 TreeSHAP）的高效实现，以及强大的可视化工具。

我们将逐步演示为示例模型生成 SHAP 解释的过程。如先决条件中所述，假定您已熟悉 Python 以及使用 `scikit-learn` 和 `pandas` 等库的基本机器学习 (machine learning)工作流程。

### 环境设置

首先，请确保您已安装必要的库。您主要需要 `shap`、`scikit-learn`、`pandas` 和 `numpy`。如果您尚未安装，通常可以使用 pip 进行安装：

```bash
pip install shap scikit-learn pandas numpy matplotlib
```

现在，让我们在 Python 脚本或 Notebook 中导入所需的模块：

```python
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# 可选：为 Jupyter 等环境初始化 JS 可视化支持
# shap.initjs() # 如果使用 Jupyter notebooks/lab，请取消注释此行
```

`shap.initjs()` 调用在 Jupyter Notebooks 等某些交互式环境中渲染 SHAP 图表是必需的。

### 准备数据和训练模型

为了演示 SHAP，我们需要一个训练好的机器学习 (machine learning)模型。让我们创建一个合成数据集并训练一个 `RandomForestClassifier`。基于树的模型是一个很好的起点，因为 `shap` 库包含 `TreeExplainer`，如前所述，它是针对这些类型模型的高度优化方法。

```python
# 生成合成分类数据
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=2, n_classes=2, random_state=42)
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# 训练 RandomForestClassifier 模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"模型准确率: {model.score(X_test, y_test):.4f}")
```

我们现在有一个训练好的 `RandomForestClassifier` 可用于解释。

### 选择和使用 SHAP 解释器

`shap` 库提供了针对不同模型类型优化的各种“解释器”。由于我们训练的是随机森林，最有效的选择是 `shap.TreeExplainer`。如果我们处理的模型没有专门的解释器（比如复杂的神经网络 (neural network)或使用非线性核的 SVM），我们可以使用与模型无关的 `shap.KernelExplainer`。

让我们实例化 `TreeExplainer`：

```python
# 创建一个 TreeExplainer 对象
explainer = shap.TreeExplainer(model)
```

创建解释器对象通常需要传入训练好的模型。对于某些解释器，例如 `KernelExplainer`，您可能还需要提供一个背景数据集（通常是训练数据的样本）来表示特征值的预期分布。`TreeExplainer` 直接从树结构和训练数据分布中推导预期，使其初始化更简单。

### 计算 SHAP 值

解释器准备就绪后，我们可以为任何我们想要解释的实例集计算 SHAP 值。通常，您可以解释测试集上的预测或感兴趣的特定实例。`shap_values` 方法计算这些值。

```python
# 计算测试集的 SHAP 值
# 对于 TreeExplainer，shap_values 通常返回一个列表（每个输出类别一个）
# 或者一个单独的数组用于回归。对于使用 scikit-learn 进行的二分类，
# 它通常返回两个数组的列表 [shap_values_class_0, shap_values_class_1]。
# 我们通常对正类别（类别 1）的 SHAP 值感兴趣。
shap_values = explainer.shap_values(X_test)

# 如果 shap_values 是一个列表（分类常见），则选择类别 1 的值
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    # 处理回归或其他输出可能为单个数组的情况
    shap_values_class1 = shap_values

# 解释器还提供预期值（基准值）
# 这是训练数据（或背景数据）上的平均模型预测
expected_value = explainer.expected_value
# 对于分类，expected_value 可能是一个列表 [类别 0 的 E[y], 类别 1 的 E[y]]
if isinstance(expected_value, list):
    expected_value_class1 = expected_value[1]
else:
    expected_value_class1 = expected_value

print(f"SHAP 值形状: {shap_values_class1.shape}") # 应该是 (样本数, 特征数)
print(f"类别 1 的预期值（基准值）: {expected_value_class1}")
```

`shap_values` 变量（此处为 `shap_values_class1`）存放一个 NumPy 数组，其中每行对应 `X_test` 中的一个实例，每列对应一个特征。值 `shap_values[i, j]` 表示特征 `j` 对实例 `i` 预测的贡献，使其偏离基准值（`expected_value`）。`expected_value` 表示解释器训练或校准时所用数据集上的平均预测输出。对于分类模型，SHAP 值通常计算的是模型在最终激活函数 (activation function)（例如对数几率或概率）*之前* 的输出。`TreeExplainer` 通常计算其边缘输出。

请记住加和性质：对于任何单个预测 $i$，其所有特征的 SHAP 值之和加上基准值等于该预测的模型输出：
$\text{模型输出}(x_i) \approx \text{预期值} + \sum_{j=1}^{M} \text{shap值}(x_i, j)$
其中 $M$ 是特征的数量。

### SHAP 解释可视化

`shap` 库在可视化方面表现出色，使计算出的 SHAP 值更易于理解。

#### 瀑布图（局部解释）

瀑布图是可视化单个预测解释的有效方式。它们显示了哪些特征使模型输出更高（正 SHAP 值，通常显示为红色），哪些使模型输出更低（负 SHAP 值，通常显示为蓝色）。

```python
# 解释测试集中第一个实例的预测
instance_index = 0

shap.force_plot(expected_value_class1,
                shap_values_class1[instance_index, :],
                X_test.iloc[instance_index, :],
                matplotlib=True) # 如果需要静态图，请使用 matplotlib=True
```

> 瀑布图显示了单个预测的特征贡献。将预测推向正类别（值 > 基准值）的特征显示为红色，而推向负类别的特征显示为蓝色。条形的长度表示特征影响的程度。

您还可以为多个实例创建瀑布图，这些图表可以旋转并垂直堆叠：

```python
# 可视化前 100 个测试实例的解释
shap.force_plot(expected_value_class1,
                shap_values_class1[:100, :],
                X_test.iloc[:100, :])
```

> 堆叠瀑布图可视化了多个实例（此处为测试集中的前 100 个）的 SHAP 值。每个实例是一行，允许比较不同样本的特征影响。实例通常按相似性排序。

#### 汇总图（全局解释）

汇总图通过结合所有实例的 SHAP 值，提供特征重要性的全局概览。默认的 `summary_plot` 显示了每个特征的重要性（平均绝对 SHAP 值）以及该特征的 SHAP 值分布。

```python
# 创建整体特征重要性的汇总图
shap.summary_plot(shap_values_class1, X_test, plot_type="dot")
```



![SHAP 汇总图（点状）](plots/4086-0.json)



> SHAP 汇总图（点状版本）。特征按其全局重要性（平均绝对 SHAP 值）排序。每个点代表特定实例和特征的 SHAP 值。水平位置显示 SHAP 值（影响），颜色表示原始特征值（高或低）。这不仅显示了 *哪些* 特征重要，还显示了它们的值如何影响预测。例如，`feature_0` 的高值似乎会强烈增加预测向类别 1 的趋势。

另外，条形图汇总图简单地显示每个特征的平均绝对 SHAP 值，给出了全局特征重要性的直接排名。

```python
# 创建条形汇总图
shap.summary_plot(shap_values_class1, X_test, plot_type="bar")
```



![SHAP 特征重要性（条形）](plots/4086-1.json)



> SHAP 汇总图（条形版本）。此图通过显示每个特征在所有提供实例上的平均绝对 SHAP 值来汇总影响。它按特征对模型预测的整体贡献给出清晰的排名。

#### 依赖图（特征关联）

依赖图显示了模型输出对单个特征的依赖性如何在其值范围内变化。它们绘制了特征值与所有实例对应的 SHAP 值。着色可以可选地表示与另一个特征的关联效果。

```python
# 为 'feature_0' 创建一个依赖图
# 可选择性地根据 'feature_3' 着色以查看关联效果
shap.dependence_plot("feature_0", shap_values_class1, X_test, interaction_index="feature_3")

# 为 'feature_4' 创建一个没有明确关联着色的依赖图
# shap.dependence_plot("feature_4", shap_values_class1, X_test, interaction_index=None)
```



![feature_0 的 SHAP 依赖图](plots/4086-2.json)



> SHAP 依赖图显示了 `feature_0` 的值及其 SHAP 值在所有测试样本中的关系。颜色表示 `feature_3` 的值，显示了潜在的关联效果。此处，`feature_0` 的值越高通常会导致 SHAP 值越高（增加类别 1 的可能性）。着色可能表明这种效果受到 `feature_3` 的调节。

这些图表提供了一个强大的工具集，可将 SHAP 值的理论原理转化为关于模型行为的可操作见解，无论是针对个体预测还是总体趋势。下一节将提供一个动手实践练习，以巩固这些实现步骤。

## 参考资料

- [A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) — Scott M. Lundberg, Su-In Lee (2017)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 30; Pages: 4765-4774; DOI: [10.5555/3295222.3295230](https://doi.org/10.5555/3295222.3295230)
  介绍SHAP框架，将Shapley值与机器学习模型的局部解释联系起来。描述了KernelSHAP和TreeSHAP。
- [SHAP: A unified approach to explain the output of any machine learning model (Python Library Documentation)](https://shap.readthedocs.io/en/latest/) — Scott Lundberg and SHAP Contributors (2024)
  `shap` Python库的官方文档，提供了安装说明、API细节以及各种解释器和可视化示例。
- [Interpretable Machine Learning: A Guide for Making Black Box Models Explainable](https://christophm.github.io/interpretable-ml-book/) — Christoph Molnar (2024)
  在线书籍涵盖了可解释机器学习的概念，包括专门介绍SHAP值及其应用的章节。
