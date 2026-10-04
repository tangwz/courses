---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "practice-scaling-transforming"
sourceId: 1361
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/practice-scaling-transforming"
title: "动手实践：特征缩放与转换"
description: "使用 Scikit-learn 应用不同的缩放和转换方法，并分析它们的作用。"
order: 10
plots: ["plots/1361-0.json", "plots/1361-1.json", "plots/1361-2.json", "plots/1361-3.json"]
sourceHash: "e5ea176d62e6fe60f2045a26e932fb23d674794acd4f562aa244350483fdd1b4"
sourceCorrections: []
---

数值特征的缩放和转换方法是数据预处理中的基础。这里将演示如何在一个示例数据集上使用 Python 的 Scikit-learn 库应用这些技术。这项动手练习将阐明 `StandardScaler`、`MinMaxScaler`、`RobustScaler`、`PowerTransformer` 和 `QuantileTransformer` 的工作原理及其效果差异。

### 设置与初始数据查看

首先，导入所需的库并创建一个示例数据集。我们将使用 Pandas 进行数据操作，NumPy 进行数值运算，Matplotlib 和 Seaborn 进行初步可视化观察（尽管最终图表将以网络友好的格式呈现），以及 Scikit-learn 提供缩放和转换工具。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split
import scipy.stats as stats

# 生成具有不同尺度和偏度的人造数据
np.random.seed(42) # 用于结果复现
data = pd.DataFrame({
    'Feature_A': np.random.rand(100) * 100, # 尺度 0-100
    'Feature_B': np.random.randn(100) * 10 + 50, # 正态分布，不同尺度
    'Feature_C': np.random.exponential(scale=20, size=100) + 1, # 指数分布（偏斜），尺度 > 0
    'Feature_D': np.random.rand(100) * 10 - 5 # 包含负值
})

# 为 Feature_A 添加一些异常值
data.loc[[10, 30, 90], 'Feature_A'] = [250, -80, 300]

# 拆分数据用于演示（可选但推荐）
# 在实际场景中，只对训练数据拟合转换器
X_train, X_test = train_test_split(data, test_size=0.3, random_state=42)

print("原始数据描述（训练集）：")
print(X_train.describe())

print("\n原始数据前几行（训练集）：")
print(X_train.head())
```

在应用任何转换之前，我们先将训练特征的分布情况可视化。这有助于找出不同的尺度和偏度，并说明为什么缩放和转换可能是必需的。

```python
# 可视化原始分布（使用 seaborn 进行说明，最终用 Plotly JSON 呈现）
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
sns.histplot(X_train['Feature_A'], kde=True, ax=axes[0], color='#4dabf7')
axes[0].set_title('Feature_A 分布')
sns.histplot(X_train['Feature_B'], kde=True, ax=axes[1], color='#748ffc')
axes[1].set_title('Feature_B 分布')
sns.histplot(X_train['Feature_C'], kde=True, ax=axes[2], color='#f06595')
axes[2].set_title('Feature_C 分布（偏斜）')
sns.histplot(X_train['Feature_D'], kde=True, ax=axes[3], color='#94d82d')
axes[3].set_title('Feature_D 分布')
plt.tight_layout()
# plt.show() # 我们将在下面用 Plotly 呈现此思想
```

让我们表示 `Feature_C` 的分布，它显示出明显的偏斜。



![Feature_C 的原始分布（偏斜）](plots/1361-0.json)



> Feature\_C 的分布图，显示其在任何转换前的右偏特性。

我们能看到不同的范围（例如，`Feature_A` 可能从 -80 到 300，`Feature_D` 从 -5 到 5）以及不同的分布形状（`Feature_C` 明显右偏）。

### 应用缩放技术

缩放调整特征的范围，而不会显著改变其分布的形状。记住要在训练数据上 `fit` 缩放器，然后 `transform` 训练和测试数据。

#### 标准化（Z-分数缩放）

`StandardScaler` 移除均值并将特征缩放到单位方差。公式是 $Z = (x - \mu) / \sigma$。

```python
# 初始化并拟合 StandardScaler
scaler_standard = StandardScaler()
scaler_standard.fit(X_train) # 只对训练数据进行拟合

# 转换训练和测试数据
X_train_std = scaler_standard.transform(X_train)
X_test_std = scaler_standard.transform(X_test)

# 转换回 DataFrame 以便查看
X_train_std_df = pd.DataFrame(X_train_std, columns=X_train.columns, index=X_train.index)

print("\n标准化数据描述（训练集）：")
print(X_train_std_df.describe().round(2)) # 均值应接近 0，标准差应接近 1
```

请注意，标准化后所有特征的 `mean`（均值）约等于 0，`std`（标准差）约等于 1。

#### 归一化 (normalization)（最小-最大缩放）

`MinMaxScaler` 将特征缩放到一个固定范围，通常是 [0, 1]。公式是 $X_{\text{缩放后}} = (x - \text{最小值}) / (\text{最大值} - \text{最小值})$。

```python
# 初始化并拟合 MinMaxScaler
scaler_minmax = MinMaxScaler()
scaler_minmax.fit(X_train)

# 转换训练和测试数据
X_train_minmax = scaler_minmax.transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

# 转换回 DataFrame
X_train_minmax_df = pd.DataFrame(X_train_minmax, columns=X_train.columns, index=X_train.index)

print("\n最小-最大缩放数据描述（训练集）：")
print(X_train_minmax_df.describe().round(2)) # 最小值应为 0，最大值应为 1
```

此时，所有特征的 `min` 为 0，`max` 为 1，符合预期。

#### 抗离群值缩放

`RobustScaler` 使用抗离群值的统计量，特别是四分位数之间的范围 (IQR)。它会移除中位数并根据分位数范围（默认是 IQR：Q3 - Q1）缩放数据。

```python
# 初始化并拟合 RobustScaler
scaler_robust = RobustScaler()
scaler_robust.fit(X_train)

# 转换训练和测试数据
X_train_robust = scaler_robust.transform(X_train)
X_test_robust = scaler_robust.transform(X_test)

# 转换回 DataFrame
X_train_robust_df = pd.DataFrame(X_train_robust, columns=X_train.columns, index=X_train.index)

print("\nRobust 缩放数据描述（训练集）：")
print(X_train_robust_df.describe().round(2)) # 中位数应接近 0
```

`RobustScaler` 将数据围绕中位数（约等于 0）居中，并根据 IQR 进行缩放。此方法受我们在 `Feature_A` 中引入的大异常值的影响较小。

让我们可视化这些缩放器对含有异常值的 `Feature_A` 的作用。



![缩放器对 Feature_A 的作用（含异常值）](plots/1361-1.json)



> 比较应用不同缩放器后 Feature\_A 的分布。请注意，与受异常值影响、范围被拉伸的 StandardScaler 和 MinMaxScaler 相比，RobustScaler 如何集中了数据的主体部分。

### 应用转换技术

转换旨在改变分布的形状，通常使其更接近高斯分布或均匀分布。这对假设正态性的模型有利。

#### 幂变换（Box-Cox 和 Yeo-Johnson）

`PowerTransformer` 应用 Box-Cox（要求数据为正值）或 Yeo-Johnson（处理正值、零值和负值数据）转换，以稳定方差并最小化偏度。

让我们对 `Feature_C`（偏斜，正值）和 `Feature_D`（包含负值）应用 Yeo-Johnson。

```python
# 初始化并拟合 PowerTransformer（Yeo-Johnson）
pt_yj = PowerTransformer(method='yeo-johnson', standardize=True) # standardize=True 会在转换后应用 Z 缩放
# 对选定的训练数据列进行拟合
pt_yj.fit(X_train[['Feature_C', 'Feature_D']])

# 转换训练数据
X_train_yj = pt_yj.transform(X_train[['Feature_C', 'Feature_D']])
X_train_yj_df = pd.DataFrame(X_train_yj, columns=['Feature_C_yj', 'Feature_D_yj'], index=X_train.index)

# Box-Cox 只应用于 Feature_C（必须严格为正值）
pt_bc = PowerTransformer(method='box-cox', standardize=True)
pt_bc.fit(X_train[['Feature_C']]) # 只对 Feature_C 进行拟合

# 转换训练数据中的 Feature_C
X_train_bc = pt_bc.transform(X_train[['Feature_C']])
X_train_bc_df = pd.DataFrame(X_train_bc, columns=['Feature_C_bc'], index=X_train.index)

# 合并转换后的特征以便可视化
X_train_transformed = pd.concat([X_train_yj_df, X_train_bc_df], axis=1)

print("\n转换后数据前几行（训练集）：")
print(X_train_transformed.head())
```

让我们可视化原始的 `Feature_C` 及其转换后的版本。



![Feature_C 上的幂变换](plots/1361-2.json)



> Feature\_C 分布的比较：原始（偏斜）、Yeo-Johnson 转换后和 Box-Cox 转换后。两种转换都显著降低了偏度，使分布更对称。

我们还可以使用概率图（Q-Q 图）直观评估转换后的分布与正态分布的接近程度。点近似落在对角线上表明正态性。

```python
# 使用 Q-Q 图可视化正态性（使用 Matplotlib/SciPy）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
stats.probplot(X_train['Feature_C'], dist="norm", plot=axes[0])
axes[0].set_title('原始 Feature_C Q-Q 图')
stats.probplot(X_train_transformed['Feature_C_yj'], dist="norm", plot=axes[1])
axes[1].set_title('Yeo-Johnson Feature_C Q-Q 图')
stats.probplot(X_train_transformed['Feature_C_bc'], dist="norm", plot=axes[2])
axes[2].set_title('Box-Cox Feature_C Q-Q 图')
plt.tight_layout()
# plt.show() # 在真实界面中，这些图会显示转换后点更接近直线。
```

Q-Q 图将直观地确认，转换后特征的点与对角线对齐 (alignment)得比原始偏斜特征好得多，这表明其更接近正态分布。

#### 分位数转换

`QuantileTransformer` 根据分位数将数据分布映射到均匀分布或正态分布。它可以使不同的分布变得更相似。

```python
# 初始化并拟合 QuantileTransformer（转换为均匀分布）
qt_uniform = QuantileTransformer(output_distribution='uniform', n_quantiles=min(len(X_train), 100), random_state=42)
qt_uniform.fit(X_train)

# 转换训练数据
X_train_qt_uniform = qt_uniform.transform(X_train)
X_train_qt_uniform_df = pd.DataFrame(X_train_qt_uniform, columns=X_train.columns, index=X_train.index)

# 初始化并拟合 QuantileTransformer（转换为正态分布）
qt_normal = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(X_train), 100), random_state=42)
qt_normal.fit(X_train)

# 转换训练数据
X_train_qt_normal = qt_normal.transform(X_train)
X_train_qt_normal_df = pd.DataFrame(X_train_qt_normal, columns=X_train.columns, index=X_train.index)

print("\n分位数转换数据（均匀分布）描述：")
print(X_train_qt_uniform_df.describe().round(2)) # 应近似均匀分布 [0, 1]

print("\n分位数转换数据（正态分布）描述：")
print(X_train_qt_normal_df.describe().round(2)) # 应近似正态分布（均值~0，标准差~1）
```

让我们可视化分位数转换对 `Feature_C` 的作用。



![Feature_C 上的分位数转换](plots/1361-3.json)



> Feature\_C 分布的比较：原始（偏斜）、分位数转换为均匀分布后以及分位数转换为正态分布后。转换器根据秩有效地重塑了数据。

### 与管道集成

在实践中，这些转换器通常作为 Scikit-learn `Pipeline` 中的步骤来使用。这确保了在交叉验证期间以及对新数据进行预测时，缩放/转换的正确应用。

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression # 示例模型

# 创建管道：缩放 -> 幂变换（Yeo-Johnson） -> 线性回归
# 如有需要，只对特定列应用转换（使用 ColumnTransformer - 更高级）
# 为简单起见，这里假设我们应用于传入的所有特征

# 针对受益于 Yeo-Johnson 的 Feature_C 和 Feature_D 的示例
pipeline = Pipeline([
    ('scaler', RobustScaler()), # 首先处理潜在的异常值
    ('transformer', PowerTransformer(method='yeo-johnson', standardize=True)),
    ('model', LinearRegression()) # 示例模型步骤
])

# 接着你将在训练数据（X_train, y_train）上拟合此管道
# pipeline.fit(X_train[['Feature_C', 'Feature_D']], y_train) # 假设 y_train 存在

# 管道在拟合期间自动对缩放器/转换器应用 fit_transform
# 并在预测/评分期间对缩放器/转换器应用 transform

print("\n管道已创建（示例结构）：")
print(pipeline)
```

### 总结

本次实践说明了如何使用 Scikit-learn 应用各种缩放和转换技术：

- **缩放（`StandardScaler`、`MinMaxScaler`、`RobustScaler`）**：调整特征的范围/尺度。对基于距离的算法和梯度下降 (gradient descent)方法很重要。`RobustScaler` 对异常值不那么敏感。
- **转换（`PowerTransformer`、`QuantileTransformer`）**：改变分布的形状，通常是为了减少偏度或使其近似正态/均匀分布。对假设特定分布的模型有利。

你看到了如何在训练数据上拟合这些转换器，并将其应用于训练集和测试集。在应用这些方法前后可视化分布是理解其作用的重要一步。请记住，技术选择取决于你数据的具体特性和你打算使用的机器学习 (machine learning)模型的要求。通常需要通过实验和评估来为你的特定问题找到最佳方法。

## 参考资料

- [Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html) — scikit-learn developers (2023)
  Publisher: scikit-learn project
  scikit-learn 文档的这一部分详细解释并提供了所有展示的缩放和变换方法的示例，包括 StandardScaler、MinMaxScaler、RobustScaler、PowerTransformer 和 QuantileTransformer。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953242/) — Alice Zheng, Amanda Casari (2018)
  Publisher: O'Reilly Media
  本书有专门章节介绍数值特征工程，涵盖了本节中讨论的各种缩放和变换技术的原理和实际应用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  这本实践指南包含专门的数据预处理部分，解释了使用 scikit-learn 进行特征缩放和变换的原因和方法，与本课程的实践性质相符。
- [An Analysis of Transformations](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x) — G. E. P. Box and D. R. Cox (1964)
  Journal: Journal of the Royal Statistical Society. Series B (Methodological); Publisher: Blackwell Publishing for the Royal Statistical Society; Volume: 26; Pages: 211-243; DOI: [10.1111/j.2517-6161.1964.tb00553.x](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x)
  这篇开创性论文介绍了 Box-Cox 变换，为所讨论的关键幂变换技术提供了统计背景。
