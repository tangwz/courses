---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "yeo-johnson-transformation"
sourceId: 1343
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/yeo-johnson-transformation"
title: "Yeo-Johnson 变换"
description: "应用 Yeo-Johnson 变换，它与 Box-Cox 类似，但支持零值和负值。"
order: 7
plots: ["plots/1343-0.json"]
sourceHash: "fc1bcdeddc6f0a66aa8512e3f219b5af7ac9bf85c294f5f69554db00a0e7e63a"
sourceCorrections: []
---

虽然对数变换和 Box-Cox 变换在处理正偏斜数据方面表现良好，但当数据包含零值或负值时，它们会遇到局限。对数变换对非正数无定义，标准的 Box-Cox 变换要求严格正的输入。在这种情况下，Yeo-Johnson 变换便可派上用场。

该变换由 I.K. Yeo 和 R.A. Johnson 于 2000 年提出，它属于幂变换家族，与 Box-Cox 相似，但其适用范围扩展到包含非正值的数据。其目的不变：使方差稳定，减少偏斜，并使数据分布更接近正态（高斯）分布，这对某些建模算法可能有利。

### Yeo-Johnson 如何工作

与 Box-Cox 类似，Yeo-Johnson 变换寻找一个最优参数 (parameter)，通常表示为 $\lambda$，以应用幂变换。然而，它根据数据点 $x$ 是非负值还是负值，使用略有不同的公式：

- 对于 $x \ge 0$：
  
  $$
  y = \begin{cases}
      \frac{(x + 1)^\lambda - 1}{\lambda} & \text{如果 } \lambda \neq 0 \\
      \log(x + 1) & \text{如果 } \lambda = 0
      \end{cases}
  $$
  
- 对于 $x < 0$：
  
  $$
  y = \begin{cases}
      -\frac{(-x + 1)^{2 - \lambda} - 1}{2 - \lambda} & \text{如果 } \lambda \neq 2 \\
      -\log(-x + 1) & \text{如果 } \lambda = 2
      \end{cases}
  $$
  

通常无需记住这些公式。其重要原理是该变换在零点处提供了一个连续函数，并能一致地处理正值、零值和负值，以实现对称性。最优的 $\lambda$ 通常通过计算确定，常使用最大似然估计方法，以找到使所得数据分布尽可能接近高斯分布的变换。

### 使用 Scikit-learn 应用 Yeo-Johnson

Scikit-learn 通过 `PowerTransformer` 类提供了一个方便的实现。你只需指定 `method='yeo-johnson'`。

我们来生成一些包含非正值的偏斜数据，并应用该变换：

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# 生成包含零值和负值的偏斜数据
np.random.seed(42)
data_positive = np.random.exponential(scale=2, size=70)
data_nonpositive = -np.random.exponential(scale=2, size=30) + 1 # Shift to include 0 and negatives
# 移动数据以包含零值和负值
skewed_data = np.concatenate([data_positive, data_nonpositive])
skewed_data = skewed_data.reshape(-1, 1) # Reshape for Scikit-learn transformer
# 重塑数据以适应 Scikit-learn 变换器

# 初始化并应用 Yeo-Johnson 变换
yj_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
# standardize=True applies zero-mean, unit-variance scaling *after* the transformation
# standardize=True 在变换后应用零均值、单位方差缩放
# Set to False if you only want the power transformation
# 如果只想进行幂变换，请设置为 False

transformed_data_yj = yj_transformer.fit_transform(skewed_data)

# 打印找到的最优 lambda 值
print(f"Optimal lambda found: {yj_transformer.lambdas_[0]:.4f}")

# 为绘图准备数据
df = pd.DataFrame({
    'Original': skewed_data.flatten(),
    'Yeo-Johnson Transformed': transformed_data_yj.flatten()
})

# 可视化变换前后的分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['Original'], kde=True, ax=axes[0], color='#4263eb')
axes[0].set_title('Original Data Distribution')
# 设置标题为“原始数据分布”
sns.histplot(df['Yeo-Johnson Transformed'], kde=True, ax=axes[1], color='#37b24d')
axes[1].set_title('Yeo-Johnson Transformed Data Distribution')
# 设置标题为“Yeo-Johnson 变换后的数据分布”
plt.tight_layout()
plt.show()
```

运行此代码可能会显示原始数据具有显著偏斜，而变换后数据的直方图看起来更加对称且呈钟形。`standardize=True` 参数 (parameter)（默认值）确保输出数据也具有零均值和单位方差，从而结合了变换和缩放步骤。

以下是一个比较分布的 Plotly 可视化：



![Yeo-Johnson 变换前后的数据分布](plots/1343-0.json)



> 数据分布比较。左侧图显示了包含负值的原始偏斜数据，而右侧图显示了应用 Yeo-Johnson 变换后的分布，它看起来更接近正态分布。此处使用样本数据进行说明。

### 何时选择 Yeo-Johnson

- **处理非正数数据：** 相较于 Box-Cox，它的主要优点是能够自然地处理包含零值或负值的特征。
- **减少偏斜：** 与其他幂变换类似，它能有效地使偏斜分布更加对称。
- **符合模型假设：** 有助于满足某些模型所需或有利的正态性假设（例如，线性回归残差正态性）。

### 重要考虑事项

- **可解释性：** 应用 Yeo-Johnson 会改变特征的尺度和解释。变换后特征的一个单位增加不对应原始特征的一个单位增加。模型系数将与变换后的尺度相关联。
- **在训练数据上拟合：** 与所有预处理步骤一样，`PowerTransformer` *只*应在你的训练数据集上进行拟合。使用 *相同已拟合* 的变换器（带有学习到的 $\lambda$）将变换应用于你的验证集和测试集，以避免数据泄露并确保一致性。
- **标准化：** 请注意 `PowerTransformer` 中的 `standardize` 参数 (parameter)。如果为 `True`（默认），它会在 Yeo-Johnson 变换 *之后* 应用 Z-score 缩放。如果你需要在不进行标准化的前提下获取原始幂变换值，请将 `standardize` 设置为 `False`。

Yeo-Johnson 提供了一个有价值且灵活的工具，用于使数值特征的分布趋于正态，特别是当这些特征包含所有实数范围时，它克服了 Box-Cox 方法的一个主要局限。这是一种常用方法，用于为对特征分布敏感的算法准备数据。

## 参考资料

- [A New Family of Power Transformations to Improve Normality or Symmetry](https://doi.org/10.1093/biomet/87.4.954) — In-Kwon Yeo, Richard A. Johnson (2000)
  Journal: Biometrika; Publisher: Oxford University Press; Volume: 87; Pages: 954-959; DOI: [10.1093/biomet/87.4.954](https://doi.org/10.1093/biomet/87.4.954)
  介绍了Yeo-Johnson变换，详细阐述了其数学公式和处理包含正、零和负值数据的特性。
- [sklearn.preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) — Scikit-learn Developers (2023)
  Scikit-learn官方`PowerTransformer`类文档，解释了其在Yeo-Johnson变换中的使用、参数和示例。
- [Applied Predictive Modeling](https://www.springer.com/gp/book/9781461468486) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  一本关于预测建模的全面资源，其中包含数据预处理技术（包括像Yeo-Johnson这样的幂变换）的章节。
