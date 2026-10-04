---
course: "intro-feature-engineering"
chapter: "feature-scaling-transformation"
lesson: "box-cox-transformation"
sourceId: 1342
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-4-feature-scaling-transformation/box-cox-transformation"
title: "Box-Cox 变换"
description: "使用 Box-Cox 变换来稳定方差并使数据更接近高斯分布（适用于正值数据）。"
order: 6
plots: ["plots/1342-0.json"]
sourceHash: "c8eba75ff0c5e62e3368f70f2a16e95c92601bf0f754c784f31adcb0d656a3fc"
sourceCorrections: []
---

虽然标准化 (Standardization) 和归一化 (Normalization) 等缩放方法可以调整数据的范围和中心，但它们并不会根本性地改变数据分布的*形态*。如果您的数据严重偏斜，仅靠缩放无法解决问题。偏斜的分布可能会违反某些模型的假设（例如线性回归中误差的正态性假设），有时还会降低其他模型的性能。

Box-Cox 变换是一种有效的统计方法，属于幂变换家族，专门用于解决此问题。其主要目的是将非正态分布的数据（特别是右偏数据）转换为更接近正态（高斯）分布的形态。它还有助于稳定方差。

### Box-Cox 公式

该变换包含一个参数 (parameter) lambda ($\lambda$)，定义如下：


$$
y^{(\lambda)} = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{如果 } \lambda \neq 0 \\
\ln(x) & \text{如果 } \lambda = 0
\end{cases}
$$


其中，$x$ 是原始数据点，$y^{(\lambda)}$ 是变换后的数据点。

Box-Cox 变换的一个重要限制是，它**要求所有输入数据 ($x$) 严格为正 ($x > 0$)**。如果您的数据包含零或负值，则不能直接应用 Box-Cox 变换。您可以考虑添加一个小的常数来平移数据（如果这在您的场景中有意义），或者使用相关的 Yeo-Johnson 变换，我们接下来会讲到。

### 找到最优 Lambda ($\lambda$)

通常您无需手动选择 $\lambda$。$\lambda$ 的最优值通常通过计算确定。标准方法是找到使变换后的对数似然函数最大化的 $\lambda$ 值，从而有效地选择使结果数据最接近高斯分布的变换。

### 为何应用 Box-Cox？

1. **正态性：** 它有助于使数据分布更对称，呈钟形，更接近正态分布。这对于假设数据为正态分布的模型有益。
2. **方差稳定：** 它可以使数据值范围内的方差更恒定，这对于对异方差性（非恒定方差）敏感的模型有帮助。
3. **模型性能提升：** 通过解决偏斜和非恒定方差问题，Box-Cox 有时能带来线性模型和其他对这些数据属性敏感的算法的更好性能和更可靠的结果。

### 使用 Scikit-learn 实现 Box-Cox

Scikit-learn 在其 `preprocessing` 模块中提供了 `PowerTransformer` 类来应用 Box-Cox 变换。

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# 生成一些偏斜数据（例如，指数分布）
np.random.seed(42)
skewed_data = np.random.exponential(scale=2, size=1000) + 0.1 # 添加0.1以确保数值为正
skewed_data = skewed_data.reshape(-1, 1) # 调整形状以适应Scikit-learn变换器

# 使用 method='box-cox' 初始化变换器
boxcox_transformer = PowerTransformer(method='box-cox', standardize=False)
# 设置 standardize=False 只应用 Box-Cox 变换。
# 设置 standardize=True（默认）将先应用 Box-Cox，然后对结果进行标准化。

# 将变换器拟合到数据（找到最优lambda）并变换数据
boxcox_transformed_data = boxcox_transformer.fit_transform(skewed_data)

# 学习到的 lambda 值
print(f"找到最优lambda: {boxcox_transformer.lambdas_[0]:.4f}")

# --- 可视化（使用 Matplotlib/Seaborn 进行演示）---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(skewed_data, kde=True, ax=axes[0], color='#4dabf7', bins=30)
axes[0].set_title('原始偏斜数据')
axes[0].set_xlabel('值')
axes[0].set_ylabel('频率')

sns.histplot(boxcox_transformed_data, kde=True, ax=axes[1], color='#38d9a9', bins=30)
axes[1].set_title('Box-Cox 变换后的数据')
axes[1].set_xlabel('变换后的值')
axes[1].set_ylabel('频率')

plt.tight_layout()
plt.show()
```

> 直方图显示了原始右偏分布在应用 Box-Cox 变换后如何被重塑为更对称的钟形曲线。

或者，这里是显示分布的 Plotly 可视化：



![原始数据与 Box-Cox 变换后的数据对比](plots/1342-0.json)



> 数据分布在应用 Box-Cox 变换前（左，蓝色）与变换后（右，绿色）的比较。变换后的数据显示出显著降低的偏斜度。

### 重要注意事项

- **仅限正值数据：** 请记住对正输入值的严格要求。
- **可解释性：** 应用 Box-Cox（或任何幂变换）会改变特征的尺度和解释。变换后的特征的单位变化不再对应原始特征的单位变化。
- **在训练数据上拟合：** 与所有预处理步骤一样，`PowerTransformer` *仅*在您的训练数据上拟合。使用已拟合的变换器将 *相同* 的变换（使用学习到的 $\lambda$）应用于训练和测试数据集，以防止数据泄露并保持一致性。

Box-Cox 是一种处理偏斜数值数据的有价值技术，当其假设得到满足时，通常能改善模型的行为和性能，尤其是对于对特征分布敏感的模型。

## 参考资料

- [An Analysis of Transformations](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x) — G. E. P. Box and D. R. Cox (1964)
  Journal: Journal of the Royal Statistical Society. Series B (Methodological); Publisher: Blackwell Publishing for the Royal Statistical Society; Volume: 26; Pages: 211-243; DOI: [10.1111/j.2517-6161.1964.tb00553.x](https://doi.org/10.1111/j.2517-6161.1964.tb00553.x)
  介绍Box-Cox变换的原始学术论文，解释其实现正态性和方差稳定性的统计基础。
- [sklearn.preprocessing.PowerTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html) — scikit-learn developers (2023)
  scikit-learn PowerTransformer类的官方文档，该类包含Box-Cox和Yeo-Johnson变换。
- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://learning.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  一本关于特征工程的实用指南，讨论Box-Cox等各种数据变换，以提高机器学习模型性能。
