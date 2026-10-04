---
course: "intro-feature-engineering"
chapter: "feature-selection"
lesson: "embedded-lasso-l1"
sourceId: 1415
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection/embedded-lasso-l1"
title: "嵌入式方法：正则化（Lasso L1）"
description: "使用 Lasso 回归（L1 正则化），它通过将系数收缩到零来自动实现特征选择。"
order: 10
plots: ["plots/1415-0.json"]
sourceHash: "d0f8d7a8063376a727db47daae1584f026bcbc6d87a755d0645cea8a5a9c44b9"
sourceCorrections: []
---

嵌入 (embedding)式方法将特征选择过程直接融入模型训练阶段。与独立评估特征的过滤方法或在不同子集上重复训练模型的包装方法不同，嵌入式技术在学习模型参数 (parameter)的同时进行特征选择。这种方法在计算上比包装方法更高效，并且通常能更好地反映特征之间的关联，优于过滤方法。

L1正则化 (regularization)，通常称为Lasso（最小绝对收缩与选择算子），是一种嵌入式特征选择技术。这种方法在学习模型参数的同时进行特征选择，尤其适用于线性模型，例如线性回归或逻辑回归。

### 了解正则化 (regularization)

在具体查看Lasso之前，我们先简要介绍一下正则化。在机器学习 (machine learning)中，尤其是在线性模型中，我们的目标是找到能使损失函数 (loss function)（如回归中的均方误差）最小化的模型参数 (parameter)（系数）。然而，仅仅最小化训练误差可能导致模型过于复杂，从而在训练数据上过拟合 (overfitting)，在未见过的数据上表现不佳。

正则化会向损失函数添加一个惩罚项。这个惩罚项抑制模型变得过于复杂，通常通过限制系数的大小来做到。现在模型试图最小化：


$$
\text{损失函数} + \text{惩罚项}
$$


### Lasso (L1) 惩罚

Lasso采用系数向量 (vector)的$L1$范数作为其惩罚项。对于一个具有$p$个特征、系数为$\beta = (\beta_1, \beta_2, ..., \beta_p)$的线性模型，其$L1$范数是系数绝对值的总和：


$$
||\beta||_1 = \sum_{j=1}^{p} |\beta_j|
$$


Lasso回归的目标函数（最小化带有L1惩罚的均方误差）变为：


$$
\underset{\beta}{\operatorname{argmin}} \left( \sum_{i=1}^{n} (y_i - \sum_{j=1}^{p} x_{ij}\beta_j)^2 + \alpha \sum_{j=1}^{p} |\beta_j| \right)
$$


这里：

- $n$ 代表样本数量。
- $p$ 代表特征数量。
- $y_i$ 代表第$i$个样本的目标值。
- $x_{ij}$ 代表第$i$个样本的第$j$个特征的值。
- $\beta_j$ 代表第$j$个特征的系数。
- $\alpha$ (alpha) 是正则化 (regularization)强度参数 (parameter)。它控制着很好地拟合数据（最小化平方误差和）与保持系数较小（最小化L1惩罚）之间的权衡。

### Lasso 如何实现特征选择

L1惩罚的主要特点是它倾向于将一些系数收缩到*恰好为零*。这发生在优化损失函数 (loss function)时，因为L1约束的形状（在二维中是菱形，在更高维度中是多维等价物）导致了这种情况。系数被收缩到零的特征会实际从模型中移除。

- **更高的$\alpha$：** 增加对大系数的惩罚。更多的系数可能被收缩到零，形成一个更简单的模型，选择的特征也更少。
- **更低的$\alpha$：** 减少惩罚。系数收缩程度较小，更少（或没有）系数可能恰好变为零。如果$\alpha = 0$，Lasso退化为标准线性回归（无正则化 (regularization)）。

这种特性使Lasso成为一种嵌入 (embedding)式特征选择方法：拟合模型的过程本身通过仅对有用特征赋予非零系数来选择它们。



![Lasso 系数路径与 Alpha 的关系（对数刻度）](plots/1415-0.json)



> 图示了随着正则化强度（`alpha`）的增加，Lasso系数如何变化。请注意，一些系数（特征2、特征3、特征4）在不同的`alpha`值下恰好收缩到零，实际将其从模型中移除。特征1保留时间更久，但也会收缩。

### 在 Scikit-learn 中使用 Lasso

Scikit-learn 为回归任务提供 `Lasso` 类。你可以像拟合任何其他 Scikit-learn 模型一样拟合它，然后查看 `coef_` 属性，以了解哪些特征被赋予了非零系数。

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# 生成模拟数据
X, y = make_regression(n_samples=200, n_features=20, n_informative=10, noise=15, random_state=42)

# 引入一些不相关特征（使最后5个特征冗余）
X[:,-5:] = X[:,:5] * np.random.uniform(0.9, 1.1, size=(X.shape[0], 5))

# 拆分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征缩放（对 Lasso 很重要！）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化并拟合 Lasso 模型
# 调整 alpha 以控制稀疏性（alpha 越高 = 更多零）
alpha_value = 1.0
lasso = Lasso(alpha=alpha_value, random_state=42)
lasso.fit(X_train_scaled, y_train)

# 检查系数
print(f"Lasso 系数 (alpha={alpha_value}):")
print(lasso.coef_)

# 识别选定的特征（非零系数）
selected_features_mask = lasso.coef_ != 0
n_selected_features = np.sum(selected_features_mask)
print(f"\n选择的特征数量：{n_selected_features} / {X_train_scaled.shape[1]}")

# 获取选定特征的索引
selected_indices = np.where(selected_features_mask)[0]
print(f"选定特征的索引：{selected_indices}")

# 你现在可以使用这些选定的特征进行后续建模
# X_train_selected = X_train_scaled[:, selected_features_mask]
# X_test_selected = X_test_scaled[:, selected_features_mask]
```

**重要提示：** Lasso，与许多正则化 (regularization)模型和基于距离的算法一样，对输入特征的尺度敏感。在应用Lasso之前，对数据进行缩放（例如，使用`StandardScaler`或`MinMaxScaler`）是常规做法。

### 将 Lasso 用于通用特征选择

你不必仅仅使用Lasso来构建最终的预测模型。它可以在更大的机器学习 (machine learning)流程中用作特征选择步骤。Scikit-learn 的 `SelectFromModel` 元转换器简化了此操作。它根据重要性权重 (weight)（如Lasso的系数或树模型的特征重要性）选择特征。

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # 分类示例

# --- 假设 X_train_scaled, y_train 已准备就绪 ---
# 分类示例 - 使用带有 L1 惩罚的 LogisticRegression

# 创建一个流程：Lasso（作为选择器） -> 最终模型
# 使用 LassoCV 或 LogisticRegressionCV 在内部找到最佳 alpha，或手动设置 alpha
lasso_selector_model = Lasso(alpha=1.0, random_state=42) # 或用于分类的 LogisticRegression(penalty='l1', C=1.0, solver='liblinear')

selector = SelectFromModel(estimator=lasso_selector_model, threshold="median") # 选择系数大于中位数系数的特征

# 以逻辑回归作为最终估计器的示例流程
# 对于分类，你可以直接在 SelectFromModel 中使用 LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
# 对于回归，你可以接着使用另一个回归器，如 Ridge 或 LinearRegression
pipeline = Pipeline([
  ('scaler', StandardScaler()), # 确保缩放发生在流程中
  ('feature_selection', selector),
  ('classification', LogisticRegression(random_state=42)) # 示例最终模型
])

# 拟合流程（选择器和分类器都已训练）
# pipeline.fit(X_train, y_train) # 假设 y_train 是逻辑回归的类别数据

# 现在 'pipeline' 代表包含 L1 特征选择的完整过程
```

### Lasso (L1) 特征选择的优点与缺点

**优点：**

- **自动特征选择：** 直接内置于模型拟合过程。
- **处理共线性：** 如果多个特征高度相关，Lasso 倾向于随意选择其中一个，并将其他特征收缩到零，这可能是有益的。
- **生成稀疏模型：** 所得模型通常更简单，也更容易解释，因为许多系数为零。
- **计算高效：** 通常比包装方法更快，特别是对于高维数据。

**缺点：**

- **对$\alpha$敏感：** 所选特征的数量在很大程度上取决于正则化 (regularization)参数 (parameter)$\alpha$的选择。通常需要交叉验证（例如，使用`LassoCV`）来找到一个好的值。
- **可能难以处理相关特征组：** 尽管它通过选择一个特征来处理共线性，但这可能不是“最佳”的一个，有时如果相关特征代表一个有意义的组，你可能更希望保留它们。弹性网络正则化（L1和L2的组合）有时可以更好地处理这个问题。
- **性能：** 如果大多数特征确实与目标变量相关，Lasso收缩系数的趋势可能会损害预测性能，相比之下，像岭（L2）正则化这样的方法会将系数收缩到零附近，但很少使它们恰好为零。

Lasso 提供了一种强大且广泛使用的嵌入 (embedding)式特征选择方法，在处理许多特征可能不相关的高维数据集时特别有效。请记住缩放你的数据并仔细调整`alpha`参数以获得最佳结果。

## 参考资料

- [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x) — Robert Tibshirani (1996)
  Journal: Journal of the Royal Statistical Society, Series B (Methodological); Publisher: Wiley; Volume: 58; Pages: 267-288; DOI: [10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
  介绍了用于线性模型中同时进行系数收缩和变量选择的Lasso方法。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: Chapters 3 & 18; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本经典教科书，详细介绍了Lasso、Ridge和Elastic Net正则化，包括其几何属性，涵盖理论和实践。
- [1.1. Linear Models - Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso) — scikit-learn developers (2024)
  `sklearn.linear_model.Lasso`的官方文档，详细说明了其参数、使用场景以及Python中L1正则化的代码示例。
- [3.13. Feature selection - SelectFromModel](https://scikit-learn.org/stable/modules/feature_selection.html#selectfrommodel) — scikit-learn developers (2024)
  `sklearn.feature_selection.SelectFromModel`的官方文档，展示了如何使用模型导出的重要性（如Lasso系数）进行特征选择。
