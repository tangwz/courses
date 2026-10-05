# 使用Python构建回归模型

来源：[原文](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-6-introduction-regression-analysis/building-regression-models-python)

[返回章节目录](README.md) · [返回课程目录](../README.md)

线性回归包含模型结构($y = \beta_0 + \beta_1 x + \epsilon$)、最小二乘估计、系数解释和评估指标等基本概念。Python常用于实现这些模型。两个流行的库在这一方面占据主导地位：`statsmodels`和`scikit-learn`。每个库都有其优点，同时理解它们能为回归任务提供一套多功能的工具。

`statsmodels`因其对统计推断和详细模型分析的重视而常受青睐，其输出与R等统计软件中常见的结果相似。`scikit-learn`是更广泛的机器学习 (machine learning)体系的一部分，为各种算法提供了一致的API，使其适合构建预测流程。

### 使用Statsmodels进行回归

`statsmodels`提供了全面的工具，用于估计多种不同的统计模型，并进行统计检验和数据查看。它的`formula.api`子模块允许使用基于字符串的公式来指定模型，类似于R，这相当直观。

我们用一个简单的线性回归例子来说明。假设我们有一个pandas DataFrame `df`，其中包含'TargetVariable' (y) 和 'PredictorVariable' (x) 两列。

```python
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 示例数据（请替换为您的实际数据）
data = {'PredictorVariable': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'TargetVariable': np.array([2.5, 3.1, 4.5, 5.2, 6.8, 7.1, 8.5, 9.2, 10.1, 11.5])}
df = pd.DataFrame(data)

# 使用公式语法定义模型
# 'TargetVariable ~ PredictorVariable' 表示将 TargetVariable 建模为 PredictorVariable 的函数
# statsmodels 自动包含截距项
model_formula = 'TargetVariable ~ PredictorVariable'
model = smf.ols(formula=model_formula, data=df)

# 将模型拟合到数据
results = model.fit()

# 打印详细摘要
print(results.summary())
```

`results.summary()` 方法提供了一个丰富的结果输出，包含几项重要的信息：

1. **模型拟合度：** R平方 ($R^2$) 和调整R平方，表示模型解释的方差比例。F统计量及其相关的p值，用于检验模型的整体显著性。
2. **系数表：**
   - `coef`：截距项($\hat{\beta}_0$)和`PredictorVariable`的系数($\hat{\beta}_1$)的估计值。
   - `std err`：估计值的标准误差，表示其变动性。
   - `t`：t统计量，用于检验每个系数是否显著异于零。
   - `P>|t|`：与t统计量相关的p值。小的p值（通常小于0.05）表明系数具有统计显著性。
   - `[0.025 0.975]`：系数的95%置信区间。
3. **假设检验：** 关于多重共线性、异方差性等潜在问题的信息（残差正态性的Omnibus、Jarque-Bera检验，条件数）。

当你主要目标是理解数据中关系的统计特性和显著性时，这种详细摘要是`statsmodels`的一大优势。

### 使用Scikit-learn进行回归

`scikit-learn`提供了一个更简洁的接口，在不同机器学习 (machine learning)模型之间保持一致，这在将回归集成到更大的工作流程或与其他预测算法比较时很有利。

下面是使用`scikit-learn`执行简单线性回归的方法：

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 示例数据（与之前相同）
data = {'PredictorVariable': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'TargetVariable': np.array([2.5, 3.1, 4.5, 5.2, 6.8, 7.1, 8.5, 9.2, 10.1, 11.5])}
df = pd.DataFrame(data)

# 准备数据
# X 需要是一个二维数组（或DataFrame）；y 是一个一维数组（或Series）
X = df[['PredictorVariable']] # 注意双括号使其保持为DataFrame
y = df['TargetVariable']

# 初始化模型
sk_model = LinearRegression()

# 拟合模型
sk_model.fit(X, y)

# 获取估计的系数
intercept = sk_model.intercept_ # Beta_0
coefficient = sk_model.coef_[0]  # Beta_1（它是一个数组）

print(f"Intercept (beta_0): {intercept:.4f}")
print(f"Coefficient (beta_1): {coefficient:.4f}")

# 进行预测
y_pred = sk_model.predict(X)

# 评估模型
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R-squared (R2): {r2:.4f}")
```

关于`scikit-learn`方法的重要说明：

- **数据格式：** `scikit-learn`通常要求特征 (`X`) 是一个二维数组结构（例如，形状为 `(样本数, 特征数)` 的DataFrame或NumPy数组），而目标 (`y`) 则是一个一维数组结构（例如，pandas Series或形状为 `(样本数,)` 的NumPy数组）。
- **API：** `fit()` 方法训练模型，`predict()` 方法对新数据生成预测。模型参数 (parameter)（如系数）作为拟合模型对象的属性存储（例如，`sk_model.intercept_`，`sk_model.coef_`）。
- **评估：** 模型评估通常使用 `sklearn.metrics` 模块中的函数完成，这需要真实目标值和模型的预测值。

与`statsmodels`相比，`scikit-learn`提供的内置统计摘要输出较少。它的优势在于其一致的API、易于集成到机器学习流程（包括数据预处理、交叉验证和模型选择），以及对多种算法的支持。

### 扩展到多元线性回归

这两个库都可以轻松处理多个预测变量。

**Statsmodels：** 只需在公式字符串中添加更多变量即可。

```python
# 假设 df 包含 'Predictor1'、'Predictor2'、'TargetVariable' 列
model_formula_multi = 'TargetVariable ~ Predictor1 + Predictor2'
multi_model_sm = smf.ols(formula=model_formula_multi, data=df).fit()
# print(multi_model_sm.summary()) # 摘要包括 Predictor1 和 Predictor2 的系数
```

**Scikit-learn：** 在特征DataFrame `X` 中包含更多列。

```python
# 假设 df 包含 'Predictor1'、'Predictor2'、'TargetVariable' 列
X_multi = df[['Predictor1', 'Predictor2']]
y = df['TargetVariable']

multi_model_sk = LinearRegression()
multi_model_sk.fit(X_multi, y)

# 系数将是一个数组，包含 Predictor1 和 Predictor2 的值
# print(multi_model_sk.coef_)
# print(multi_model_sk.intercept_)
```

### 选择合适的工具

- 当你的重点是统计推断、系数假设检验、理解置信区间以及获取详细诊断摘要时，请使用 **`statsmodels`**。它擅长解释当前数据中的关系。
- 当你的主要目标是预测、将回归集成到机器学习 (machine learning)流程、执行交叉验证或使用一致的接口将回归与其他预测模型进行比较时，请使用 **`scikit-learn`**。

熟悉这两个库能提供灵活性。你可以使用`statsmodels`进行初步的数据分析和模型理解，然后转向`scikit-learn`来构建一个可部署的预测模型。接下来的实践部分将为你提供直接将这些工具应用于数据集的经验。

## 参考资料

- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  为统计学习提供了坚实的基础，涵盖了线性回归理论、模型评估和机器学习视角的解释，现在包含Python应用。
- [Statsmodels Documentation: Ordinary Least Squares (OLS)](https://www.statsmodels.org/stable/regression.html#ordinary-least-squares-ols) — The Statsmodels Developers (2024)
  statsmodels库的官方文档，提供了使用其统计模型（包括OLS）进行推断和综合输出分析的详细指南。
- [scikit-learn User Guide: 1.1. Linear Models](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares-and-ridge-regression) — The scikit-learn developers (2024)
  Publisher: scikit-learn project
  scikit-learn线性模型的官方用户指南，解释了`LinearRegression`的API及其在机器学习工作流中的应用。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本实用的教科书，用于使用`scikit-learn`构建和评估机器学习模型，其中包含线性回归和更广泛的机器学习概念的实践示例。

---

[上一节](06-%E5%A4%9A%E5%85%83%E7%BA%BF%E6%80%A7%E5%9B%9E%E5%BD%92%E6%A6%82%E8%BF%B0.md) · [下一节](08-%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5%EF%BC%9A%E6%8B%9F%E5%90%88%E4%B8%8E%E8%AF%84%E4%BC%B0%E7%BA%BF%E6%80%A7%E6%A8%A1%E5%9E%8B.md)
