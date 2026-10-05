# 使用 Python 进行假设检验

来源：[原文](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation/hypothesis-tests-python)

[返回章节目录](README.md) · [返回课程目录](../README.md)

使用 Python 执行假设检验，`scipy.stats` 模块使其过程直接简便。该模块提供了一套完善的函数，用于执行常见的统计检验。将涵盖的检验有 T 检验、卡方检验，以及对方差分析的简要介绍。

### 使用 SciPy 进行 T 检验

T 检验主要用于比较均值。SciPy 提供了这些检验的主要变体函数。请记住，这些检验通常假定数据服从正态分布，尽管它们对违反假设的情况有一定抵抗力，尤其在样本量较大时。

#### 单样本 T 检验

此检验检查单个样本的均值是否与已知或假设的总体均值 ($\mu_0$) 显著不同。我们使用 `scipy.stats.ttest_1samp` 函数。

假设我们收集了新系统中特定类型查询的处理时间（毫秒）数据。我们想检验平均处理时间是否与目标基准 150 毫秒显著不同。

```python
import numpy as np
from scipy import stats

# 样本处理时间（例如，来自25次查询）
processing_times = np.array([145, 155, 162, 148, 153, 160, 149, 151, 158, 147,
                           154, 163, 150, 146, 157, 152, 149, 161, 156, 150,
                           148, 159, 155, 164, 151])

# 假设的总体均值
mu_0 = 150

# 执行单样本 T 检验
t_statistic, p_value = stats.ttest_1samp(a=processing_times, popmean=mu_0)

print(f"样本均值: {np.mean(processing_times):.2f}")
print(f"T 统计量: {t_statistic:.4f}")
print(f"P 值: {p_value:.4f}")

# 解释
alpha = 0.05
if p_value < alpha:
    print(f"拒绝原假设 (p={p_value:.4f})。平均处理时间与 {mu_0} 毫秒显著不同。")
else:
    print(f"未能拒绝原假设 (p={p_value:.4f})。没有足够证据表明平均处理时间与 {mu_0} 毫秒不同。")
```

输出提供了 T 统计量和 P 值。T 统计量衡量样本均值与假设均值相距多少标准误差。P 值告诉我们，假设原假设为真，观察到与计算所得样本均值一样极端或更极端的样本均值的概率。我们将 P 值与选择的显著性水平 ($\alpha$) 进行比较，以做出判断。

#### 独立双样本 T 检验

此检验比较两个独立组的均值，以查看它们是否显著不同。例如，我们可能比较使用两种不同界面（A 组 vs. B 组）用户的绩效得分。我们使用 `scipy.stats.ttest_ind`。

```python
import numpy as np
from scipy import stats
import plotly.graph_objects as go

# 两个独立组的样本绩效得分
scores_A = np.array([85, 90, 78, 88, 92, 81, 87, 89, 80, 84])
scores_B = np.array([75, 82, 70, 79, 85, 72, 77, 81, 74, 78])

# 可选：可视化分布
fig = go.Figure()
fig.add_trace(go.Box(y=scores_A, name='Group A', marker_color='#1f77b4')) # blue
fig.add_trace(go.Box(y=scores_B, name='Group B', marker_color='#ff7f0e')) # orange
fig.update_layout(title='Performance Scores by Group', yaxis_title='Score', height=400, width=500)
# fig.show() # In a notebook environment

# 执行独立双样本 T 检验
# 默认情况下，它假定方差相等（如果 equal_var=False，则执行 Welch's T 检验）
t_statistic, p_value = stats.ttest_ind(a=scores_A, b=scores_B, equal_var=True)

print(f"A 组平均得分: {np.mean(scores_A):.2f}")
print(f"B 组平均得分: {np.mean(scores_B):.2f}")
print(f"T 统计量: {t_statistic:.4f}")
print(f"P 值: {p_value:.4f}")

# 解释
alpha = 0.05
if p_value < alpha:
    print(f"拒绝原假设 (p={p_value:.4f})。这两组的平均得分显著不同。")
else:
    print(f"未能拒绝原假设 (p={p_value:.4f})。没有足够证据表明平均得分不同。")
```



[交互图表：按组划分的绩效得分](https://apxml.com/zh/courses/probability-statistics-essentials-ml/chapter-5-hypothesis-testing-model-evaluation/hypothesis-tests-python#plot-hmt68t)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "按组划分的绩效得分",
    "yaxis": {
      "title": "得分"
    },
    "height": 400,
    "width": 500,
    "boxmode": "group"
  },
  "data": [
    {
      "type": "box",
      "y": [
        85,
        90,
        78,
        88,
        92,
        81,
        87,
        89,
        80,
        84
      ],
      "name": "A 组",
      "marker": {
        "color": "#339af0"
      }
    },
    {
      "type": "box",
      "y": [
        75,
        82,
        70,
        79,
        85,
        72,
        77,
        81,
        74,
        78
      ],
      "name": "B 组",
      "marker": {
        "color": "#ff922b"
      }
    }
  ]
}
```

</details>



> 箱线图比较了 A 组和 B 组的绩效得分分布。A 组的平均得分似乎更高。

`ttest_ind` 函数返回 T 统计量和 P 值。参数 (parameter) `equal_var` 控制是否假定两组之间方差相等（Student's T 检验）或不相等（Welch's T 检验，较新版本的 SciPy 中的默认设置可能不同，请查阅文档）。Welch's T 检验常受到青睐，因为它不需要方差相等的假设。

#### 配对双样本 T 检验

此检验用于两个样本相关或配对的情况。这通常发生于在两种不同条件下测量同一受试者（例如，治疗前后，或同一人完成任务 A 与任务 B 的表现）。我们使用 `scipy.stats.ttest_rel`。

假设我们测量了 10 名参与者在完成训练模块前后的响应时间。我们想知道训练是否显著降低了他们的响应时间。

```python
import numpy as np
from scipy import stats

# 10 名参与者训练前后的响应时间（秒）
response_before = np.array([5.2, 4.8, 6.0, 5.5, 5.1, 5.8, 4.9, 5.4, 5.6, 5.0])
response_after = np.array([4.7, 4.5, 5.5, 5.0, 4.6, 5.2, 4.4, 4.9, 5.0, 4.7])

# 计算差异
differences = response_before - response_after

# 执行配对 T 检验
t_statistic, p_value = stats.ttest_rel(a=response_before, b=response_after)

print(f"平均差异（训练前 - 训练后）: {np.mean(differences):.2f}")
print(f"T 统计量: {t_statistic:.4f}")
print(f"P 值: {p_value:.4f}")

# 解释（假定为双尾检验，通常我们会对改进情况进行单尾检验）
# 对于单尾检验（例如，训练前 > 训练后），如果 t_statistic 为正，则将 p_value 除以 2
p_value_one_tailed = p_value / 2

alpha = 0.05
if p_value_one_tailed < alpha and t_statistic > 0: # Check direction for one-tailed
    print(f"拒绝原假设 (p_one_tailed={p_value_one_tailed:.4f})。训练后响应时间显著减少。")
else:
    print(f"未能拒绝原假设 (p_one_tailed={p_value_one_tailed:.4f})。未观察到响应时间显著减少。")
```

配对 T 检验本质上是对配对观测值之间的差异执行单样本 T 检验，检验平均差异是否显著异于零。

### 使用 SciPy 进行卡方检验

卡方 ($\chi^2$) 检验用于分类数据。它们帮助确定观察到的频率是否与预期频率显著不同，或者两个分类变量之间是否存在关联。

#### 卡方拟合优度检验

此检验确定分类数据的样本分布是否与预期分布匹配。例如，用户对不同网站布局选择的分布是否与假设分布（例如，均等偏好）匹配？我们使用 `scipy.stats.chisquare`。

假设我们假设用户会均等偏好（各 1/3）选择三种布局（A、B、C）。我们观察到 150 名用户的选择。

```python
import numpy as np
from scipy import stats

# 布局 A、B、C 选择的观察频率
observed_frequencies = np.array([60, 50, 40]) # Total = 150

# 基于均等偏好的预期频率（150 个用户 / 3 种布局）
expected_frequencies = np.array([50, 50, 50])

# 执行卡方拟合优度检验
chi2_statistic, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

print(f"观察频率: {observed_frequencies}")
print(f"预期频率: {expected_frequencies}")
print(f"卡方统计量: {chi2_statistic:.4f}")
print(f"P 值: {p_value:.4f}")

# 解释
alpha = 0.05
if p_value < alpha:
    print(f"拒绝原假设 (p={p_value:.4f})。观察到的分布与预期分布显著不同。")
else:
    print(f"未能拒绝原假设 (p={p_value:.4f})。观察到的分布与预期分布一致。")
```

该函数返回 $\chi^2$ 统计量和 P 值。显著结果表明，如果偏好确实均等，观察到的选择模式不太可能出现。

#### 卡方独立性检验

此检验检查两个分类变量是否关联或独立。例如，用户群（例如，'新用户'，'回访用户'）与偏好产品类别（例如，'电子产品'，'服装'，'家居'）之间是否存在关联？我们使用 `scipy.stats.chi2_contingency`。

我们需要一个包含两个变量观察频率的列联表（交叉制表）。

```python
import numpy as np
from scipy import stats
import pandas as pd

# 列联表：观察频率（例如，用户群 vs. 产品类别）
# 行：新用户，回访用户
# 列：电子产品，服装，家居
observed_table = pd.DataFrame({
    'Electronics': [50, 80],
    'Clothing': [70, 60],
    'Home': [30, 90]
}, index=['New', 'Returning'])

print("列联表（观察频率）：")
print(observed_table)

# 执行卡方独立性检验
chi2_statistic, p_value, dof, expected_table = stats.chi2_contingency(observed=observed_table)

print(f"\n卡方统计量: {chi2_statistic:.4f}")
print(f"P 值: {p_value:.4f}")
print(f"自由度: {dof}")
print("\n预期频率表:")
print(pd.DataFrame(expected_table, index=observed_table.index, columns=observed_table.columns).round(2))

# 解释
alpha = 0.05
if p_value < alpha:
    print(f"\n拒绝原假设 (p={p_value:.4f})。用户群与偏好产品类别之间存在显著关联。")
else:
    print(f"\n未能拒绝原假设 (p={p_value:.4f})。变量之间未发现显著关联。")
```

`chi2_contingency` 返回 $\chi^2$ 统计量、P 值、自由度（dof）以及在独立性原假设下的预期频率表。显著的 P 值表明这两个变量可能相关。

### 使用 SciPy 进行方差分析

方差分析 (ANOVA) 用于比较三个或更多组的均值。最简单的形式是单向方差分析，它分析一个分类因素对连续响应变量的影响。例如，比较三种不同广告活动的平均有效性得分。我们使用 `scipy.stats.f_oneway`。

```python
import numpy as np
from scipy import stats

# 三种不同广告活动的有效性得分
scores_campaign1 = np.array([7, 8, 6, 9, 7, 8])
scores_campaign2 = np.array([5, 6, 4, 5, 6, 5])
scores_campaign3 = np.array([9, 10, 8, 11, 9, 10])

# 执行单向方差分析
f_statistic, p_value = stats.f_oneway(scores_campaign1, scores_campaign2, scores_campaign3)

print(f"有效性得分 - 广告活动 1 平均值: {np.mean(scores_campaign1):.2f}")
print(f"有效性得分 - 广告活动 2 平均值: {np.mean(scores_campaign2):.2f}")
print(f"有效性得分 - 广告活动 3 平均值: {np.mean(scores_campaign3):.2f}")
print(f"\nF 统计量: {f_statistic:.4f}")
print(f"P 值: {p_value:.4f}")

# 解释
alpha = 0.05
if p_value < alpha:
    print(f"\n拒绝原假设 (p={p_value:.4f})。各广告活动之间的平均有效性得分存在显著差异。")
else:
    print(f"\n未能拒绝原假设 (p={p_value:.4f})。未发现平均有效性得分存在显著差异。")
```

`f_oneway` 函数返回 F 统计量（组间方差与组内方差之比）和 P 值。显著结果表明至少有一个组的均值与其他组不同。请注意，方差分析告诉你 *是否存在* 差异，但不能告诉你 *哪些* 具体组之间存在差异。如果方差分析结果显著，需要进行事后检验（如 Tukey HSD）来完成两两比较。

### 结果解读

无论执行何种检验，核心解读都依赖于 P 值：

- **如果 P 值 $\le \alpha$ (例如，0.05)：** 拒绝原假设 ($H_0$)。存在统计显著证据支持备择假设 ($H_1$)。
- **如果 P 值 > $\alpha$：** 未能拒绝原假设 ($H_0$)。没有足够的统计显著证据支持备择假设。

请记住，“未能拒绝 $H_0$”并不意味着 $H_0$ 为真，仅表示在所选显著性水平下，数据没有提供足够强的证据反对它。

这些 SciPy 函数提供了用于实现 Python 假设检验的强大工具集。选择正确的检验取决于数据类型（连续型、分类型）、比较的组数以及样本是独立还是配对。在将每个检验应用于数据之前，请务必考虑其潜在假设。

## 参考资料

- [\`scipy.stats\`](https://docs.scipy.org/doc/scipy/reference/stats.html) — SciPy Developers (2023)
  SciPy库中统计函数的官方参考资料，提供本节讨论的所有假设检验的详细描述、参数和示例。
- [An Introduction to Statistical Learning: With Applications in R](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani (2021)
  Publisher: Springer; DOI: [10.1007/978-1-0716-1418-1](https://doi.org/10.1007/978-1-0716-1418-1)
  为假设检验、t检验、卡方检验和ANOVA提供了全面的理论基础，对于理解Python实现背后的统计原理至关重要。
- [Practical Statistics for Data Scientists: 50 Essential Concepts](https://www.oreilly.com/library/view/practical-statistics-for/9781492072935/) — Peter Bruce, Andrew Bruce, and Peter Gedeck (2020)
  Publisher: O'Reilly Media
  提供将统计方法（包括使用Python进行假设检验）应用于数据科学和机器学习场景的实用指导。

---

[上一节](06-%E6%96%B9%E5%B7%AE%E5%88%86%E6%9E%90%20%28ANOVA%29%20%E6%A6%82%E8%BF%B0.md) · [下一节](08-%E5%AE%9E%E8%B7%B5%EF%BC%9A%E5%B0%86T%E6%A3%80%E9%AA%8C%E5%BA%94%E7%94%A8%E4%BA%8E%E6%A0%B7%E6%9C%AC%E6%95%B0%E6%8D%AE.md)
