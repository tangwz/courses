---
course: "intro-feature-engineering"
chapter: "feature-selection"
lesson: "filter-correlation-analysis"
sourceId: 1404
sourceUrl: "https://apxml.com/zh/courses/intro-feature-engineering/chapter-6-feature-selection/filter-correlation-analysis"
title: "过滤方法：相关性分析"
description: "识别并移除高度相关的特征，以降低多重共线性。"
order: 5
plots: ["plots/1404-0.json"]
sourceHash: "9b0f1c987386900bddbe024c141cb7ec2c0fdac48696b1116980003ed15c98e3"
sourceCorrections: []
---

如本章前面所述，过滤方法根据特征的固有属性对其进行评估，不依赖于任何特定的机器学习 (machine learning)模型。此类方法中一个常用的技术是**相关性分析**，专门用于识别并可能移除冗余特征。

当多个特征传达非常相似的信息时，就会出现冗余。例如，如果您有“摄氏温度”和“华氏温度”这两个特征，它们是完全相关的，其中一个就是冗余的。包含高度相关的特征有时会使某些模型（如因多重共线性导致的线性回归）变得不稳定，并增加不必要的复杂性，而无助于提升预测能力。相关性分析可帮助我们发现这些关系。

### 理解相关性

我们通常使用**皮尔逊相关系数**（表示为$r$）来衡量两个数值特征之间的*线性*关系。该系数的取值范围从-1到+1：

- **+1：** 完美正线性关系（一个特征增加时，另一个按比例增加）。
- **-1：** 完美负线性关系（一个特征增加时，另一个按比例减少）。
- **0：** 没有线性关系。

介于0和$\pm 1$之间的值表示线性关联的强度和方向。一个接近$\pm 1$的值表明存在强的线性关系，而接近0的值则表示线性关系微弱或不存在。

### 计算和可视化相关性

在Python中，Pandas库使得使用`.corr()`方法计算DataFrame中所有数值列之间的成对相关性变得简单直接。

```python
import pandas as pd
import numpy as np

# 假设 'df' 是包含数值特征的 DataFrame
# 如果 df 包含混合类型，只选择数值列
numerical_df = df.select_dtypes(include=np.number)

# 计算相关矩阵
correlation_matrix = numerical_df.corr()

# 显示矩阵（可选）
print(correlation_matrix)
```

结果是一个方阵，行`i`和列`j`处的条目是特征`i`与特征`j`之间的相关系数。对角线元素总是1（特征与自身的相关性）。

虽然该矩阵包含原始数值，但将其可视化为热图通常能提供更清晰的视图，尤其是在处理大量特征时。我们可以使用Matplotlib或Seaborn等库，或通过Plotly生成交互式图表。



![特征相关性热图](plots/1404-0.json)



> 示例相关性热图，呈现四个特征之间线性关系的强度和方向。红色表示正相关，蓝色表示负相关，浅色表示弱相关。

### 识别并移除相关特征

核心思路是识别具有高绝对相关性（例如 $|r| > 0.8$ 或 $|r| > 0.9$）的特征对，然后决定移除哪个或哪些特征。

1. **设置阈值：** 选择一个相关系数阈值，高于此值的特征被认为是高度相关的。此选择有些主观，并取决于具体问题和建模技术。常用阈值范围为0.8到0.95。
2. **识别特征对：** 查找所有绝对相关性超过阈值的特征对。
3. **选择要剔除的特征：** 对于每个已识别的特征对，您需要一个策略来决定移除哪个特征。避免同时移除两者。常用策略包括：
   - **与目标的关联：** 如果您可以访问目标变量（并且在纯过滤上下文 (context)中小心避免不当信息泄露），可以检查特征对中哪个特征与目标的关联性更强，并保留该特征。
   - **方差：** 保留方差更大的特征（可能包含更多信息），前提是特征已进行缩放。
   - **缺失值：** 优先保留缺失值较少的特征。
   - **专业知识：** 利用对特征的理解来决定哪个更具根本性或更易于解释/衡量。
   - **任意选择：** 如果没有其他适用标准，您可以简单地移除在DataFrame排序中靠后的特征。

以下是根据阈值查找和收集要剔除特征的编程方法：

```python
import pandas as pd
import numpy as np

# 假设 'correlation_matrix' 已如上计算
# 假设 'threshold' 已定义 (例如, threshold = 0.9)

# 使用绝对相关性
abs_corr_matrix = correlation_matrix.abs()

# 获取相关矩阵的上三角部分（不包括对角线）
upper_triangle = abs_corr_matrix.where(np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool))

# 查找相关性高于阈值的特征
to_drop = set() # 使用集合避免重复
for column in upper_triangle.columns:
    highly_correlated_with = upper_triangle.index[upper_triangle[column] > threshold].tolist()
    if highly_correlated_with:
        # 基本策略：如果当前列与任何之前的列高度相关，则将其剔除
        # 这里可以实现更复杂的策略
        # 为简单起见，如果它与任何已检查的特征（索引 < 列）高度相关，则收集其列名
        if any(upper_triangle[column] > threshold):
             to_drop.add(column) # 示例：添加特征对中的第二个特征

print(f"考虑剔除的特征（基于阈值 {threshold}）：{list(to_drop)}")

# 从原始 DataFrame 中剔除选定的特征
# df_reduced = df.drop(columns=list(to_drop))
# print(f"原始特征数量：{df.shape[1]}")
# print(f"相关性过滤后的特征数量：{df_reduced.shape[1]}")
```

*注意：* 决定剔除特征对中*哪个*特征的逻辑需要仔细考虑。上面的示例使用了一种简单方法；更精细的逻辑可能涉及比较特征与目标的关联性或使用其他指标。请小心，避免因迭代不当而意外地剔除高度相关对中的两个特征。使用上三角有助于避免重复检查和自相关。

### 局限性

尽管相关性分析作为过滤方法很有用，但它存在局限性：

- **仅限线性关系：** 皮尔逊相关性主要衡量线性关联。它可能会遗漏特征之间强的非线性关系。两个特征可能以非线性方式强关联，但其相关系数接近零。
- **数值特征：** 标准相关性分析直接适用于数值特征。评估类别特征之间或数值与类别特征之间的冗余需要采用不同的技术（例如，卡方检验、方差分析F值、互信息）。
- **忽略目标变量：** 作为一种过滤方法，它通常评估特征之间的关系，而不直接考虑这些关系如何影响目标变量的预测（这与封装或嵌入 (embedding)方法不同）。移除某个特征可能不经意地丢弃掉一些信息，这些信息尽管与另一个特征存在关联，但对预测目标有其独特的贡献。

尽管存在这些局限性，但检查并移除高度相关的数值特征是特征选择过程中一个标准且通常有益的步骤，通过减少特征冗余，有助于创建更简单、更稳定的模型。

## 参考资料

- [Feature Engineering and Selection: A Practical Approach for Developers and Data Scientists](https://www.taylorfrancis.com/books/9780429177892) — Max Kuhn and Kjell Johnson (2019)
  Publisher: Chapman and Hall/CRC; DOI: [10.1201/9780429177892](https://doi.org/10.1201/9780429177892)
  本书讨论了过滤方法、相关性分析以及处理特征冗余的实用策略。
- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Rob Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38918-2](https://doi.org/10.1007/978-3-031-38918-2)
  为理解相关性、多重共线性及其对线性模型的影响提供了统计基础。
- [pandas.DataFrame.corr](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) — pandas development team (2024)
  Pandas官方文档，介绍如何计算DataFrame列之间的皮尔逊相关系数。
- [Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html) — scikit-learn developers (2024)
  对各种特征选择方法进行了概述，包括过滤方法及其在机器学习流程中的作用。
