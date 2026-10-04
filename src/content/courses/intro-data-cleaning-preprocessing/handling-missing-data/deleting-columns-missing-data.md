---
course: "intro-data-cleaning-preprocessing"
chapter: "handling-missing-data"
lesson: "deleting-columns-missing-data"
sourceId: 3969
sourceUrl: "https://apxml.com/zh/courses/intro-data-cleaning-preprocessing/chapter-2-handling-missing-data/deleting-columns-missing-data"
title: "策略二：删除列"
description: "理解在何种情况下删除整个特征（列）可能是合适的。"
order: 5
plots: ["plots/3969-0.json"]
sourceHash: "fbf40ad09d642154497e6177c24a0a1cbd508bbc955f50ee740ce8866edda8ee"
sourceCorrections: []
---

有时，缺失数据不仅仅分散在少数几行中；它可能严重影响数据集中的整个列（特征）。在上一节中，我们讨论了删除含有缺失值的行。现在，我们将考虑一种不同的、有时是必要的，方法：删除整个列。

这种策略通常用于特定列的缺失条目比例非常高的情况。为什么？因为如果列中的大部分数据缺失，该特征可能不会为您的分析或机器学习 (machine learning)模型提供太多有用信息。试图在这种列中填补大量缺失值（填充，我们将在下一节讨论）可能会引入明显的偏差或噪声，这可能会弊大于利。

### 何时考虑删除列

没有严格的规定，但一种常见做法是，如果列中有很大一部分值缺失，就考虑删除该列。多少才算“大”？这通常取决于具体情况、数据集的大小以及该特征的重要性。一些分析师使用阈值，例如50%、60%甚至70%的缺失值，作为考虑删除列的起点。

假设你有一个包含客户信息的数据集，其中包含‘传真号码’列。如今，可能很少有客户提供此信息，导致该列可能存在95%的缺失值。对于大多数分析（如预测购买行为），此列不太可能有用，并且是删除的合适选择。

我们来将其可视化。假设我们计算数据集中几列的缺失值百分比：



![每列缺失值百分比](plots/3969-0.json)



> 在这个例子中，‘Last\_Login\_Device’列有85%的缺失数据。这个高百分比使其成为一个很适合删除的选项。

### 如何删除列

大多数数据分析库都提供直接的函数来删除列。例如，使用流行的Python pandas库时，你通常会识别你希望删除的列名，然后使用类似`drop()`的函数，指定列名并指明你正在删除列（而非行）。

```python
# 使用pandas（Python）的例子
# 假设'df'是你的DataFrame，你想删除'Column_X'

# 计算缺失百分比
missing_percentages = df.isnull().mean() * 100

# 识别要删除的列（例如，阈值为70%）
columns_to_drop = missing_percentages[missing_percentages > 70].index

# 删除已识别的列
df_cleaned = df.drop(columns=columns_to_drop)

print(f"原始列: {df.columns.tolist()}")
print(f"删除后的列: {df_cleaned.columns.tolist()}")
```

### 优点与缺点

- **优点：** 简单。完全删除列消除了该特征的缺失数据问题，无需复杂的填充方法。它还可以通过降低数据集的维度来加快后续处理。
- **优点：** 避免通过填充引入潜在有偏差的数据，尤其当缺失率非常高时。
- **缺点：** 信息损失。这是最大的弊端。即使一列有许多缺失值，那些确实存在的值也可能包含有价值的信息。删除该列意味着完全失去这些信息。
- **缺点：** 对其他特征的潜在影响。有时，缺失模式本身可能提供信息，或者即使稀疏，该特征在与其他特征一起考虑时也可能很重要。

### 做出决定

删除列是一种权衡。你简化了数据集，但有失去潜在有用信息的风险。在删除列之前：

1. **评估百分比：** 量化 (quantization)实际缺失的数据量。
2. **评估特征的重要性：** 考虑此特征与你的分析目标有多大关联。它在理论上重要吗？即使稀疏，它也能有用吗？
3. **考虑替代方案：** 考虑填充方法（下一节将讨论）是否可行且损害较小，即使缺失百分比有些高。

选择删除列是比删除单个行更激进的步骤。当特征中缺失数据的比例非常大，以至于该特征本身不太可能对分析做出有意义的贡献，或者当填充似乎太不可靠时，通常会采用这种方法。

## 参考资料

- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and Jupyter](https://www.oreilly.com/library/view/python-for-data-analysis/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  Python数据操作的入门资源，涵盖使用pandas处理缺失数据和重构数据集的实用方法。
- [Flexible Imputation of Missing Data, Second Edition](https://doi.org/10.1201/9780429492259) — Stef van Buuren (2018)
  Publisher: CRC Press; Pages: 444; DOI: [10.1201/9780429492259](https://doi.org/10.1201/9780429492259)
  从统计和方法论角度详细阐述了缺失数据问题，讨论了包括删除在内的各种处理策略的假设和影响。
- [pandas.DataFrame.drop](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html) — pandas development team (2024)
  用于从DataFrame中删除指定行或列的pandas函数官方文档。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQG_FRPHKQdtGAwic1KE16snmaFW3mJ79Z96EGXZI5YWA-bpigrGgVtz6PjHQKCrNx2AXIoV9gqlAA1wnEo8JoZtsEJJ-3bus-i5-UBkXMHrab1Bu_6lBF6Cen8W99Xo0vTCWh_B28O3rg4R3tMeWllrTVx0iX4IGIx3Ep1Tm-W-y8jQarVRCSE=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  讨论了数据预处理技术，包括缺失值处理，作为为机器学习模型准备数据的基本步骤。
