---
course: "data-visualization-matplotlib-seaborn"
chapter: "seaborn-plotting-categorical-data"
lesson: "seaborn-categorical-boxplot"
sourceId: 1202
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-6-seaborn-plotting-categorical-data/seaborn-categorical-boxplot"
title: "分类变量的箱线图 (boxplot)"
description: "应用 Seaborn 的箱线图来比较不同类别的数据分布。"
order: 4
plots: ["plots/1202-0.json"]
sourceHash: "2eb8dbb322d45005acdbf190d7636ccb0092ccd720adeab81d08cc39a91252cf"
sourceCorrections: []
---

虽然条形图能让我们了解数值变量在不同类别中的集中趋势（如均值或中位数），但它们却不能很好地说明数据在每个类别中是如何分布的。数值是紧密集中，还是广泛散布？有很多异常值吗？为了回答这些问题并比较整体数据分布情况，我们可以使用箱线图。

箱线图（或称箱须图）简洁地展示了数据集的分布概况。它呈现了五个重要统计量：

1. **中位数 (Q2):** 数据的中间值（50th 百分位数），由箱体内的线表示。
2. **下四分位数 (Q1):** 25% 的数据低于此值（25th 百分位数）。这是箱体的底部边缘。
3. **上四分位数 (Q3):** 75% 的数据低于此值（75th 百分位数）。这是箱体的顶部边缘。
4. **四分位距 (IQR):** Q1 和 Q3 之间的范围 (IQR = Q3 - Q1)。箱体本身代表四分位距，包含中间 50% 的数据。
5. **须（或触须）:** 从箱体延伸出的线，通常显示 Q1 和 Q3 之外 1.5 倍四分位距内的数据范围。超出须的数值常被视为潜在异常值，并单独绘制。

Seaborn 的 `boxplot` 函数专门用于创建箱线图，使得比较不同类别的数据分布变得容易。

### 使用 `seaborn.boxplot` 创建箱线图

基本语法包括指定一个轴（通常是 `x`）上的分类变量，另一个轴（通常是 `y`）上的数值变量，以及使用 `data` 参数 (parameter)指定包含数据的 DataFrame。

我们来使用 Seaborn 自带的常用“tips”数据集。我们可以比较每周各天的总账单金额的分布情况。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例数据集
tips = sns.load_dataset("tips")

# 创建箱线图
plt.figure(figsize=(8, 5)) # 调整图表大小以提高可读性
sns.boxplot(x="day", y="total_bill", data=tips, palette=["#74c0fc", "#ffc078", "#8ce99a", "#ffc9c9"])

# 添加标题和标签（可选但建议）
plt.title("每日总账单金额分布")
plt.xlabel("星期")
plt.ylabel("总账单 ($)")

# 显示图表
plt.show()
```



![每日总账单金额分布](plots/1202-0.json)



> 使用 Seaborn 的 `boxplot` 函数绘制的每周各天总账单金额分布。

### 解读箱线图

从上图我们可以得出一些观察结果：

- **中位数：** 每个箱体内的线表示总账单的中位数。周六和周日的总账单中位数通常高于周四和周五。
- **分布（四分位距）：** 箱体的高度（四分位距）表明了中间 50% 账单的分布情况。周六的四分位距似乎更大，表示账单金额的变动性比周四更大。
- **须：** 须显示了常见账单的范围。周末（周六、周日）的范围似乎更广，数值也更高。
- **异常值：** 绘制在须之外的独立点代表潜在异常值。周日和周六显示有几个高值异常值，表示这些天有一些特别大的账单。周四也有一个相当高的异常值。

与仅显示每日平均账单的条形图相比，箱线图让我们对每天账单金额的变动情况有了更全面的认识。

### 自定义箱线图

与其他 Seaborn 函数类似，`boxplot` 提供了多种自定义选项。

- **方向：** 你可以通过交换 `x` 和 `y` 的位置或设置 `orient='h'` 来创建水平箱线图。

  ```python
  # 水平箱线图
  sns.boxplot(x="total_bill", y="day", data=tips, orient='h', palette=["#74c0fc", "#ffc078", "#8ce99a", "#ffc9c9"])
  plt.title("每日总账单金额分布")
  plt.xlabel("总账单 ($)")
  plt.ylabel("星期")
  plt.show()
  ```
- **顺序：** 使用 `order` 参数 (parameter)并传入一个类别名称列表，以控制类别的显示顺序。

  ```python
  # 指定日期的顺序
  day_order = ["Thur", "Fri", "Sat", "Sun"]
  sns.boxplot(x="day", y="total_bill", data=tips, order=day_order, palette=["#74c0fc", "#ffc078", "#8ce99a", "#ffc9c9"])
  # ... (添加标题/标签并显示图表)
  ```
- **色调（Hue）：** 你可以使用 `hue` 参数添加另一个分类维度，这会在 x 轴上的每个主要类别中，为 `hue` 变量的每个水平创建独立的、并排的箱体。例如，你可以按天比较账单，并根据顾客是否吸烟进行划分。

  ```python
  # 将“smoker”作为色调维度添加
  sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, palette="pastel")
  plt.title("按日期和吸烟者状态划分的总账单分布")
  plt.xlabel("星期")
  plt.ylabel("总账单 ($)")
  plt.show()
  ```

箱线图在你想要比较数值变量在由一个或多个分类变量定义的多个组中的分布时，特别有效。它们提供了一种快速方法来评估各组之间集中趋势、分布范围以及异常值存在情况的差异。

## 参考资料

- [seaborn.boxplot](https://seaborn.pydata.org/generated/seaborn.boxplot.html) — Seaborn Developers (2024)
  Seaborn `boxplot` 函数的官方文档，详细介绍了其参数和用法。
- [Python for Data Analysis: Data Wrangling with pandas, NumPy, and IPython](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本使用 Pandas、Matplotlib 和 Seaborn 等库进行 Python 数据处理和可视化的实用指南。
- [An Introduction to Statistical Learning: With Applications in Python](https://www.statlearning.com/) — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani, Jonathan Taylor (2023)
  Publisher: Springer; DOI: [10.1007/978-3-031-38747-0](https://doi.org/10.1007/978-3-031-38747-0)
  为数据分析和可视化提供了全面的统计学基础，包括箱线图背后的原理。
