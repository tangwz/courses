---
course: "intro-eda-course"
chapter: "univariate-analysis"
lesson: "visualizing-bar-charts"
sourceId: 1160
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-3-univariate-analysis/visualizing-bar-charts"
title: "可视化分类变量：柱状图"
description: "使用柱状图有效呈现分类变量的频率分布。"
order: 6
plots: ["plots/1160-0.json"]
sourceHash: "c79c51378155e8ed892d6b80ff1dc239bb893ab04201c91306bdbc084624d27b"
sourceCorrections: []
---

在计算出分类变量的频率计数后，下一步就是将这些分布进行可视化。尽管数字表格很精确，但图表形式通常能更直观地呈现数据中的相对频率和模式。在显示单个分类变量的类别频率或比例时，最常用且有效的可视化方式是柱状图。

柱状图使用矩形条，其长度与它们所代表的值成比例。对于单变量分类分析，条形图通常显示落入每个类别的观测值的计数（频率）或比例。这使得比较不同类别变得一目了然。

### 使用 Python 创建柱状图

Matplotlib 和 Seaborn 等 Python 库提供了便捷的函数，可以直接从 Pandas Series 或 DataFrame 生成柱状图。Seaborn 在 Matplotlib 的基础上构建，提供了专门用于统计可视化的函数，对于常见图表通常只需较少代码。

创建类别计数柱状图的一种常用方法是使用 Seaborn 的 `countplot` 函数。它会自动计算指定列中每个类别的频率并绘图。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 假设 'df' 是你的 DataFrame，'product_category' 是你关注的列
# 示例 DataFrame 创建：
data = {'product_category': ['Electronics', 'Clothing', 'Groceries', 'Electronics', 'Clothing', 'Electronics', 'Home Goods', 'Clothing', 'Groceries', 'Electronics']}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 5)) # 可选：调整图表大小
sns.countplot(data=df, x='product_category', palette=['#4dabf7', '#69db7c', '#ff922b', '#be4bdb']) # 使用调色板中的颜色
plt.title('产品类别频率')
plt.xlabel('产品类别')
plt.ylabel('计数')
plt.xticks(rotation=45) # 如果标签重叠，则旋转
plt.tight_layout() # 调整布局
plt.show()
```

[Image 17](https://apxml.s3.amazonaws.com/content_images/section_1160_1.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070819Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=ec85ccb9e60a20e4d75b48bd6b19ed8d63158b10d35f23848a059668d5971163)
或者，你可以先使用 Pandas 计算值计数，然后使用 Matplotlib 或 Pandas 的绘图函数：

```python
import matplotlib.pyplot as plt
import pandas as pd

# 假设 'df' 是你的 DataFrame，'product_category' 是列
# 示例 DataFrame 创建（与上面相同）：
data = {'product_category': ['Electronics', 'Clothing', 'Groceries', 'Electronics', 'Clothing', 'Electronics', 'Home Goods', 'Clothing', 'Groceries', 'Electronics']}
df = pd.DataFrame(data)

category_counts = df['product_category'].value_counts()

plt.figure(figsize=(8, 5))
category_counts.plot(kind='bar', color=['#4dabf7', '#69db7c', '#ff922b', '#be4bdb'])
plt.title('产品类别频率')
plt.xlabel('产品类别')
plt.ylabel('计数')
plt.xticks(rotation=45, ha='right') # 旋转并对齐标签
plt.tight_layout()
plt.show()
```

[Image 18](https://apxml.s3.amazonaws.com/content_images/section_1160_2.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070819Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=dc91220b884348c3001c4d54f92741c90f1713b8dbbe988e49f7e8b57618b790)
两种方法都能达到类似的结果。Seaborn 的 `countplot` 对于简单的频率图可能更直接一些，而 Pandas 的方法则在绘图前明确提供了计数，这可能很有用。

### 解读柱状图

查看分类变量的柱状图时，请考虑以下几点：

1. **比较高度：** 哪些类别出现最频繁？哪些最不频繁？条形的相对高度直接表示相对计数。
2. **分布形态：** 尽管“形态”更正式地与数值分布关联，但也要留意是否有任何模式。是否存在一个主导类别？分布在各类别中是相对均匀的，还是严重偏向一两个类别？
3. **类别数量：** 有多少个不同的类别？类别数量非常多可能会使标准柱状图显得杂乱。在这种情况下，你可以考虑将不那么常见的类别归入“其他”类别，或者使用水平柱状图（在 Pandas 绘图中是 `kind='barh'`，在 Seaborn 中是 `y=` 而不是 `x=`）以便更好地阅读标签。

下面是一个使用 Plotly 呈现的虚构客户满意度评分分布示例。



![客户满意度评分分布](plots/1160-0.json)



> 客户满意度回应分布，显示“满意”是最常见的评分。

柱状图是 EDA（探索性数据分析）工具包中的一项基本工具，用于理解分类数据的构成。它们将频率表转换为易于理解的视觉格式，显示数据集中不同群体的普遍性和分布情况。

## 参考资料

- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  一份关于创建有效数据可视化的综合指南，其中包含专门讨论如何适当使用和解读分类数据的条形图的章节。
- [seaborn.countplot - seaborn 0.13.0 documentation](https://seaborn.pydata.org/generated/seaborn.countplot.html) — Michael Waskom and the Seaborn development team (2024)
  Seaborn `countplot` 函数的官方文档，详细说明了其参数和用于可视化分类数据频率分布的用法。
- [Visualization - pandas 2.2.0 documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html#bar-plots) — The Pandas Development Team (2024)
  Pandas 官方用户指南的可视化章节，展示了如何使用 `.plot()` 方法直接从 Series 或 DataFrame 生成条形图。
