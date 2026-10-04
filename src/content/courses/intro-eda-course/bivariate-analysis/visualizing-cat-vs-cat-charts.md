---
course: "intro-eda-course"
chapter: "bivariate-analysis"
lesson: "visualizing-cat-vs-cat-charts"
sourceId: 1174
sourceUrl: "https://apxml.com/zh/courses/intro-eda-course/chapter-4-bivariate-analysis/visualizing-cat-vs-cat-charts"
title: "分类变量间的可视化：堆叠条形图与分组条形图"
description: "创建堆叠或分组条形图，以可视化两个分类变量间的关联。"
order: 6
plots: ["plots/1174-0.json", "plots/1174-1.json"]
sourceHash: "5c2b76fd31dfa3265ef238fc903134b55e5a9abd4e5785fb2af87f45a57d371b"
sourceCorrections: []
---

虽然交叉表能提供两个分类变量间类别组合的精确数值计数，但可视化通常能使模式和关联更直接地显现。条形图是可视化分类数据的标准工具，在处理两个分类变量时，我们通常使用分组条形图或堆叠条形图。这些图表有助于我们比较不同类别组合的频率或比例。

### 分组条形图

分组条形图将代表一个分类变量计数（或比例）的条形并排排列，并根据第二个变量的类别进行分组。当您想直接比较一个变量在另一个变量不同级别上的计数时，这种格式尤为适用。

考虑一个船上乘客的数据集，其中包含生存状态（`Survived`：是/否）与乘客等级（`Pclass`：一等/二等/三等）两个变量。分组条形图是一种有效的方式，可以为每个乘客等级并排显示“是”和“否”的生存情况的条形。这种可视化方法使得可以轻易看出，例如，一等舱幸存者的绝对数量是高于还是低于一等舱非幸存者的数量，以及这种比较在二等舱和三等舱中如何变化。

让我们看看如何使用Seaborn创建它。假设您有一个包含`CategoryA`和`CategoryB`列的Pandas DataFrame `df`：

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据（请替换为您的实际数据）
data = {'CategoryA': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'X', 'X', 'Y', 'Y'],
        'CategoryB': ['A', 'A', 'B', 'B', 'A', 'B', 'B', 'A', 'A', 'B']}
df = pd.DataFrame(data)

# 创建分组条形图
plt.figure(figsize=(8, 5)) # 可选：调整图表大小
sns.countplot(data=df, x='CategoryA', hue='CategoryB', palette=['#4dabf7', '#ff922b'])

# 添加标签和标题以增强清晰度
plt.xlabel("类别 A")
plt.ylabel("计数")
plt.title("类别 A 中类别 B 的计数（分组）")
plt.legend(title='类别 B')
plt.tight_layout() # 调整布局
plt.show()
```

此代码使用Seaborn的`countplot`函数。通过指定`x='CategoryA'`和`hue='CategoryB'`，Seaborn会自动计算每个组合的计数，并将其绘制为分组条形图。`hue`参数 (parameter)在这里很重要；它告知Seaborn应使用哪个变量来对每个`x`类别中的条形进行分组。



![类别 A 中类别 B 的计数（分组）](plots/1174-0.json)



> 分组条形图比较了类别 A（'X'，'Y'）的每个级别中类别 B（'A'，'B'）的计数。

### 堆叠条形图

堆叠条形图也表示两个分类变量之间组合的计数，但它不是将条形并排放置，而是将代表第二个变量类别的条形堆叠在第一个变量的每个条形之上。每个堆叠条形的总高度代表主变量（x轴上的变量）该类别的总计数。

堆叠图对于了解第二个变量在第一个变量的每个类别中的*构成*或*比例*非常出色。使用我们的船上乘客示例，一个将`Pclass`放在x轴上的堆叠条形图将显示一等、二等和三等舱的单个条形。每个条形将按颜色划分，显示该舱内幸存者（“是”）和非幸存者（“否”）的比例。这使得可以方便地目视比较不同舱位的生存*率*。

您可以在创建交叉表后使用Pandas的绘图功能，或直接使用Seaborn的`histplot`（它也可以处理分类数据的计数）并使用`multiple='stack'`参数 (parameter)来创建堆叠条形图。

使用Pandas和Matplotlib在创建交叉表后进行绘制：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 假设 'df' 是包含 'CategoryA' 和 'CategoryB' 的 DataFrame
# 1. 创建交叉表
cross_tab = pd.crosstab(df['CategoryA'], df['CategoryB'])

# 2. 绘制堆叠条形图
ax = cross_tab.plot(kind='bar', stacked=True, figsize=(8, 5), color=['#4dabf7', '#ff922b'])

# 添加标签和标题
plt.xlabel("类别 A")
plt.ylabel("计数")
plt.title("类别 A 中类别 B 的构成（堆叠）")
plt.xticks(rotation=0) # 保持 x 轴标签水平
plt.legend(title='类别 B')
plt.tight_layout()
plt.show()
```

![Image 20](https://apxml.s3.amazonaws.com/content_images/section_1174_1.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070832Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=f6eead937592eb774774ec49fa766ecc883d8a1f3696ea4470d343509bf7aaaa)
或者，使用Seaborn的`histplot`：

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 使用 histplot 进行堆叠
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='CategoryA', hue='CategoryB', multiple='stack', palette=['#4dabf7', '#ff922b'], shrink=0.8) # shrink 参数在条形之间添加间隙

plt.xlabel("类别 A")
plt.ylabel("计数")
plt.title("类别 A 中类别 B 的构成（堆叠）")
plt.tight_layout()
plt.show()
```

![Image 21](https://apxml.s3.amazonaws.com/content_images/section_1174_2.webp?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAZQ3DPGY5OEILO6DN%2F20261002%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20261002T070832Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=119d36bb6475aa0ecfcb2491e1ec7d1958cd7d82bbc09f0ad41248077c4f88b8)
有时，如果将条形标准化以代表100%，则比较比例会更容易。这通常被称为“填充”堆叠条形图。

```python
# 从交叉表计算比例
cross_tab_prop = cross_tab.apply(lambda x: x / x.sum() * 100, axis=1)

# 绘制 100% 堆叠条形图
ax = cross_tab_prop.plot(kind='bar', stacked=True, figsize=(8, 5), color=['#4dabf7', '#ff922b'])

plt.xlabel("类别 A")
plt.ylabel("百分比 (%)")
plt.title("类别 A 中类别 B 的比例构成（100% 堆叠）")
plt.xticks(rotation=0)
plt.legend(title='类别 B', loc='center left', bbox_to_anchor=(1, 0.5)) # 将图例移到外部
plt.tight_layout()
plt.show()
```



![类别 A 中类别 B 的比例构成（100% 堆叠）](plots/1174-1.json)



> 一个100%堆叠条形图，显示了类别 A（'X'，'Y'）的每个级别中类别 B（'A'，'B'）的百分比分布。

### 选择合适的图表

- 当您的主要目的是比较第二个变量在第一个变量的不同类别中的绝对计数时，请使用**分组条形图**。例如，比较男性幸存者与女性幸存者的原始数量。
- 当您的主要目的是了解和比较第二个变量在第一个变量的每个类别中的*比例构成*时，请使用**堆叠条形图**（特别是100%版本）。例如，比较男性和女性之间的生存*率*（百分比）。

这两种图表都是您探索性数据分析工具集中有益的补充，它们提供对分类数据中隐藏关联的视觉理解，补充了从交叉表获得的数值汇总。请记住，始终清晰地标注坐标轴，并在需要时提供图例以方便解读。

## 参考资料

- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O’Reilly Media, Inc.
  数据可视化最佳实践的综合指南，包括条形图等图表类型的详细解释及其适当用法。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  关于使用Python中的Pandas进行数据操作和分析的权威著作，包含使用Matplotlib和Seaborn进行数据可视化的实践示例。
- [seaborn.countplot](https://seaborn.pydata.org/generated/seaborn.countplot.html) — Michael Waskom (2023)
  Seaborn的countplot函数的官方文档，详细说明了可视化分类数据计数的参数和示例。
- [seaborn.histplot](https://seaborn.pydata.org/generated/seaborn.histplot.html) — Michael Waskom (2024)
  Seaborn的histplot函数的官方文档，涵盖了使用'multiple'参数创建堆叠条形图的用法。
