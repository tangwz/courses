# 练习：使用Seaborn重制图表

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-4-seaborn-enhanced-visuals-intro/practice-seaborn-basic-plots)

[返回章节目录](README.md) · [返回课程目录](../README.md)

重现常用图表，利用Seaborn的高级界面、样式设置能力、调色板以及 `lineplot` 和 `scatterplot` 可视化效果。重点是实现比单独使用Matplotlib时更整洁的代码和更好的视觉效果。

首先，让我们确保环境已准备就绪。我们将导入Seaborn、Matplotlib的Pyplot模块（用于标题等可能的自定义设置）、NumPy（用于生成示例数据）和Pandas（因为Seaborn通常与DataFrame配合得最好）。

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 为演示设置Seaborn默认样式
sns.set_style("whitegrid")
```

### 练习1：使用Seaborn创建折线图

想象您有表示10天内温度读数的数据。让我们生成一些示例数据并使用Seaborn的 `lineplot` 绘制它。

```python
# 生成示例数据
days = np.arange(1, 11)
temperature = np.array([15, 16, 18, 17, 19, 21, 20, 22, 23, 21]) + np.random.randn(10) * 0.5

# 使用Seaborn创建折线图
plt.figure(figsize=(8, 4)) # 可选：调整图表大小
sns.lineplot(x=days, y=temperature, marker='o') # 添加标记以提高清晰度

# 添加标题和标签（使用Matplotlib函数）
plt.title('10天温度趋势')
plt.xlabel('天')
plt.ylabel('温度 (℃)')

# 显示图表
plt.show()
```

请注意 `sns.lineplot` 如何直接接收x和y数据。我们之前使用了 `sns.set_style("whitegrid")`，它自动为图表提供了美观的背景网格。我们仍然使用Matplotlib的 `plt.title`、`plt.xlabel` 和 `plt.ylabel` 来添加描述性标签，表明了Seaborn和Matplotlib如何配合工作。`marker='o'` 参数 (parameter)为折线上的每个数据点添加了圆形标记 (token)。

### 练习2：使用Seaborn和Hue创建散点图

现在，让我们创建一个散点图，查看两个变量之间的关系，例如学习时长和考试分数。我们还将使用 `hue` 参数 (parameter)引入第三个分类变量，比如“学习小组”（A组或B组）。

```python
# 生成示例数据
np.random.seed(42) # 为了结果可复现
study_hours = np.random.rand(50) * 10
scores = 20 + 6 * study_hours + np.random.randn(50) * 8 # 基础分数 + 时长影响 + 噪声
study_group = np.random.choice(['Group A', 'Group B'], 50)

# 创建Pandas DataFrame（Seaborn的推荐输入）
data = pd.DataFrame({
    'Study Hours': study_hours,
    'Exam Score': scores,
    'Group': study_group
})

# 使用Seaborn创建散点图
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Study Hours', y='Exam Score', hue='Group', palette='viridis')

# 添加标题和标签
plt.title('按小组划分的考试分数与学习时长关系')
plt.xlabel('学习时长')
plt.ylabel('考试分数')

# 显示图表
plt.show()
```

这里，我们首先将数据组织成一个Pandas DataFrame。这是使用Seaborn时非常常见的一种做法。然后我们将DataFrame传递给 `sns.scatterplot` 的 `data` 参数。我们只需提供列的字符串名称，就指定了X轴、Y轴以及 `hue` 参数的分类变量。

请注意Seaborn如何自动：

1. 根据 'viridis' 调色板将'Group'类别映射到不同颜色。
2. 创建图例来解释颜色映射。

这比单独使用Matplotlib实现相同结果要简洁得多，后者通常需要您单独绘制每个组并手动创建图例。

### 练习3：尝试不同样式和调色板

Seaborn让改变整体外观变得容易。让我们重新生成散点图，但这次改变样式和上下文 (context)，并使用不同的调色板。

```python
# 设置不同样式和上下文
sns.set_style("darkgrid")
sns.set_context("talk") # 选项：'paper', 'notebook', 'talk', 'poster'

# 使用不同调色板重制散点图
plt.figure(figsize=(8, 5))
sns.scatterplot(data=data, x='Study Hours', y='Exam Score', hue='Group', palette='bright')

# 添加标题和标签
plt.title('考试分数与学习时长（深色网格，演讲上下文）')
plt.xlabel('学习时长')
plt.ylabel('考试分数')

# 显示图表
plt.show()

# 如果后续图表需要，可重置为默认样式和上下文
# sns.set_style("whitegrid")
# sns.set_context("notebook")
```

通过将 `sns.set_style` 更改为 `"darkgrid"` 并将 `sns.set_context` 更改为 `"talk"`，后续图表的背景、网格线、字体大小和线宽将全局调整，使它们适合用于演示。我们还切换到 `"bright"` 调色板以获得不同的视觉效果。尝试这些设置是快速调整图表以适应目标受众和媒介的好方法。

本次练习展示了如何使用Seaborn的 `lineplot` 和 `scatterplot` 函数，其与Pandas DataFrames的整合，运用 `hue` 参数 (parameter)进行分类编码，以及如何使用样式、上下文和调色板轻松自定义外观。正如您所见，Seaborn通常为创建统计信息丰富且视觉效果好的图表提供了更直接的途径。

## 参考资料

- [Seaborn: Statistical Data Visualization](https://seaborn.pydata.org/) — Michael Waskom and the Seaborn development team (2024)
  官方文档，提供丰富的教程、API参考和使用Python进行统计数据可视化的示例。
- [Matplotlib Documentation](https://matplotlib.org/) — John Hunter and the Matplotlib development team (2024)
  Matplotlib官方文档，包含Python绘图基础、自定义选项以及与其他库的集成。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用Pandas进行数据操作的重要参考书，其中包含使用Matplotlib和Seaborn进行数据可视化的章节。
- [Data Visualization with Python and JavaScript: Scrape, Clean, Explore, and Transform Your Data Into Insights](https://www.oreilly.com/library/view/data-visualization-with/9781098111878/) — Kyran Dale (2022)
  Publisher: O'Reilly Media; Pages: 566
  提供使用Matplotlib和Seaborn等Python库创建各种数据可视化的实用指导和详细示例。

---

[上一节](06-%E4%BA%86%E8%A7%A3%20Seaborn%20%E7%9A%84%E6%95%B0%E6%8D%AE%E8%BE%93%E5%85%A5%E8%A6%81%E6%B1%82.md) · [下一节](../05-%E4%BD%BF%E7%94%A8%20Seaborn%20%E5%8F%AF%E8%A7%86%E5%8C%96%E6%95%B0%E6%8D%AE%E5%88%86%E5%B8%83/01-%E7%9B%B4%E6%96%B9%E5%9B%BE%E5%92%8C%E6%A0%B8%E5%AF%86%E5%BA%A6%E4%BC%B0%E8%AE%A1%20%28histplot%2C%20kdeplot%29.md)
