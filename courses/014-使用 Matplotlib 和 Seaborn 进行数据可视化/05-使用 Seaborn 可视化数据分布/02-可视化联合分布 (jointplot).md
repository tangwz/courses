# 可视化联合分布 (jointplot)

来源：[原文](https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-5-seaborn-visualizing-distributions/seaborn-jointplot)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然使用直方图或KDE图来了解单个变量的分布很有用，但我们通常需要明白两个变量如何相互影响。它们之间有关系吗？一个变量的分布是否会随着另一个变量的值而变化？将两个变量的*联合分布*可视化有助于回答这些问题。

Seaborn 提供了一个方便的函数 `jointplot()`，专门为此目的而设计。它创建一个多面板图，既显示了两个变量之间的关系（联合分布），也显示了每个变量沿边距的各自分布（边际分布）。

可以将其看作是结合了散点图（用于查看关系）和直方图（用于查看单独的分布），全部呈现在一个图中。

### 创建一个基本联合图

让我们使用 `jointplot()` 函数来查看 Seaborn 内置 `tips` 数据集中总账单金额与小费金额之间的关系。首先，我们需要加载数据并导入所需的库：

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 加载示例 tips 数据集
tips = sns.load_dataset("tips")

# 显示前几行以查看数据
print(tips.head())
```

现在，我们可以使用 `seaborn.jointplot()` 创建一个联合图：

```python
# 创建 total_bill 和 tip 的联合图
sns.jointplot(data=tips, x="total_bill", y="tip")

plt.suptitle("总账单和小费金额的联合分布", y=1.02) # 在图上方稍加标题
plt.tight_layout() # 调整布局
plt.show()
```

运行此代码后，您将看到一个包含三个部分的图：

1. **中心图：** 一个散点图，显示单个数据点，其中x轴是 `total_bill`，y轴是 `tip`。这有助于可视化这两个变量之间的关系或关联。默认情况下，`jointplot` 在此处使用散点图。
2. **顶部边际图：** 一个直方图（或KDE图），显示x轴变量 (`total_bill`) 的分布。
3. **右侧边际图：** 一个直方图（或KDE图），显示y轴变量 (`tip`) 的分布。

这种组合视图使您能够同时评估 `total_bill` 和 `tip` 之间的关系，并独立地了解每个变量的形状、中心和离散程度。

### 不同类型的联合图

默认的 `jointplot` 在中心视图使用散点图，但您可以使用 `kind` 参数 (parameter)更改它。这使得您能够以不同的方式呈现联合分布，根据您的数据以及您想要突出的内容，这可能更合适。

`kind` 的常见选项包括：

- `kind="scatter"` (默认)：显示单个点。适用于点不重叠太多的小数据集。
- `kind="kde"`：对联合图（中心）和边际图都使用核密度估计。这显示了分布的平滑表示，有助于可视化点的密度。
- `kind="hist"`：对中心图使用2D直方图（类似热力图），对边际图使用标准直方图。适用于点重叠明显的大数据集。
- `kind="hex"`：类似于 `kind="hist"`，但中心图使用六边形分箱而不是方形分箱。这是处理大数据集中重叠点的另一种方法。
- `kind="reg"`：在中心视图的散点图中添加回归线，并在边距显示带有直方图的KDE图。这有助于突出线性关系。
- `kind="resid"`：在中心图绘制线性回归的残差。这更专门用于检查线性模型的假设。

让我们尝试创建一个带有核密度估计的联合图：

```python
# 使用 KDE 创建联合图
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde")

plt.suptitle("总账单和小费的联合 KDE 图", y=1.02)
plt.tight_layout()
plt.show()
```

此图使用平滑曲线而非条形图或点。中心图显示等高线，其中较暗的区域表示数据点浓度较高。边际图是平滑的1D KDE曲线。

这是一个使用六边形分箱的例子，它非常适合散点图变得拥挤的大数据集：

```python
# 使用六边形分箱创建联合图
# 使用稍大的数据集以获得更好效果（模拟）
import numpy as np
np.random.seed(0) # 用于重现性
data_large = pd.DataFrame({
    'x_var': np.random.randn(1000),
    'y_var': np.random.randn(1000) * 0.5 + np.random.randn(1000) * 0.2
})

sns.jointplot(data=data_large, x="x_var", y="y_var", kind="hex", color="#4263eb") # 使用靛蓝色

plt.suptitle("更大数据的联合六边形图", y=1.02)
plt.tight_layout()
plt.show()
```

在此六边形分箱图中，中心区域使用六边形，每个六边形的颜色强度表示落在其中的数据点数量。这避免了散点图中数据点过多时出现的重叠问题。

您还可以使用 `color`、`height` 等参数（用于控制大小）进一步自定义外观，以及 `joint_kws`、`marginal_kws` 来向底层绘图函数传递特定参数。

`jointplot()` 是一个有用的工具，因为它将三个图打包成一个连贯的图形，提供双变量和单变量分布的简洁概述。在数据分析阶段，它特别有效，能够快速了解数值变量对之间的关系和分布。

## 参考资料

- [seaborn.jointplot](https://seaborn.pydata.org/generated/seaborn.jointplot.html) — Michael Waskom and the Seaborn development team (2023)
  `jointplot`函数的官方文档，详细说明其参数并展示用于可视化联合分布和边际分布的各种`kind`选项。
- [Python Data Science Handbook: Essential Tools for Working with Data](https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html) — Jake VanderPlas (2016)
  Publisher: O'Reilly Media; Pages: Chapter 4: Visualization With Seaborn
  该书关于Seaborn统计图的章节说明了如何使用`jointplot`来理解数值变量之间的关系。
- [Fundamentals of Data Visualization](https://clauswilke.com/dataviz/) — Claus O. Wilke (2019)
  Publisher: O'Reilly Media
  数据可视化设计指南，提供有效展示变量之间关系的原则，包括散点图、热图和等高线图。

---

[上一节](01-%E7%9B%B4%E6%96%B9%E5%9B%BE%E5%92%8C%E6%A0%B8%E5%AF%86%E5%BA%A6%E4%BC%B0%E8%AE%A1%20%28histplot%2C%20kdeplot%29.md) · [下一节](03-%E7%94%A8%E4%BA%8E%E6%B1%87%E6%80%BB%E7%BB%9F%E8%AE%A1%E7%9A%84%E7%AE%B1%E7%BA%BF%E5%9B%BE%20%28boxplot%29.md)
