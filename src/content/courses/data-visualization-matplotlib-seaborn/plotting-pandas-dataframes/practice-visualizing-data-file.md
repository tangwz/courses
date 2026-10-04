---
course: "data-visualization-matplotlib-seaborn"
chapter: "plotting-pandas-dataframes"
lesson: "practice-visualizing-data-file"
sourceId: 1222
sourceUrl: "https://apxml.com/zh/courses/data-visualization-matplotlib-seaborn/chapter-8-plotting-pandas-dataframes/practice-visualizing-data-file"
title: "动手实践：从文件中进行数据可视化"
description: "使用Pandas从文件加载数据，并使用Matplotlib和Seaborn创建多种图表。"
order: 7
plots: ["plots/1222-0.json"]
sourceHash: "cb8ca99c48597e3f13a039e58c61ba29f4dac563934d556a35923f16cd77589a"
sourceCorrections: []
---

使用Pandas加载数据以及Matplotlib和Seaborn与DataFrame的交互方式是数据可视化的基础。接下来，我们将通过一个完整的示例，从文件中加载数据开始，直至生成有意义的图表。

假设你有一个数据集，其中包含不同产品类别的月度销售数据，存储在一个逗号分隔值 (CSV) 文件中。我们的目标是加载这些数据并直观呈现销售走势。

### 设置与数据加载

首先，请确保你的工作目录下有一个名为 `monthly_sales.csv` 的CSV文件，内容如下：

```csv
Month,CategoryA_Sales,CategoryB_Sales
Jan,150,80
Feb,160,95
Mar,175,90
Apr,180,105
May,195,110
Jun,210,100
```

现在，让我们导入所需的库并将数据加载到Pandas DataFrame中，开始我们的Python脚本。

```python
# 导入库
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 从CSV文件加载数据集
try:
    df_sales = pd.read_csv('monthly_sales.csv')
    print("数据加载成功:")
    print(df_sales.head()) # 显示前几行
except FileNotFoundError:
    print("错误：未找到'monthly_sales.csv'文件。")
    # 你可能希望停止执行或以不同方式处理此情况
    # 在本示例中，如果文件未找到，我们将创建一个占位符DataFrame
    data = {'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'CategoryA_Sales': [150, 160, 175, 180, 195, 210],
            'CategoryB_Sales': [80, 95, 90, 105, 110, 100]}
    df_sales = pd.DataFrame(data)
    print("\n已创建占位数据:")
    print(df_sales.head())

# 基本数据类型检查
print("\n数据信息:")
df_sales.info()
```

这段代码首先使用标准别名导入Pandas、Matplotlib的Pyplot模块和Seaborn。然后，它尝试加载 `monthly_sales.csv`。我们包含了一个 `try-except` 块来处理文件可能缺失的情况，此时会创建一个默认的DataFrame，以便代码的其余部分仍能运行进行演示。最后，`df_sales.head()` 显示了前几行，而 `df_sales.info()` 提供了列及其数据类型的汇总。你应该会看到 `Month` 是一个对象（字符串），而销售列是整数 (`int64`)。

### 使用Matplotlib进行可视化

让我们直接使用Matplotlib，创建一个折线图，显示两个类别的销售走势。我们将把 `df_sales` DataFrame中的相关列传递给 `plt.plot()` 函数。

```python
# 确保数据在绘图前已加载
if 'df_sales' in locals():
    # 为图表创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 5)) # 调整图形大小

    # 绘制类别A销售额
    ax.plot(df_sales['Month'], df_sales['CategoryA_Sales'], marker='o', linestyle='-', color='#4263eb', label='Category A')

    # 绘制类别B销售额
    ax.plot(df_sales['Month'], df_sales['CategoryB_Sales'], marker='s', linestyle='--', color='#f76707', label='Category B')

    # 添加标题和标签
    ax.set_title('按类别划分的月度销售走势 (Matplotlib)')
    ax.set_xlabel('月份')
    ax.set_ylabel('销售单位')

    # 添加图例
    ax.legend()

    # 改善布局并显示图表
    plt.tight_layout()
    plt.show()
else:
    print("DataFrame 'df_sales' 不可用于绘图。")
```

在这里，我们显式选择 `Month` 列作为x轴，相应的销售列（`CategoryA_Sales`、`CategoryB_Sales`）作为y轴。我们使用标记 (token) (`marker`)、线型 (`linestyle`)、颜色 (`color`)来自定义图表，并使用 `label` 添加标签以提高清晰度，这些标签随后通过 `ax.legend()` 显示。我们还使用 `Axes` 对象 (`ax`) 上的方法设置了标题和轴标签。`plt.tight_layout()` 调整间距，`plt.show()` 显示图表。这种方法通过直接使用Matplotlib函数，让你对每个元素有精细的控制。

### 使用Pandas内置绘图功能进行可视化

Pandas DataFrame有一个方便的 `.plot()` 方法，它是Matplotlib的封装。让我们使用此方法创建一个类似的折线图。

```python
# 确保数据在绘图前已加载
if 'df_sales' in locals():
    # 使用Pandas的.plot()方法
    ax = df_sales.plot(kind='line', x='Month', y=['CategoryA_Sales', 'CategoryB_Sales'],
                       marker='o', figsize=(8, 5),
                       color=['#4263eb', '#f76707']) # 为线条指定颜色

    # 设置标题和标签（Pandas会设置一些默认值，但我们可以覆盖它们）
    ax.set_title('按类别划分的月度销售走势 (Pandas .plot)')
    ax.set_xlabel('月份')
    ax.set_ylabel('销售单位')
    ax.legend(title='Category') # 自定义图例

    # 显示图表
    plt.tight_layout()
    plt.show()
else:
    print("DataFrame 'df_sales' 不可用于绘图。")
```

这段代码更简洁。我们直接在 `df_sales` DataFrame上调用 `.plot()`。我们指定 `kind='line'`，x轴的列 (`x='Month'`)，以及y轴的列列表 (`y=['CategoryA_Sales', 'CategoryB_Sales']`)。Pandas会自动处理多条线的绘制和添加基本图例。我们仍然可以访问底层Matplotlib `Axes` 对象（由 `.plot()` 返回）进行进一步自定义，例如设置标题和标签。

### 使用Seaborn进行可视化

Seaborn擅长创建具有统计信息且美观的图表，通常代码量更少，特别是在处理“整洁”数据时。我们当前的DataFrame格式（通常称为“宽”格式）对于一些偏好“长”格式（每个观察值是一行）的Seaborn函数来说并不理想。让我们首先将DataFrame *转换* 为长格式，这是一个常见的数据准备步骤。

```python
# 确保数据在进行转换/绘图前已加载
if 'df_sales' in locals():
    # 将DataFrame从宽格式转换为长格式
    df_sales_long = pd.melt(df_sales,
                            id_vars=['Month'], # 要保留的标识符列
                            value_vars=['CategoryA_Sales', 'CategoryB_Sales'], # 要取消透视的列
                            var_name='Category', # 保存原始列名的新列的名称
                            value_name='Sales') # 保存值的新列的名称

    print("\n长格式DataFrame:")
    print(df_sales_long.head())

    # 现在，使用Seaborn创建折线图
    plt.figure(figsize=(8, 5)) # 使用Matplotlib控制图形大小
    sns.lineplot(data=df_sales_long, x='Month', y='Sales', hue='Category',
                 marker='o', palette=['#4263eb', '#f76707']) # 使用hue参数区分类别

    # 添加标题和标签（Seaborn会设置一些默认值）
    plt.title('按类别划分的月度销售走势 (Seaborn)')
    plt.xlabel('月份')
    plt.ylabel('销售单位')

    # 改善布局并显示图表
    plt.tight_layout()
    plt.show()

    # 示例：创建柱状图比较每个类别的总销售额
    # 注意：这需要先进行数值聚合，或者使用barplot的估计器
    category_totals = df_sales_long.groupby('Category')['Sales'].sum().reset_index()
    print("\n每个类别的总销售额:")
    print(category_totals)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=category_totals, x='Category', y='Sales', palette=['#4263eb', '#f76707'])
    plt.title('总销售额比较')
    plt.xlabel('产品类别')
    plt.ylabel('总销售单位')
    plt.tight_layout()
    plt.show()

else:
    print("DataFrame 'df_sales' 不可用于绘图。")
```

首先，我们使用 `pd.melt` 来转换 `df_sales`。`id_vars=['Month']` 保留了“Month”列，而 `value_vars` 指定了我们希望合并其值的列。`var_name='Category'` 创建了一个包含原始列名（“CategoryA\_Sales”、“CategoryB\_Sales”）的新列，而 `value_name='Sales'` 创建了一个包含相应销售数据的新列。

有了长格式数据 (`df_sales_long`)，创建Seaborn折线图 (`sns.lineplot`) 变得简单。我们将整个DataFrame传递给 `data`，并指定 `x`、`y` 和重要的 `hue` 的列名作为字符串。`hue='Category'` 参数 (parameter)指示Seaborn为“Category”列中的每个独立值绘制单独的线条（具有不同的颜色和可能的样式）。Seaborn会自动处理图例并应用其默认样式。我们使用 `palette` 来指定颜色。

我们还添加了第二个示例，使用了 `sns.barplot`。请注意，对于显示*总*销售额的柱状图，我们首先需要使用Pandas的 `groupby()` 和 `sum()` 计算这些总数，然后再用Seaborn绘图。Seaborn的 `barplot` 默认情况下会在x轴的每个类别存在多个值时显示均值（和置信区间）；由于我们预先聚合了数据，它会直接显示总和。

这是使用Plotly语法演示 `barplot` 可能的样子：



![总销售额比较](plots/1222-0.json)



> 简单的柱状图，比较了整个期间内类别A与类别B的总销售单位。

此次实践操作展示了如何使用Pandas加载数据，并根据你的需求选择最合适的绘图工具（Matplotlib、Pandas的 `.plot()` 或Seaborn）。你看到了Pandas如何与这两个库良好集成，以及数据重塑（例如转换格式）如何促进可视化，特别是使用Seaborn时。你现在具备加载自己的数据集并开始生成富有洞察力的图表的能力。

## 参考资料

- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — The Pandas Development Team (2023)
  提供了关于数据加载（例如`read_csv`）、DataFrame操作和内置绘图功能的全面信息。
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html) — The Matplotlib Development Team (2023)
  官方学习资源，用于创建、定制和显示静态、动画和交互式可视化图表。
- [Seaborn Documentation](https://seaborn.pydata.org/index.html) — Michael Waskom and the Seaborn development team (2023)
  提供了统计数据可视化的详细指南和示例，包括不同数据格式的最佳实践。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104030/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  由Pandas创建者撰写的权威指南，涵盖了使用Python进行数据操作、分析和可视化。
