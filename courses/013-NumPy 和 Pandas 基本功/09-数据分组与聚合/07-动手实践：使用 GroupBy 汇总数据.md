# 动手实践：使用 GroupBy 汇总数据

来源：[原文](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas/groupby-summarizing-practice)

[返回章节目录](README.md) · [返回课程目录](../README.md)

分组和聚合是数据操作中的基本概念。我们将通过使用示例数据集的多个例子，演示 Pandas `groupby()` 的分拆-应用-组合模式。

### 设置示例数据

首先，确保已导入 Pandas。如果您在 Jupyter Notebook 或 Python 脚本中跟随操作，请从这里开始：

```python
import pandas as pd
import numpy as np # 通常与 Pandas 一起使用很有用
```

现在，我们来创建一个简单的 DataFrame，它表示不同产品类别在各个区域的销售数据。这将是我们练习的数据。

```python
# 创建示例数据
data = {'Category': ['Electronics', 'Clothing', 'Electronics', 'Groceries', 'Clothing', 'Groceries', 'Electronics', 'Clothing'],
        'Region': ['North', 'South', 'North', 'East', 'West', 'East', 'West', 'South'],
        'Sales': [1200, 300, 1500, 250, 450, 200, 1800, 350],
        'Quantity': [2, 3, 1, 5, 4, 4, 3, 4]}
sales_df = pd.DataFrame(data)

print("示例销售数据：")
print(sales_df)
```

这应该输出：

```
Sample Sales Data:
      Category Region  Sales  Quantity
0  Electronics  North   1200         2
1     Clothing  South    300         3
2  Electronics  North   1500         1
3    Groceries   East    250         5
4     Clothing   West    450         4
5    Groceries   East    200         4
6  Electronics   West   1800         3
7     Clothing  South    350         4
```

### 按单列分组

我们的首要任务是找出每个产品`类别`的总销售额。这需要按`类别`拆分 DataFrame，然后对每个组内的`销售额`列应用 `sum()` 函数，最后合并结果。

```python
# 按“类别”分组并计算每个类别的“销售额”总和
category_sales_sum = sales_df.groupby('Category')['Sales'].sum()

print("\n每个类别的总销售额：")
print(category_sales_sum)
```

输出：

```
Total Sales per Category:
Category
Clothing        1100
Electronics     4500
Groceries        450
Name: Sales, dtype: int64
```

请注意，结果是一个 Pandas Series，其索引是我们分组的`类别`，值是汇总后的`销售额`。

如果我们想知道每个类别的*平均*销售数量呢？

```python
# 按“类别”分组并计算“数量”的平均值
category_quantity_avg = sales_df.groupby('Category')['Quantity'].mean()

print("\n每个类别的平均数量：")
print(category_quantity_avg)
```

输出：

```
Average Quantity per Category:
Category
Clothing       3.666667
Electronics    2.000000
Groceries      4.500000
Name: Quantity, dtype: float64
```

您也可以类似地应用其他聚合函数，例如 `count()`、`size()`、`min()`、`max()`、`std()`（标准差）等。请记住，`count()` 不包括缺失值 (NaN)，而 `size()` 包括它们。

### 应用多个聚合操作

通常，您会希望一次性计算每个组的多个汇总统计量。`.agg()` 方法非常适合此目的。

我们来计算每个`类别`的总销售额 (`sum`) 和平均数量 (`mean`)。

```python
# 按“类别”分组并应用多个聚合操作
category_summary = sales_df.groupby('Category').agg(
    Total_Sales=('Sales', 'sum'),
    Average_Quantity=('Quantity', 'mean'),
    Number_of_Sales=('Sales', 'count') # 统计非缺失的销售条目
)

print("\n每个类别的汇总统计量：")
print(category_summary)
```

输出：

```
Summary Statistics per Category:
             Total_Sales  Average_Quantity  Number_of_Sales
Category
Clothing            1100          3.666667                3
Electronics         4500          2.000000                3
Groceries            450          4.500000                2
```

在这里，我们向 `.agg()` 传递了一个字典。字典的键（'Total\_Sales'、'Average\_Quantity'、'Number\_of\_Sales'）成为结果 DataFrame 中的新列名。值是元组，其中第一个元素是要聚合的原始列（'Sales' 或 'Quantity'），第二个元素是聚合函数（'sum'、'mean'、'count'）。这提供了一个清晰且有条理的汇总。

或者，如果您想将相同的聚合函数集应用于所有数值列（或选定列），可以传递一个函数列表：

```python
# 计算每个类别的最小和最大销售额
category_sales_min_max = sales_df.groupby('Category')['Sales'].agg(['min', 'max'])

print("\n每个类别的最小和最大销售额：")
print(category_sales_min_max)
```

输出：

```
Min and Max Sales per Category:
             min   max
Category
Clothing     300   450
Electronics 1200  1800
Groceries    200   250
```

### 按多列分组

通过向 `groupby()` 传递列名列表，我们可以创建更细粒度的分组。我们来找出`区域`和`类别`每个组合的总销售额。

```python
# 同时按“区域”和“类别”分组，然后对“销售额”求和
region_category_sales = sales_df.groupby(['Region', 'Category'])['Sales'].sum()

print("\n每个区域和类别的总销售额：")
print(region_category_sales)
```

输出：

```
Total Sales per Region and Category:
Region  Category
East    Groceries     450
North   Electronics  2700
South   Clothing      650
West    Clothing      450
        Electronics  1800
Name: Sales, dtype: int64
```

结果是一个带有 `MultiIndex`（区域、类别）的 Series。这个分层索引表示分组的组合。

您也可以在这里应用多个聚合操作：

```python
# 同时按“区域”和“类别”分组，计算销售额总和与数量平均值
region_category_summary = sales_df.groupby(['Region', 'Category']).agg(
    Total_Sales=('Sales', 'sum'),
    Average_Quantity=('Quantity', 'mean')
)

print("\n每个区域和类别的汇总：")
print(region_category_summary)
```

输出：

```
Summary per Region and Category:
                     Total_Sales  Average_Quantity
Region Category
East   Groceries             450               4.5
North  Electronics          2700               1.5
South  Clothing              650               3.5
West   Clothing              450               4.0
       Electronics          1800               3.0
```

这给我们一个行上带有 `MultiIndex` 的 DataFrame。

### 分组数据的可视化

聚合数据通常通过可视化更容易理解。我们来使用 Plotly 创建一个简单的条形图，显示每个类别的总销售额。



[交互图表：各产品类别的总销售额](https://apxml.com/zh/courses/essential-numpy-pandas/chapter-9-grouping-aggregating-data-pandas/groupby-summarizing-practice#plot-hrtdoq)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "Clothing",
        "Electronics",
        "Groceries"
      ],
      "y": [
        1100,
        4500,
        450
      ],
      "marker": {
        "color": [
          "#228be6",
          "#ae3ec9",
          "#74b816"
        ]
      }
    }
  ],
  "layout": {
    "title": {
      "text": "各产品类别的总销售额"
    },
    "xaxis": {
      "title": {
        "text": "类别"
      }
    },
    "yaxis": {
      "title": {
        "text": "总销售额 ($)"
      }
    },
    "template": "plotly_white",
    "width": 600,
    "height": 400
  }
}
```

</details>



> 各产品类别产生的总销售额。在这个小型数据集中，电子产品明显占主导地位。

### 练习任务

现在，请使用 `sales_df` DataFrame 独自尝试以下任务：

1. 找出每个`区域`的平均`销售额`。
2. 统计每个`区域`的销售记录数量 (`size()`)。
3. 找出每个`区域`和`类别`组合中的`最大`销售数量。
4. 计算每个`区域`的`销售额`总和与`数量`总和。使用 `.agg()` 方法。

这些练习巩固了您应用 `groupby()` 和聚合函数，从数据中提取有价值汇总的不同方法。掌握分组是使用 Pandas 进行数据分析的重要一步。

## 参考资料

- [Group by: split-apply-combine](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html) — pandas development team (2023)
  Pandas官方用户指南中关于数据分组的部分，详细介绍了拆分-应用-组合模式和核心的groupby功能。
- [Python for Data Analysis](https://www.oreilly.com/library/view/python-for-data/9781098104023/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  一本关于使用Python和Pandas进行数据操作的著名书籍，其中有一章详细介绍了使用groupby和聚合方法进行数据分组。
- [pandas.DataFrame.agg](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html) — pandas development team (2023)
  .agg()方法的官方API参考，展示了如何在分组数据上执行多个和命名聚合。

---

[上一节](06-%E9%81%8D%E5%8E%86%E5%88%86%E7%BB%84.md) · [下一节](../10-%E7%BB%84%E5%90%88%20DataFrames/01-%E6%95%B0%E6%8D%AE%E5%90%88%E5%B9%B6%E4%BB%8B%E7%BB%8D.md)
