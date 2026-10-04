# 实践：使用 Pandas 整理数据

来源：[原文](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/practice-pandas-wrangling)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据整理（即清洗、转换和组织原始数据的过程）是任何数据分析或机器学习 (machine learning)项目的主要组成部分。在实际场景中，使用 Pandas 练习应用数据操作。动手练习有助于巩固对使用 Pandas 加载、清洗、转换、分组和合并数据的理解。

我们将通过一系列针对示例数据集的任务进行操作，模拟您可能遇到的常见工作流程。

### 设置练习 DataFrame

让我们从创建一个代表产品销售数据的示例 DataFrame 开始。该数据集包含常见问题，如缺失值和不一致的数据类型，我们需要处理这些问题。

```python
import pandas as pd
import numpy as np

# 示例销售数据
data = {
    'ProductID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010'],
    'Category': ['Electronics', 'Clothing', 'Electronics', 'Home Goods', 'Clothing', 'Electronics', 'Home Goods', None, 'Clothing', 'Electronics'],
    'Price': [599.99, 49.95, 1200.00, 25.50, 35.00, 750.80, None, 15.00, 55.00, 1150.50],
    'UnitsSold': [50, 120, 15, 200, 150, 25, 80, 300, 110, None],
    'Discount': ['5%', '10%', '5%', 'None', '15%', '7%', '10%', '5%', '12%', '8%']
}

sales_df = pd.DataFrame(data)

# 显示初始 DataFrame
print("初始销售数据：")
print(sales_df)

# 显示数据类型和非空计数
print("\n初始 DataFrame 信息：")
sales_df.info()
```

### 任务 1：初步检查和基本清洗

首先，查看 DataFrame。使用 `.info()`（如上所示）和 `.describe()` 来理解其结构、数据类型和摘要统计。找出潜在问题，如缺失值 (NaN 或 None) 和不正确的数据类型。

```python
# 获取数值列的摘要统计
print("\n摘要统计：")
print(sales_df.describe())

# 检查缺失值
print("\n每列的缺失值：")
print(sales_df.isnull().sum())
```

您会注意到 `Category`、`Price` 和 `UnitsSold` 中存在缺失值。`Discount` 列也存储为对象（字符串）而非数值，这会阻止直接计算。

### 任务 2：处理缺失数据

处理在任务 1 中找出的缺失值。针对不同列，可能需要采用不同的策略。

1. **类别：** 由于 `Category` 是分类数据，我们可以用 'Unknown' 这样的占位符或众数（最常见的类别）填充缺失值。我们使用 'Unknown'。
2. **价格：** 对于 `Price` 等数值列，用平均值或中位数填充是常见的做法。考虑到可能存在的异常值（如 P003），中位数可能更适合。
3. **销售单位：** 为了本练习的简单起见，我们用平均销售单位数填充缺失的 `UnitsSold`。

```python
# 用 'Unknown' 填充缺失的 Category
sales_df['Category'].fillna('Unknown', inplace=True)

# 计算价格中位数（不包括 NaN）并填充缺失的 Price
median_price = sales_df['Price'].median()
sales_df['Price'].fillna(median_price, inplace=True)

# 计算销售单位平均值（不包括 NaN）并填充缺失的 UnitsSold
mean_units_sold = sales_df['UnitsSold'].mean()
sales_df['UnitsSold'].fillna(mean_units_sold, inplace=True)

# 验证缺失值已处理
print("\n处理后的缺失值：")
print(sales_df.isnull().sum())
```

### 任务 3：数据转换和特征工程

现在，让我们清理现有列并创建新的、可能有所帮助的列。

1. **转换折扣：** 将 `Discount` 列从字符串百分比转换为表示折扣因子的数值浮点数（例如，'5%' 变为 0.05）。适当处理 'None' 值，可能将其转换为 0% 折扣。
2. **计算总收入：** 创建一个新列 `TotalRevenue`，计算方式为 `价格 * 销售单位`。
3. **计算折扣后收入：** 使用 `TotalRevenue` 和数值型 `Discount` 因子创建另一个列 `DiscountedRevenue`。
4. **将销售单位转换为整数：** 由于 `UnitsSold` 代表计数，在处理缺失值后将其转换为整数类型。

```python
# 将 Discount 字符串转换为数值浮点数
# 将 'None' 替换为 '0%'，移除 '%'，转换为浮点数，并除以 100
sales_df['Discount'] = sales_df['Discount'].replace('None', '0%')
sales_df['DiscountFactor'] = sales_df['Discount'].str.replace('%', '').astype(float) / 100.0

# 将 UnitsSold 转换为整数类型
sales_df['UnitsSold'] = sales_df['UnitsSold'].astype(int)

# 创建 TotalRevenue 列
sales_df['TotalRevenue'] = sales_df['Price'] * sales_df['UnitsSold']

# 创建 DiscountedRevenue 列
sales_df['DiscountedRevenue'] = sales_df['TotalRevenue'] * (1 - sales_df['DiscountFactor'])

# 显示转换后的 DataFrame 结构和示例数据
print("\n转换后的 DataFrame：")
print(sales_df.info())
print("\n转换后的示例数据：")
print(sales_df.head())
```

### 任务 4：分组和聚合

使用 `groupby()` 方法通过基于类别聚合数据来获得信息。

1. **计算每类别平均价格：** 找出每个类别中产品的平均价格。
2. **计算每类别总销售单位：** 对每个类别中的销售单位数量求和。
3. **多重聚合：** 同时计算每个类别的总收入和平均折扣因子。

```python
# 每类别平均价格
avg_price_per_category = sales_df.groupby('Category')['Price'].mean()
print("\n每类别平均价格：")
print(avg_price_per_category)

# 每类别总销售单位
total_units_per_category = sales_df.groupby('Category')['UnitsSold'].sum()
print("\n每类别总销售单位：")
print(total_units_per_category)

# 多重聚合：每类别总收入和平均折扣
category_summary = sales_df.groupby('Category').agg(
    TotalRevenue=('TotalRevenue', 'sum'),
    AverageDiscount=('DiscountFactor', 'mean')
)
print("\n类别摘要（总收入和平均折扣）：")
print(category_summary)
```

### 任务 5：合并 DataFrame

假设您有另一个包含每个产品商店信息的数据集。让我们为此创建一个小的 DataFrame 并将其与我们的 `sales_df` 合并。

```python
# 示例商店数据
store_data = {
    'StoreID': ['S1', 'S2', 'S1', 'S3', 'S2', 'S3', 'S1', 'S2', 'S3', 'S1'],
    'ProductID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008', 'P009', 'P010']
}
store_df = pd.DataFrame(store_data)

print("\n商店信息：")
print(store_df)

# 基于 ProductID 合并 sales_df 和 store_df
full_df = pd.merge(sales_df, store_df, on='ProductID', how='left')

print("\n合并后的 DataFrame（销售 + 商店信息）：")
print(full_df.head())
```

此合并操作根据共同的 `ProductID` 列，为每个产品行添加了 `StoreID`。

### 任务 6：快速可视化

让我们使用分组数据创建一个简单的可视化图表。我们可以使用 Plotly Express（通常导入为 `px`）绘制每类别总销售单位。这提供了我们聚合结果的快速视觉摘要。

```python
import plotly.express as px

# 准备绘图数据（使用之前计算的 Series）
plot_data = total_units_per_category.reset_index() # 将 Series 转换为 DataFrame

# 创建条形图
fig = px.bar(plot_data, x='Category', y='UnitsSold',
             title='每产品类别总销售单位',
             color='Category',
             color_discrete_map={ # 可选：使用课程调色板
                 'Electronics': '#1c7ed6',
                 'Clothing': '#d6336c',
                 'Home Goods': '#0ca678',
                 'Unknown': '#adb5bd'
             },
             labels={'UnitsSold': '总销售单位'})

# 显示图表（在笔记本/兼容环境中）或显示 JSON
# fig.show() # 如果在支持渲染图表的环境中运行，请取消此行的注释

# 显示用于网页渲染的 Plotly JSON
plotly_json = fig.to_json()
print("\nPlotly 图表 JSON：")
```



[交互图表：每产品类别总销售单位](https://apxml.com/zh/courses/intermediate-python-programming-ml/chapter-3-data-manipulation-pandas/practice-pandas-wrangling#plot-1180qle)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "marker": {
        "color": "#1c7ed6",
        "pattern": {
          "shape": ""
        }
      },
      "name": "电子产品",
      "orientation": "v",
      "showlegend": true,
      "text": "33",
      "type": "bar",
      "x": [
        "Electronics"
      ],
      "xaxis": "x",
      "y": [
        40
      ]
    },
    {
      "marker": {
        "color": "#d6336c",
        "pattern": {
          "shape": ""
        }
      },
      "name": "服装",
      "orientation": "v",
      "showlegend": true,
      "text": "120<br>150<br>110",
      "type": "bar",
      "x": [
        "Clothing"
      ],
      "xaxis": "x",
      "y": [
        380
      ]
    },
    {
      "marker": {
        "color": "#0ca678",
        "pattern": {
          "shape": ""
        }
      },
      "name": "家居用品",
      "orientation": "v",
      "showlegend": true,
      "text": "200<br>80",
      "type": "bar",
      "x": [
        "Home Goods"
      ],
      "xaxis": "x",
      "y": [
        280
      ]
    },
    {
      "marker": {
        "color": "#adb5bd",
        "pattern": {
          "shape": ""
        }
      },
      "name": "未知",
      "orientation": "v",
      "showlegend": true,
      "text": "300",
      "type": "bar",
      "x": [
        "Unknown"
      ],
      "xaxis": "x",
      "y": [
        300
      ]
    }
  ],
  "layout": {
    "legend": {
      "tracegroupgap": 0,
      "orientation": "v"
    },
    "margin": {
      "t": 60
    },
    "template": "plotly",
    "title": {
      "text": "每产品类别总销售单位"
    },
    "xaxis": {
      "anchor": "y",
      "domain": [
        0.0,
        1.0
      ],
      "title": {
        "text": "类别"
      }
    },
    "yaxis": {
      "anchor": "x",
      "domain": [
        0.0,
        1.0
      ],
      "title": {
        "text": "总销售单位"
      }
    }
  }
}
```

</details>



> 条形图显示按产品类别分组的销售单位总和。

### 实践总结

通过这些任务，您已练习了：

- 检查 DataFrame 以理解其内容并找出问题。
- 使用不同的填充策略 (`fillna`) 处理缺失数据。
- 清理数据类型并转换列（例如，字符串操作，使用 `.astype()` 进行类型转换）。
- 从现有数据创建新特征（特征工程）。
- 使用 `groupby()` 聚合数据以按类别汇总信息。
- 使用 `pd.merge()` 组合数据集。
- 从处理过的数据生成简单的可视化图表。

这些步骤代表了数据准备中的常见顺序。掌握这些 Pandas 操作对于有效准备数据以进行后续分析或机器学习 (machine learning)模型训练非常重要。您可以随意修改示例数据或将这些技术应用于您遇到的其他数据集。

## 参考资料

- [pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html) — pandas development team (2024)
  Pandas库用于数据操作和分析的官方指南，内容全面。
- [Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython](https://books.google.com/books/about/Python_for_Data_Analysis.html?id=zI41DwAAQBAJ) — Wes McKinney (2017)
  Publisher: O'Reilly Media; Pages: 547
  由pandas创建者所著，是理解Python中pandas进行数据分析和操作的基础书籍。与涵盖的主题高度相关。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125970/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本广受好评的书，包含了在端到端机器学习流程中，使用pandas进行数据准备和特征工程的实用示例。

---

[上一节](08-Pandas%E4%B8%AD%E7%9A%84%E6%97%B6%E9%97%B4%E5%BA%8F%E5%88%97%E6%95%B0%E6%8D%AE%E5%A4%84%E7%90%86.md) · [下一节](../04-%E4%BD%BF%E7%94%A8%20Matplotlib%20%E5%92%8C%20Seaborn%20%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96/01-Matplotlib%20%E7%BB%98%E5%9B%BE%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86.md)
