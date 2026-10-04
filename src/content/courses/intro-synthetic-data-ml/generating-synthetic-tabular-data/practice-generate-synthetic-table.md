---
course: "intro-synthetic-data-ml"
chapter: "generating-synthetic-tabular-data"
lesson: "practice-generate-synthetic-table"
sourceId: 5672
sourceUrl: "https://apxml.com/zh/courses/intro-synthetic-data-ml/chapter-3-generating-synthetic-tabular-data/practice-generate-synthetic-table"
title: "动手实践：生成一个合成表格"
description: "创建简单表格格式（例如 CSV）合成数据集的实践环节。"
order: 6
plots: ["plots/5672-0.json", "plots/5672-1.json"]
sourceHash: "d981dd27adc91df558b5a1c75c651ff5ebbc7bdbbf82943d5c9c5b89efad51c6"
sourceCorrections: []
---

生成一个包含多列的简单合成数据集，模拟一个基本的客户交易表。本练习重点在于独立生成每列值的方法，这些值从统计分布或预定义规则中获取。Python 以及常用的 `pandas` 和 `numpy` 库是数据处理的常用工具。

### 设置您的环境

首先，请确保您已安装 Python。您还需要 `pandas` 和 `numpy` 库。如果尚未安装，通常可以使用 pip 进行安装：

```bash
pip install pandas numpy
```

安装完成后，让我们将它们导入到我们的 Python 脚本或笔记本中：

```python
import pandas as pd
import numpy as np

print("库导入成功！")
```

### 定义合成表格的结构

假设我们需要一个代表客户交易的数据集。我们将创建一个包含以下列的表格：

- `CustomerID`：每个客户的唯一标识符（整数）。
- `Age`：客户的年龄（整数）。
- `ProductCategory`：所购产品的类别（文本/分类）。
- `PurchaseAmount`：交易中花费的金额（浮点数/数值）。

我们将为这个结构生成 100 行合成数据。

### 生成每列数据

我们将根据简单的规则或分布独立生成每列的数据。

#### 1. 生成客户 ID

对于 `CustomerID`，我们可以简单地创建一个从 1 到我们所需行数（本例中为 100）的唯一整数序列。

```python
num_rows = 100
customer_ids = np.arange(1, num_rows + 1)

# 显示前 5 个生成的 ID
print(customer_ids[:5])
```

#### 2. 生成客户年龄

假设客户年龄通常服从某种正态分布。我们可以使用 `numpy` 从正态分布中抽取年龄。我们将年龄集中在 35 岁左右，标准差为 10。由于年龄必须为正数且通常为整数，我们将取绝对值并转换为整数。

```python
# 从正态分布（均值=35，标准差=10）中抽取年龄
np.random.seed(42) # 用于可重现的结果
ages_float = np.random.normal(loc=35, scale=10, size=num_rows)

# 确保年龄为正数并转换为整数
ages = np.abs(ages_float).astype(int)

# 确保最小年龄，例如 18 岁
ages = np.maximum(ages, 18)

# 显示前 5 个生成的年龄
print(ages[:5])
```

*自我修正*：最初，我只使用了 `np.abs().astype(int)`。增加了 `np.maximum(ages, 18)` 使合成数据更可信，假设交易来自成年人。设置 `np.random.seed(42)` 使示例可重现。

#### 3. 生成产品类别

对于 `ProductCategory`，我们将定义一个可能的类别列表，并为每行从该列表中随机选择。我们还可以分配概率，使某些类别比其他类别更常见。

```python
categories = ['Electronics', 'Clothing', 'Groceries', 'Home Goods']
# 分配概率：电子产品 (30%)、服装 (25%)、杂货 (25%)、家居用品 (20%)
category_probabilities = [0.30, 0.25, 0.25, 0.20]

# 根据概率生成类别
product_categories = np.random.choice(categories, size=num_rows, p=category_probabilities)

# 显示前 5 个生成的类别
print(product_categories[:5])
```

#### 4. 生成购买金额

我们来生成 `PurchaseAmount`。为简单起见，我们可以使用均匀分布，假设购买金额范围从 $5 到$500。

```python
# 从 5 到 500 的均匀分布中生成购买金额
purchase_amounts = np.random.uniform(low=5.0, high=500.0, size=num_rows)

# 四舍五入到小数点后两位以表示货币
purchase_amounts = np.round(purchase_amounts, 2)

# 显示前 5 个生成的金额
print(purchase_amounts[:5])
```

### 组装合成表格

现在我们已经为每列生成了数据，我们可以将它们组合到一个 `pandas` DataFrame 中。DataFrame 本质上是一个表格，非常适合我们的结构化数据。

```python
# 创建一个字典来保存我们的数据
synthetic_data = {
    'CustomerID': customer_ids,
    'Age': ages,
    'ProductCategory': product_categories,
    'PurchaseAmount': purchase_amounts
}

# 创建 DataFrame
synthetic_df = pd.DataFrame(synthetic_data)

# 显示合成表格的前几行
print("合成表格的前 5 行：")
print(synthetic_df.head())
```

### 检查生成的表格

让我们快速检查一下使用一些基本的 `pandas` 函数创建的表格。

```python
# 获取列和数据类型的信息
print("\n表格信息：")
synthetic_df.info()

# 获取数值列的基本统计信息
print("\n基本统计信息：")
print(synthetic_df.describe())

# 获取分类列的值计数
print("\n产品类别计数：")
print(synthetic_df['ProductCategory'].value_counts())
```

您应该会看到输出，确认了数据类型（ID 和 Age 为整数，Category 为对象/字符串，Amount 为浮点数）以及数值列的均值、最小值、最大值等汇总统计信息，以及每个产品类别的计数。请注意，类别计数与我们指定的概率大致相符。

### 可视化合成数据

可视化可以帮助我们快速理解合成数据中的分布。我们来为 `Age` 列创建一个直方图。



![合成客户年龄分布](plots/5672-0.json)



> 直方图显示了合成数据集中不同年龄组的频率。分布峰值在 30 岁中期左右，符合我们的生成方法预期。

让我们也可视化 `PurchaseAmount` 的分布。



![合成购买金额分布](plots/5672-1.json)



> 直方图显示了不同购买金额范围的频率。分布看起来相对均匀，因为我们是从均匀分布中抽样的。

### 总结与后续步骤

恭喜！您已成功使用 Python 生成了一个简单的合成表格数据集。我们定义了一个结构，使用 `numpy` 独立为每列生成了数据（数值数据从分布中抽取，分类数据从带有概率的列表中抽取），并将其组装成一个 `pandas` DataFrame。

这种方法简单明了，但有局限性。最重要的是，独立生成列意味着我们没有明确地模拟列 *之间* 的任何关联（例如，年龄较大的客户可能更倾向于购买某些类别，或者不同类别的购买金额可能不同）。在真实数据中，这种关联常常存在。“保留基本列关联”一节介绍了这一挑战，更高级的方法旨在捕捉这些依赖关系。

这个实践练习提供了一个基础。您可以通过改变分布、概率或添加更多列来进行尝试。这个简单的合成表格现在可以用于数据加载流程的基本测试或简单的模型训练场景，同时要记住它的局限性。接下来的章节将涉及生成其他类型的数据，例如图像，并讨论如何评估我们创建的合成数据的质量。

## 参考资料

- [Random sampling (\`numpy.random\`)](https://numpy.org/doc/stable/reference/random/index.html) — NumPy Developers (2023)
  Publisher: NumPy Project
  NumPy随机数生成模块的官方文档，用于从统计分布中抽样。
- [pandas.DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) — Wes McKinney and the pandas development team (2023)
  pandas DataFrame的官方文档，详细介绍了其结构和数据操作方法。
- [Python for Data Analysis](https://wesmckinney.com/book/) — Wes McKinney (2022)
  Publisher: O'Reilly Media
  使用Python进行数据操作和分析的指南，重点介绍pandas和NumPy。
