---
course: "julia-for-machine-learning"
chapter: "julia-data-manipulation-preparation"
lesson: "julia-data-cleaning-feature-practice"
sourceId: 6858
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/julia-data-cleaning-feature-practice"
title: "动手实践：数据清洗与特征创建"
description: "使用 Julia 的数据操作工具，在样本数据集上应用数据清洗技术并创建新特征。"
order: 7
plots: ["plots/6858-0.json", "plots/6858-1.json", "plots/6858-2.json"]
sourceHash: "c4d3093a14c40f5d58759f74c8e7c79984a85d58bf0ee76eef096778cba29dd9"
sourceCorrections: []
---

为了使原始数据集更适合机器学习 (machine learning)模型，应用清洗技术并构建新特征。这个动手练习将展示如何使用 `DataFrames.jl` 进行数据操作，以及其他有用的 Julia 包。

### 准备工作：我们的样本数据集

在此练习中，我们将处理一个小型、有代表性的产品销售数据集。假设这些数据来自一个电商平台。它包括产品信息、销售数据和客户评论。我们的目标是清洗这些数据并提取新的、可能具有预测性的特征。

首先，让我们确保拥有必要的 Julia 包。您主要需要 `DataFrames.jl`、`CSV.jl`（用于加载数据，尽管我们在此处内联定义）、`Statistics.jl` 用于计算（如均值）、`CategoricalArrays.jl` 用于分箱，以及 `Plots.jl` 用于可视化（尽管我们将使用 JSON 来呈现图表以便网页显示）。

```julia
using DataFrames, CSV, Statistics, CategoricalArrays # Plots.jl 用于实际绘图
```

让我们定义数据。在典型情况下，您会使用 `CSV.read("your_file.csv", DataFrame)` 从 CSV 文件加载数据。对于本例，我们将直接创建 DataFrame：

```julia
data_string = """
ProductID,Category,Price,UnitsSold,DiscountApplied,ReviewScore
1,Electronics,799.99,15,0.1,4.5
2,Books,24.50,120,,3.8
3,Electronics,1200.00,5,0.15,4.9
4,Home Goods,49.95,60,0.05,4.2
5,Books,19.99,200,0.0,
6,Electronics,99.99,30,0.07,2.5
7,Home Goods,89.00,25,0.1,9.9
8,Books,32.00,80,0.02,4.1
"""
df = CSV.read(IOBuffer(data_string), DataFrame, missingstrings=["", "NA"])

println("初始 DataFrame（前 5 行）：")
println(first(df, 5))

println("\nDataFrame 描述：")
println(describe(df, :eltype, :mean, :median, :nmissing))
```

`describe(df)` 的输出立即显示了一些情况：

- `DiscountApplied` 有一个缺失值。其类型为 `Union{Missing, Float64}`。
- `ReviewScore` 也有一个缺失值，类型为 `Union{Missing, Float64}`。
- 其他列如 `Price` 和 `UnitsSold` 看起来是完整的。

### 数据清洗：处理不完善之处

干净的数据是可靠机器学习 (machine learning)的根本。我们将处理缺失值和潜在异常值。

#### 处理缺失值

`DiscountApplied` 和 `ReviewScore` 中存在缺失数据。可以应用不同的策略。

1. **填充 `DiscountApplied`**:
   对于 `DiscountApplied`，缺失值可能表示未应用折扣，或者可能是数据输入错误。假设只有少量缺失值，使用均值或中位数进行填充是数值数据的常见方法。让我们使用均值。`skipmissing` 函数在此处有助于计算可用数据的统计信息。

   ```julia
   mean_discount = mean(skipmissing(df.DiscountApplied))
   println("平均折扣（忽略缺失值）：", round(mean_discount, digits=2))

   # 使用 coalesce 填充缺失值
   df.DiscountApplied = coalesce.(df.DiscountApplied, mean_discount)

   println("\n填充 DiscountApplied 后的 DataFrame（显示相关列）：")
   println(select(df, [:ProductID, :DiscountApplied, :ReviewScore]))
   println("\n填充后缺失值检查：")
   println(describe(df, :nmissing)) # 检查 DiscountApplied 的缺失值
   ```

   `coalesce` 函数在这里很有用；它返回其第一个非 `missing` 的参数 (parameter)。通过将其与原始列和计算出的均值进行广播操作，我们将 `missing` 条目替换为 `mean_discount`。
2. **处理缺失的 `ReviewScore`**:
   对于 `ReviewScore`，缺失值意味着我们不知道客户的意见。根据后续任务和缺失数据的量，我们可以填充它（例如，使用评论分数的中间值）或删除行。如果评论质量非常重要且填充可能引入偏差，那么删除可能是更安全的选择，前提是它不会丢弃过多数据。对于本例，让我们删除 `ReviewScore` 缺失的行。我们将在新的 DataFrame `df_cleaned` 上操作，以保留 `df`。

   ```julia
   df_cleaned = dropmissing(df, :ReviewScore)

   println("\n删除 ReviewScore 缺失行后的 DataFrame：")
   println(df_cleaned)
   println("\ndf_cleaned 的缺失值检查：")
   println(describe(df_cleaned, :nmissing))
   ```

   现在，`df_cleaned` 在 `ReviewScore` 中没有缺失值。我们丢失了一行数据（ProductID 5）。

#### 识别和处理异常值

异常值是与其他观测值显著不同的数据点。它们会扭曲统计分析并降低模型性能。`ReviewScore` 列在我们原始数据中有一个 `9.9` 的条目，如果分数通常在 1-5 的范围内，这看起来不寻常。

让我们使用箱线图可视化 `df_cleaned` 中的 `ReviewScore`（在钳制之前仍然包含 9.9 值），以帮助发现这一点。



![评论分数分布（钳制前）](plots/6858-0.json)



> 评论分数的箱线图。9.9 的点明显远离其他点，表明它可能是一个异常值或错误。

假设有效的评论分数应在 1 到 5 之间，我们可以限制 `ReviewScore` 的值。`clamp` 函数非常适合此操作。

```julia
df_cleaned.ReviewScore = map(s -> clamp(s, 1.0, 5.0), df_cleaned.ReviewScore)

println("\n钳制 ReviewScore (1.0 至 5.0) 后的 DataFrame：")
println(select(df_cleaned, [:ProductID, :ReviewScore]))
println("\n钳制 ReviewScore 后的描述：")
println(describe(df_cleaned, cols=:ReviewScore))
```

分数 `9.9` 现在调整为 `5.0`，这是我们有效范围的最大值。

### 特征创建：获取新信息

特征工程是利用领域知识从现有数据创建新特征的过程，以使机器学习 (machine learning)算法更好地工作。

#### 创建 `Revenue` (收入) 特征

我们的数据集包含 `Price`（价格）、`UnitsSold`（销售单位）和 `DiscountApplied`（应用折扣）。我们可以结合这些来计算每个产品销售产生的实际收入。
收入的公式可以是：$收入 = 价格 \times 销售单位 \times (1 - 应用折扣)$。

```julia
df_cleaned.Revenue = df_cleaned.Price .* df_cleaned.UnitsSold .* (1.0 .- df_cleaned.DiscountApplied)

println("\n包含新 '收入' 特征的 DataFrame：")
println(select(df_cleaned, [:ProductID, :Price, :UnitsSold, :DiscountApplied, :Revenue]))
```

请注意使用 `.` 进行广播式的元素级操作，这是 Julia 处理数组和 DataFrame 列的惯用方式。这个新的 `Revenue` 列对于回归任务来说可能是一个非常有用的目标变量，或者对于其他模型来说是一个有用的特征。

#### 将 `Price` 分箱到类别中

有时，连续数值特征在转换为分类箱后更有用。例如，我们可以将产品分为 `低价`、`中价` 或 `高价`。这可以通过 `CategoricalArrays.jl` 中的 `cut` 函数来完成。

```julia
price_bins = [0, 50, 500, Inf] # 边界：(0, 50], (50, 500], (500, Inf]
price_labels = ["低", "中", "高"]
df_cleaned.PriceCategory = cut(df_cleaned.Price, price_bins, labels=price_labels, extend=true)

println("\n包含 'PriceCategory' 特征的 DataFrame：")
println(select(df_cleaned, [:ProductID, :Price, :PriceCategory, :Revenue]))
```

`cut` 函数中的 `extend=true` 参数 (parameter)确保边界上的值被正确包含。这个 `PriceCategory` 特征可能会显示出与价格区间相关的销售或评论模式。

#### 从 `Category` 创建布尔特征

我们可以创建一个特定特征，标记 (token)产品是否属于某个类别。例如，一个 `IsElectronic` 特征。

```julia
df_cleaned.IsElectronic = (df_cleaned.Category .== "Electronics")

println("\n包含 'IsElectronic' 特征的 DataFrame：")
println(select(df_cleaned, [:ProductID, :Category, :IsElectronic]))
```

这个布尔特征可以直接被许多机器学习算法使用。

### 最终检查与可视化

让我们看看我们完全处理过的 DataFrame，并可视化一些新特征。

```julia
println("\n最终处理的 DataFrame (df_cleaned)：")
println(df_cleaned)
```

现在，让我们可视化新 `Revenue` 特征的分布。



![计算收入的分布](plots/6858-1.json)



> 直方图显示了 `Revenue` 特征的分布。大多数产品收入处于较低范围，其中一个产品收入明显较高。

以及 `PriceCategory` 特征的计数：

```julia
price_category_counts = by(df_cleaned, :PriceCategory, :Count => length) # 为清晰起见重命名

# 对于绘图，我们需要 x 和 y 的实际值
# df_cleaned 中的 PriceCategory 值：高, 低, 高, 低, 中, 中, 低
# 计数：低: 3, 中: 2, 高: 2
```



![按价格类别划分的产品数量](plots/6858-2.json)



> 条形图显示了每个定义的价格类别中的产品数量：低、中和高。

### 行动总结

在本动手实践部分，我们完成了典型的数据准备流程：

1. **加载**数据并进行初步检查，以了解其结构并识别即时问题。
2. **清洗**数据，通过：
   - 使用列均值填充缺失的 `DiscountApplied` 值。
   - 删除 `ReviewScore` 缺失的行。
   - 识别并限制 `ReviewScore` 列中的异常值。
3. **构建新特征**，通过：
   - 从现有销售数据计算 `Revenue`（收入）。
   - 将 `Price`（价格）分箱到 `PriceCategory`（价格类别）。
   - 从 `Category`（类别）列创建布尔型 `IsElectronic`（是否电子产品）标记 (token)。

这些步骤已将我们的原始数据集转换为更结构化且可能信息更丰富的格式，可用于机器学习 (machine learning)项目的后续阶段，例如模型训练和评估。有效的数据准备通常是一个迭代过程，此处显示的技术为处理各种数据质量问题和丰富数据集提供了扎实的基础。

## 参考资料

- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — DataFrames.jl Developers (2023)
  Julia 中数据操作的官方指南，包含数据清洗和转换功能。
- [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3) — Max Kuhn and Kjell Johnson (2013)
  Publisher: Springer; DOI: [10.1007/978-1-4614-6849-3](https://doi.org/10.1007/978-1-4614-6849-3)
  本书详细解释了数据预处理、特征选择和特征工程技术。
- [Hands-On Julia for Data Science](https://www.packtpub.com/product/hands-on-julia-for-data-science/9781838640061) — Tom Kwong (2020)
  Publisher: Packt Publishing
  一本实用书籍，涵盖使用 Julia 包进行数据操作、清洗和准备，适合初学者。
- [Feature Engineering for Machine Learning: Principles and Techniques for Building Predictive Models](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  提供从原始数据创建有效特征的具体方法和策略。
