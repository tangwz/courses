# 在 Julia 中应用特征工程

来源：[原文](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/julia-feature-engineering)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在 Julia 中应用特征工程技术对于提升机器学习 (machine learning)模型表现具有显著作用。从现有数据集中创建新的、有价值的特征通常能显著改变机器学习模型的表现。Julia，特别是与 `DataFrames.jl` 包结合使用时，为数据准备流程中的此环节提供了一个强大而灵活的环境。

### 从现有数据派生新特征

特征工程的技巧包括将原始数据转换为能更好反映学习算法所处理问题的格式。以下是常见的技术以及如何在 Julia 中实现它们：

#### 数值转换

原始数值特征通常可以通过转换来捕捉更复杂的关联或更好地符合模型假设。

- **多项式特征：** 如果您怀疑特征与目标变量之间存在非线性关联，那么多项式特征可能会有帮助。对于特征 $x$，您可能会创建 $x^2$、$x^3$ 等。在 Julia 中，使用 DataFrame 列上的元素级操作可以轻松实现这一点。

  ```julia
  using DataFrames

  # 假设 df 是一个现有 DataFrame，其中包含 :Age 列
  # df = DataFrame(Age = [25, 30, 35, 40, 45]) 
  # 创建一个 Age_Squared 特征：
  # df.Age_Squared = df.Age .^ 2
  ```

  这会在您的 DataFrame 中添加一个新列 `Age_Squared`，其中每个值都是对应 `Age` 的平方。
- **交互特征：** 这些特征通过组合两个或多个特征来创建，通常通过乘法或加法。它们可以捕捉协同效应，即特征的组合影响与其各自独立影响不同。对于特征 $x_1$ 和 $x_2$，交互特征可以是 $x_1 \cdot x_2$。

  ```julia
  # 假设 df 包含 :Price 和 :Quantity 列
  # df = DataFrame(Price = [10, 20, 15], Quantity = [2, 1, 3])
  # 创建一个 Total_Cost 特征（价格 * 数量）：
  # df.Total_Cost = df.Price .* df.Quantity
  ```

#### 增强类别特征

除了简单的一热编码或标签编码（在数据转换部分已介绍）之外，您还可以从类别数据中派生出更精细的特征。

- **频率编码：** 该技术将每个类别替换为其在数据集中的频率或计数。如果类别的普遍性具有信息量，该方法会很有用。

  ```julia
  using DataFrames

  # 假设 df 包含 :City 列
  # df = DataFrame(City = ["London", "Paris", "London", "Tokyo", "Paris", "London"])

  # 计算频率
  city_frequencies = combine(groupby(df, :City), nrow => :City_Frequency)

  # 将频率连接回原始 DataFrame
  # df = leftjoin(df, city_frequencies, on = :City)
  ```

  此操作后，`df` 将包含一个新列 `City_Frequency`（例如，对于“London”，其值将是 3）。
- **目标编码（均值编码）：** 这种强大的技术用该类别对应的目标变量的平均值来替换类别。例如，如果您正在预测房价，您可以用该城市中的平均房价来替换城市类别。

  **注意：** 如果未仔细实施，目标编码存在较高的数据泄露风险。编码应该*只*从训练集中得出，然后应用于验证/测试集。为了获得好的结果，它通常在交叉验证循环中执行。由于其正确实施时的复杂性，本节不包含详细的实现方法，但对于更高级的模型构建来说，这是一项值得了解的重要技术。

#### 从日期和时间中提取信息

如果您的数据集包含日期或时间列，提取其组成部分可以创建有价值的特征。Julia 中的 `Dates` 模块在此处必不可少。

```julia
using DataFrames, Dates

# 假设 df 包含 :TransactionDate 列（例如，Vector{Date} 类型）
# df = DataFrame(TransactionDate = [Date(2023,1,15), Date(2023,1,20), Date(2023,2,10)])

# 提取月份
df.Transaction_Month = month.(df.TransactionDate)

# 提取星期几（1=星期一，7=星期日）
df.Transaction_DayOfWeek = dayofweek.(df.TransactionDate)

# 提取年份
df.Transaction_Year = year.(df.TransactionDate)

# 您也可以创建布尔特征，例如 Is_Weekend
# df.Is_Weekend = dayofweek.(df.TransactionDate) .>= 6
```

这些新特征可以捕捉季节性、趋势或特定日期的模式。

#### 简单的文本派生特征

虽然全面的自然语言处理（NLP）是一个宽泛的话题，您可以从文本数据中提取简单但有效的特征：

- **文本长度：** 文本字段中的字符数量。
- **词数：** 文本字段中的词语数量。

```julia
using DataFrames

# 假设 df 包含 :CommentText 列
# df = DataFrame(CommentText = ["Great product!", "Not satisfied.", "Excellent service and quality."])

# 文本长度
df.Comment_Length = length.(df.CommentText)

# 词数（简单版本）
df.Comment_WordCount = length.(split.(df.CommentText)) # 按空格分割
```

### DataFrames.jl 在特征创建中的作用

`DataFrames.jl` 在 Julia 的特征工程中扮演核心角色。其函数支持灵活高效的列操作。

- **`transform` 和 `transform!`：** 这些函数对于基于现有列添加新列非常有用。`transform!` 会原地修改 DataFrame，而 `transform` 则返回一个新的 DataFrame。您可以将函数应用于列，通常使用 `ByRow` 进行逐行操作或使用广播。

  ```julia
  using DataFrames

  # df = DataFrame(A = [1, 2, 3], B = [4, 5, 6])

  # 使用 transform! 添加新列 C = A + B
  # transform!(df, [:A, :B] => ByRow(+) => :C)

  # 使用 transform 创建一个新 DataFrame，并添加列 D = A * 2
  # df_new = transform(df, :A => ByRow(x -> x * 2) => :D)
  ```
- **自定义函数：** 对于更复杂的特征逻辑，您可以定义自己的 Julia 函数并应用它们。

  ```julia
  using DataFrames

  # df = DataFrame(Value = [10, 25, 5, 40])

  function categorize_value(v)
      if v < 10
          return "Low"
      elseif v < 30
          return "Medium"
      else
          return "High"
      end
  end

  # 应用自定义函数创建新列 :Value_Category
  # df.Value_Category = categorize_value.(df.Value)
  # 或者使用 transform：
  # transform!(df, :Value => ByRow(categorize_value) => :Value_Category_Transform)
  ```

下图显示了如何使用不同类型的原始特征来生成新的、可能更有用的特征。

> 原始数据特征通过 Julia 中的各种工程技术处理，以生成可以增强模型理解和表现的新特征。

### 示例：特征创建实践

我们来看一个结合了其中一些技术的小例子。假设我们有一个产品销售数据集。

```julia
using DataFrames, Dates

# 初始 DataFrame
sales_df = DataFrame(
    ProductID = ["A01", "B02", "A01", "C03"],
    SalePrice = [19.99, 25.00, 18.50, 99.99],
    UnitsSold = [10, 5, 12, 2],
    SaleDate = [Date(2023, 3, 10), Date(2023, 3, 12), Date(2023, 4, 1), Date(2023, 4, 5)]
)

println("原始 DataFrame:")
println(sales_df)

# 1. 创建交互特征：TotalRevenue
sales_df.TotalRevenue = sales_df.SalePrice .* sales_df.UnitsSold

# 2. 从 SaleDate 中提取 SaleMonth
sales_df.SaleMonth = month.(sales_df.SaleDate)

# 3. 创建特征：Price_Per_Unit（如果是有意义的，否则跳过）
# 在此示例中，我们假设 SalePrice 已经是每单位的价格。
# 如果 SalePrice 是捆绑销售的价格，那么除以 UnitsSold 可能会创建这样的特征。

# 4. 对 ProductID 进行频率编码（在此批次中作为产品受欢迎程度的代表）
product_counts = combine(groupby(sales_df, :ProductID), nrow => :Product_SaleCount)
sales_df = leftjoin(sales_df, product_counts, on = :ProductID)

println("\n包含工程化特征的 DataFrame:")
println(sales_df)
```

**示例输出：**

Original DataFrame:

```
4×4 DataFrame
 Row │ ProductID  SalePrice  UnitsSold  SaleDate   
     │ String     Float64    Int64      Date       
─────┼─────────────────────────────────────────────
   1 │ A01            19.99         10  2023-03-10
   2 │ B02            25.0           5  2023-03-12
   3 │ A01            18.5          12  2023-04-01
   4 │ C03            99.99          2  2023-04-05
```

DataFrame with Engineered Features:

```
4×7 DataFrame
 Row │ ProductID  SalePrice  UnitsSold  SaleDate    TotalRevenue  SaleMonth  Product_SaleCount 
     │ String     Float64    Int64      Date        Float64       Int64      Int64             
─────┼──────────────────────────────────────────────────────────────────────────────────────────
   1 │ A01            19.99         10  2023-03-10        199.9           3                  2
   2 │ A01            18.5          12  2023-04-01        222.0           4                  2
   3 │ B02            25.0           5  2023-03-12        125.0           3                  1
   4 │ C03            99.99          2  2023-04-05        199.98          4                  1
```

此示例演示了如何添加诸如 `TotalRevenue`、`SaleMonth` 和 `Product_SaleCount` 之类的新列，每个新列都可能为机器学习 (machine learning)模型提供新的信号。

### 重要考量

- **向量 (vector)化操作：** 尽可能使用 Julia 的向量化 (quantization)操作（例如，使用点 `.` 语法，如 `df.A .* df.B`）或 `DataFrames.jl` 中对整列进行操作的函数。这通常比在用户代码中逐行迭代效率更高。
- **管道集成：** 您在此处构建的特征将成为机器学习 (machine learning)模型的输入。后续章节将讨论如何将这些预处理步骤集成到可复现的机器学习管道中，例如使用 `MLJ.jl`。新创建的数值特征本身可能需要缩放或归一化 (normalization)。
- **领域知识：** 有效的特征工程通常高度依赖于对数据和问题领域的理解。领域专业知识可以指导您创建真正有意义和预测性的特征。
- **迭代与评估：** 特征工程很少是一次性完成的过程。您可能会尝试创建多个新特征，训练模型，评估其表现，然后通过改进特征或创建新特征来迭代。

通过在 Julia 中应用这些技术，您可以显著提高输入数据的质量，为更准确的机器学习模型奠定基础。本章后面的实践部分将提供将这些技能应用于数据集的机会。

## 参考资料

- [Feature Engineering for Machine Learning: Principles and Techniques for Data Scientists](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) — Alice Zheng and Amanda Casari (2018)
  Publisher: O'Reilly Media
  一本关于特征工程的实用指南，涵盖了与本节内容相关的各种技术和最佳实践，包括数值和分类转换。
- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl developers (2024)
  DataFrames.jl 包的官方文档，提供了在 Julia 中用于数据操作和特征创建的函数的详细信息，如示例中所示。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://books.google.com/books/about/Hands_On_Machine_Learning_with_Scikit_Le.html?id=N7l3EAAAQBAJ) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  一本全面的机器学习指南，其中包含专门介绍数据准备、特征缩放和适用于不同机器学习场景的各种特征工程技术的部分。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本统计学习的理论基础书籍，提供了对各种建模技术的理论和实践见解，包括在回归模型中使用多项式和交互项。

---

[上一节](04-%E7%89%B9%E5%BE%81%E5%B7%A5%E7%A8%8B%E5%8E%9F%E5%88%99.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20Plots.jl%20%E5%92%8C%20Makie.jl%20%E8%BF%9B%E8%A1%8C%E6%95%B0%E6%8D%AE%E5%8F%AF%E8%A7%86%E5%8C%96.md)
