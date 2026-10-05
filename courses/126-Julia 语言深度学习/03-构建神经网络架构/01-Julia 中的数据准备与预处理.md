# Julia 中的数据准备与预处理

来源：[原文](https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/data-preparation-preprocessing-julia)

[返回章节目录](README.md) · [返回课程目录](../README.md)

在构建像多层感知器 (MLP)、卷积神经网络 (neural network) (CNN) 或循环神经网络 (RNN) 这样复杂的神经网络架构之前，您必须先确保您的数据处于合适的状态。“垃圾进，垃圾出”这句老话在深度学习 (deep learning)中尤其适用。高质量、准备得当的数据是训练高效模型的前提。将介绍使用 Julia 强大的数据操作和科学计算环境进行数据准备和预处理的基本方法。这将包括清理数据、转换特征以及构建数据集，使其可供 Flux.jl 使用。

### 必要的第一步：准备数据

数据准备是指将原始数据集转换为神经网络 (neural network)可以处理的干净、有结构的数据格式。这通常包括以下几个步骤：

1. **加载数据**：将数据从文件（如 CSV）或数据库读取到 Julia 结构中，通常是 `DataFrame`。
2. **清理数据**：处理缺失值、纠正错误和消除不一致。
3. **数据转换**：缩放数值特征、编码类别特征，并可能创建新特征。
4. **数据构建**：将数据重塑为神经网络层所需的特定数组格式，并将其分成训练集、验证集和测试集。

Julia 凭借 `DataFrames.jl`、`CSV.jl`、`Statistics.jl` 和 `MLUtils.jl` 等包，为这些任务提供了高效且富有表达力的环境。其性能特点在处理深度学习 (deep learning)中常见的海量数据集时尤其有利。

我们来看一个典型的数据准备流程。

> 深度学习项目中数据准备的一般流程。

### 常见预处理操作

神经网络 (neural network)对输入数据的呈现方式很敏感。以下是一些您会遇到的常见预处理操作。

#### 处理缺失值

"数据集常含有缺失值。您如何处理它们会明显影响模型性能。"

- **识别**：在 `DataFrames.jl` 中，缺失值由 `missing` 表示。您可以使用 `ismissing()` 等函数来识别它们，或者使用 `describe(df, :nmissing)` 获取摘要。

  ```julia
  using DataFrames, Statistics

  # 包含缺失值的示例 DataFrame
  data = DataFrame(A = [1, 2, missing, 4, 5], B = [missing, 0.2, 0.3, 0.4, 0.5])
  println(describe(data, :nmissing))
  ```
- **策略**：

  - **删除**：如果只有少量行包含缺失值且数据集很大，您可以删除这些行 (`dropmissing(df)`)。如果某列有太多缺失值且不那么重要，您可以删除该列。
  - **填充**：填充缺失值。常见策略包括：
    - **均值/中位数填充**：用列的均值或中位数替换缺失的数值。如果数据有异常值，通常更倾向于使用中位数。
    - **众数填充**：用列的众数（最频繁的值）替换缺失的类别值。
    - **基于模型的填充**：使用其他特征预测并填充缺失值（更高级）。

  ```julia
  # 用列 A 的均值填充缺失值
  mean_A = mean(skipmissing(data.A))
  data.A = coalesce.(data.A, mean_A)

  # 用特定值（例如 0.0）填充列 B 的缺失值
  replace!(data.B, missing => 0.0)
  println(data)
  ```

  `coalesce` 函数很实用，因为它返回第一个非 `missing` 参数 (parameter)。

#### 特征缩放：将数据规整到特定范围

神经网络，特别是那些使用基于梯度的优化器训练的神经网络，当输入特征在相似的尺度上时，通常表现更好，收敛更快。特征范围的较大差异可能导致训练过程不稳定。

- **最小-最大缩放（归一化 (normalization)）**：将特征重新缩放到固定范围，通常是 [0, 1] 或 [-1, 1]。缩放到 [0, 1] 的公式是：

  
  $$
  X_{\text{缩放后}} = \frac{X - X_{\text{最小值}}}{X_{\text{最大值}} - X_{\text{最小值}}}
  $$
  

  其中 $X_{\text{最小值}}$ 和 $X_{\text{最大值}}$ 分别是该特征的最小值和最大值。

  ```julia
  # 数值特征示例
  feature = [10.0, 20.0, 15.0, 30.0, 25.0]

  function min_max_scale(X)
      X_min = minimum(X)
      X_max = maximum(X)
      return (X .- X_min) ./ (X_max - X_min)
  end

  scaled_feature = min_max_scale(feature)
  # scaled_feature 将是 [0.0, 0.5, 0.25, 1.0, 0.75]
  println("原始数据: ", feature)
  println("最小-最大缩放后: ", scaled_feature)
  ```
- **标准化（Z-分数归一化）**：将特征重新缩放，使其均值 ($\mu$) 为 0，标准差 ($\sigma$) 为 1。公式是：

  
  $$
  X_{\text{标准化}} = \frac{X - \mu}{\sigma}
  $$
  

  与最小-最大缩放相比，此方法受异常值的影响较小。

  ```julia
  using Statistics

  # 数值特征示例
  feature = [10.0, 20.0, 15.0, 30.0, 25.0]

  function standardize(X)
      mu = mean(X)
      sigma = std(X)
      return (X .- mu) ./ sigma
  end

  standardized_feature = standardize(feature)
  println("标准化后: ", standardized_feature)
  # 检查均值和标准差（应接近 0 和 1）
  println("标准化后的均值: ", mean(standardized_feature))
  println("标准化后的标准差: ", std(standardized_feature))
  ```

最小-最大缩放和标准化之间的选择取决于数据和神经网络架构。对于图像，像素值通常缩放到 [0, 1]。对于其他类型的数据，标准化是常见的默认设置。



[交互图表：最小-最大缩放效果](https://apxml.com/zh/courses/julia-deep-learning/chapter-3-constructing-nn-architectures/data-preparation-preprocessing-julia#plot-1b45f8f)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "x": [
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        10,
        20,
        15,
        30,
        25
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "原始数据",
      "marker": {
        "color": "#4dabf7"
      }
    },
    {
      "x": [
        1,
        2,
        3,
        4,
        5
      ],
      "y": [
        0.0,
        0.5,
        0.25,
        1.0,
        0.75
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "最小-最大缩放 (0-1)",
      "yaxis": "y2",
      "marker": {
        "color": "#20c997"
      }
    }
  ],
  "layout": {
    "title": "最小-最大缩放效果",
    "xaxis": {
      "title": "样本索引"
    },
    "yaxis": {
      "title": "原始值",
      "titlefont": {
        "color": "#4dabf7"
      },
      "tickfont": {
        "color": "#4dabf7"
      }
    },
    "yaxis2": {
      "title": "缩放值",
      "overlaying": "y",
      "side": "right",
      "range": [
        0,
        1
      ],
      "titlefont": {
        "color": "#20c997"
      },
      "tickfont": {
        "color": "#20c997"
      }
    },
    "legend": {
      "x": 0.05,
      "y": 0.95
    },
    "autosize": true,
    "height": 350
  }
}
```

</details>



> 特征值在最小-最大缩放前后的比较。缩放后的值被映射到辅助 y 轴上的 [0,1] 范围。

#### 编码类别特征

神经网络需要数值输入。类别特征（例如“颜色”：“红色”、“蓝色”、“绿色”，或“城市”：“纽约”、“伦敦”）必须转换为数值格式。

- **整数编码（标签编码）**：为每个类别分配一个唯一的整数。例如，“红色” -> 0，“蓝色” -> 1，“绿色” -> 2。
  这种方法简单，但可能暗示一个不存在的序数关系（例如，绿色 > 蓝色 > 红色）。它适用于序数数据（例如，“低”、“中”、“高”）。

  ```julia
  categories = ["cat", "dog", "bird", "cat", "dog"]
  unique_cats = unique(categories)
  cat_to_int = Dict(c => i for (i, c) in enumerate(unique_cats))
  encoded_integers = [cat_to_int[c] for c in categories]
  # 示例: 如果 unique_cats = ["cat", "dog", "bird"]
  # cat_to_int = Dict("cat"=>1, "dog"=>2, "bird"=>3)
  # encoded_integers = [1, 2, 3, 1, 2]
  println("整数编码后: ", encoded_integers)
  ```
- **独热编码**：为每个唯一的类别创建一个新的二进制（0 或 1）特征。对于给定样本，对应其类别的特征为 1，所有其他特征为 0。
  示例：
  “红色” -> `[1, 0, 0]`
  “蓝色” -> `[0, 1, 0]`
  “绿色” -> `[0, 0, 1]`
  这种方法避免了强加人为的顺序，但如果唯一类别很多，可能会导致高维特征空间。`Flux.jl` 提供了 `Flux.onehot` 和 `Flux.onehotbatch` 来实现此功能。

  ```julia
  using Flux

  # 使用与上述相同的类别
  # 假设 unique_cats = ["cat", "dog", "bird"]
  # Flux.onehotbatch(data, labels)
  # 'data' 是类别向量，'labels' 是唯一的、已排序的类别列表
  one_hot_encoded = Flux.onehotbatch(categories, unique_cats)
  # 这会生成一个 Flux.OneHotMatrix
  # 若要将其视为常规矩阵:
  # Matrix(one_hot_encoded)
  # 对于 categories = ["cat", "dog", "bird", "cat", "dog"]
  # 以及 unique_cats = ["bird", "cat", "dog"] 的结果 (如果未提供排序，Flux 会内部排序标签)
  # 它将是:
  # 0  1  0  0  1
  # 1  0  0  1  0
  # 0  0  1  0  0
  # (如果 "bird" 是标签 1, "cat" 标签 2, "dog" 标签 3 (排序后))
  # 为清晰起见，我们明确定义标签
  explicit_labels = ["cat", "dog", "bird"] # 顺序对于解释很重要
  one_hot_encoded_explicit = Flux.onehotbatch(categories, explicit_labels)
  println("独热编码后 (矩阵表示):")
  display(Matrix(one_hot_encoded_explicit)) # display 可以很好地打印矩阵
  # explicit_labels = ["cat", "dog", "bird"] 时的输出:
  # 1  0  0  1  0  (cat)
  # 0  1  0  0  1  (dog)
  # 0  0  1  0  0  (bird)
  ```

对于基数非常高（许多唯一值）的特征，例如用户 ID 或词汇表 (vocabulary)中的单词，整数编码和独热编码都不是理想选择。在这种情况下，通常使用嵌入 (embedding)层，我们将在本章后面讨论。

### 构建神经网络 (neural network)输入数据

特征清理和转换后，整个数据集需要构建成 `Flux.jl` 模型可以处理的格式。这通常意味着将数据转换为特定数值类型的 Julia `Array`，通常是 `Float32`，以提高深度学习 (deep learning)库和 GPU 的效率。

- **输入形状**：不同的神经网络层期望特定维度的输入：

  - **全连接层**：通常期望一个矩阵，其中每列是一个样本，行是特征。如果您有 $N$ 个样本和 $F$ 个特征，则输入矩阵将是 $F \times N$。
  - **卷积层（用于图像）**：通常期望一个 4D 数组：`(width, height, channels, batch_size)`，通常缩写为 WHCN。
  - **循环层（用于序列）**：期望序列，通常是一个矩阵，其中列是时间步长，行是每个时间步长的特征，或者是一个序列向量 (vector)。
- **重塑数据**：您通常会使用 `reshape` 来将数据调整为正确的维度。例如，如果您有一组扁平的图像向量，您可以将它们重塑为 CNN 的 WHCN 格式。

  ```julia
  # 假设您有 100 张图像，每张都是 28x28 灰度图
  # 并且它们被加载为 100 个矩阵（28x28）的向量
  # images_vector = [rand(Float32, 28, 28) for _ in 1:100];

  # 对于 Flux，最好使用单个 4D 数组：(W, H, C, N)
  # W=28, H=28, C=1（灰度图）, N=100
  # 示例:
  num_samples = 100
  img_width = 28
  img_height = 28
  channels = 1 # 灰度图

  # 扁平化数据（例如，来自 CSV，其中每行都是一张图像）
  # 每行有 28*28 = 784 像素。共 100 行。
  # 如果特征是行，这将是 784x100 矩阵
  # 如果样本是行，则是 100x784。假设样本是行。
  flat_data_as_rows = rand(Float32, num_samples, img_width * img_height)

  # 要与 Flux Dense 层一起使用，我们需要特征 x 样本：(784, 100)
  data_for_dense = permutedims(flat_data_as_rows, (2,1)) # 变为 784x100

  # 要与 Flux Conv 层一起使用，我们需要 WHCN：(28, 28, 1, 100)
  # 首先，确保数据是 (features, samples) 即 784x100
  # 然后将每列（样本）重塑为 WxHxC
  data_for_cnn = reshape(data_for_dense, img_width, img_height, channels, num_samples)

  println("Dense 层的形状: ", size(data_for_dense))
  println("CNN 层的形状: ", size(data_for_cnn))
  ```
- **数据类型转换**：确保您的数据类型为 `Float32`（如果精度非常重要，则使用 `Float64`，但 `Float32` 是深度学习的标准）。

  ```julia
  # 如果 data_matrix 是 Array{Float64, 2}
  # data_matrix_f32 = Float32.(data_matrix)
  ```

### 分割数据集

训练前的一个重要步骤是将数据集分成至少两个，最好是三个子集：

1. **训练集**：用于训练模型。模型通过根据这些数据调整其参数 (parameter)来学习。
2. **验证集**：用于调整超参数 (hyperparameter)（如学习率、层数）和进行早期停止。它在调优过程中对模型在训练集上的拟合程度提供无偏评估。
3. **测试集**：用于对训练后的模型在未见过的数据上的性能进行最终的、无偏的评估。此集合应仅在模型完全训练和调优后使用一次。

常见的分割比例是训练集占 60-80%，验证集占 10-20%，测试集占 10-20%。`MLUtils.jl` 提供了 `splitobs` 来实现此功能。

```julia
using MLUtils

# 假设 `features` 是您的输入数据（例如，一个矩阵）
# `labels` 是您的目标数据（例如，一个向量或矩阵）
# features = rand(Float32, 10, 1000) # 10 个特征，1000 个样本
# labels = rand(Float32, 1, 1000)   # 1 个输出，1000 个样本

# 示例:
num_total_samples = 1000
X_data = rand(Float32, 5, num_total_samples) # 5 个特征
Y_data = Flux.onehotbatch(rand(1:3, num_total_samples), 1:3) # 3 个类别

# 分割为训练集 (70%) 和测试集 (30%)
(X_train, Y_train), (X_test, Y_test) = splitobs((X_data, Y_data), at=0.7, shuffle=true)

# 进一步将测试集分割为验证集和测试集（例如，从原始数据中分出 15% 用于验证，15% 用于测试）
# 原始测试集占总数的 30%。我们想将其 50/50 分割。
# 因此，当前 X_test 的 0.5 份将用于验证。
(X_val, Y_val), (X_test_final, Y_test_final) = splitobs((X_test, Y_test), at=0.5, shuffle=false) # 通常这里不需要打乱

println("训练样本数: ", size(X_train, 2))
println("验证样本数: ", size(X_val, 2))
println("测试样本数: ", size(X_test_final, 2))
```

在分割时，特别是对于类别不平衡的分类任务，使用*分层抽样*是一种好的做法。这有助于保证每个类别的比例在训练集、验证集和测试集中大致相同。如果标签提供得当，`MLUtils.splitobs` 通常可以处理这种情况，或者您可能需要使用 `MLJ.jl` 中更专业的工具，或实现自定义逻辑进行分层。

### 简要工作流程示例

让我们用一个小型数据集将其中一些想法结合起来。假设我们有关于水果的数据：

| 重量 (克) | 颜色 | 质地 | 是否甜 |
| --- | --- | --- | --- |
| 150 | 红色 | 光滑 | 1 |
| 120 | 绿色 | 光滑 | 0 |
| 160 | 红色 | 凹凸 | 1 |
| missing | 黄色 | 光滑 | 1 |

我们的目标是预测 `IsSweet`（1 表示甜，0 表示不甜）。

1. **加载和表示数据（为清晰起见，使用 DataFrame）**

   ```julia
   using DataFrames, Statistics, Flux

   df = DataFrame(
       Weight = [150.0, 120.0, 160.0, missing],
       Color = ["Red", "Green", "Red", "Yellow"],
       Texture = ["Smooth", "Smooth", "Bumpy", "Smooth"],
       IsSweet = [1, 0, 1, 1]
   )
   ```
2. **处理缺失值**（填充 `Weight`）

   ```julia
   mean_weight = mean(skipmissing(df.Weight))
   df.Weight = coalesce.(df.Weight, mean_weight)
   println("处理缺失值后的 DataFrame:")
   display(df)
   ```
3. **特征缩放**（最小-最大缩放 `Weight`）

   ```julia
   # 使用之前定义的函数
   function min_max_scale_col(X_col)
       X_min = minimum(X_col)
       X_max = maximum(X_col)
       return (X_col .- X_min) ./ (X_max - X_min)
   end
   df.Weight_scaled = min_max_scale_col(df.Weight)
   ```
4. **编码类别特征**（`Color`、`Texture`）使用独热编码

   ```julia
   unique_colors = unique(df.Color)
   color_onehot = Flux.onehotbatch(df.Color, unique_colors)

   unique_textures = unique(df.Texture)
   texture_onehot = Flux.onehotbatch(df.Texture, unique_textures)

   # 将 OneHotArrays 转换为常规矩阵以便组合
   color_matrix = Float32.(Matrix(color_onehot))
   texture_matrix = Float32.(Matrix(texture_onehot))
   ```
5. **将特征组合成一个矩阵（特征 x 样本）**

   ```julia
   # 缩放后的重量（1 个特征）
   weight_feature_matrix = reshape(Float32.(df.Weight_scaled), 1, nrow(df))

   # 组合所有特征矩阵
   # 特征是行，样本是列
   X_matrix = vcat(weight_feature_matrix, color_matrix, texture_matrix)
   println("\n最终特征矩阵 X (Float32):")
   display(X_matrix)
   println("X 的大小: ", size(X_matrix))

   # 准备标签 Y（1 x 样本）
   Y_matrix = reshape(Float32.(df.IsSweet), 1, nrow(df))
   println("\n标签矩阵 Y (Float32):")
   display(Y_matrix)
   println("Y 的大小: ", size(Y_matrix))
   ```

   这个 `X_matrix`（特征）和 `Y_matrix`（标签）现在处于适合作为 Flux 模型输入的格式。它们是 `Float32` 数组，`X_matrix` 中的特征按行排列，样本按列排列。

严谨的数据准备和预处理是根本。这些步骤可以保证您的模型以信息丰富且计算友好的格式接收数据。在数据清理、转换并正确构建之后，您现在可以考虑如何在训练期间高效地加载和处理它，这是下一节关于 `MLUtils.jl` 的主题。

## 参考资料

- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  提供了深度学习的全面理论基础，强调了数据预处理对于模型稳定性和性能的重要性。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125974/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media
  提供了数据预处理的实用示例和方法，包括特征缩放、分类编码和缺失值处理，适用于各种机器学习任务。
- [Flux.jl Documentation](https://fluxml.ai/Flux.jl/stable/) — The Flux.jl Community (2025)
  官方文档详细介绍了如何为 Flux.jl 神经网络模型准备和构造数据，包括独热编码。
- [MLUtils.jl Documentation](https://juliaml.github.io/MLUtils.jl/stable/) — The MLUtils.jl Community (2025)
  MLUtils.jl 的官方文档，解释了其数据处理工具，例如将数据集划分为训练集、验证集和测试集。
- [DataFrames.jl Documentation](https://dataframes.juliadata.org/stable/) — The DataFrames.jl Community (2024)
  DataFrames.jl 的官方文档，涵盖了 Julia 中数据加载、操作以及处理缺失值的有效策略。

---

[上一节](../02-Flux.jl%20%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%85%A5%E9%97%A8/09-%E5%AE%9E%E8%B7%B5%E6%93%8D%E4%BD%9C%EF%BC%9A%E4%BD%BF%E7%94%A8%20Flux%20%E6%9E%84%E5%BB%BA%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E5%9B%9E%E5%BD%92%E5%99%A8.md) · [下一节](02-%E5%A4%84%E7%90%86%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%9A%E4%BD%BF%E7%94%A8%20MLUtils.jl%20%E7%9A%84%E8%BF%AD%E4%BB%A3%E5%99%A8%E5%92%8C%E5%8A%A0%E8%BD%BD%E5%99%A8.md)
