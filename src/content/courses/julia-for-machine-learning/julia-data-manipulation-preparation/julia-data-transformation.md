---
course: "julia-for-machine-learning"
chapter: "julia-data-manipulation-preparation"
lesson: "julia-data-transformation"
sourceId: 6850
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/julia-data-transformation"
title: "数据转换：缩放、编码和分箱"
description: "在 Julia 中应用常用的数据转换方法，如特征缩放、分类编码和数据分箱。"
order: 3
plots: ["plots/6850-0.json", "plots/6850-1.json"]
sourceHash: "206b9cd6783a96fba6e26331d66609fe56afd87ed659f7acdc405c56f1fa6733"
sourceCorrections: []
---

数据转换是为机器学习 (machine learning)算法准备数据的关键步骤。原始的、干净的数据并非总是机器学习算法的最佳格式。不同的算法对输入数据有不同的要求。例如，有些算法对输入特征的尺度很敏感，而另一些则要求所有输入都是数值型的。Julia中三种常见且重要的数据转换技术包括：数值特征缩放、分类特征编码和连续数据分箱。

## 数值特征缩放

当数值输入特征处于相似的尺度时，许多机器学习 (machine learning)算法表现会更好或收敛更快。例如，计算数据点之间距离的算法，如k近邻（k-NN）和支持向量 (vector)机（SVM），或使用梯度下降 (gradient descent)的算法，如线性回归和神经网络 (neural network)，都可能对特征缩放敏感。如果特征的范围不同（例如，一个特征范围从0到1，另一个从0到1,000,000），那么范围较大的特征可能会主导计算结果。

### 标准化（Z-score 归一化 (normalization)）

标准化对特征进行重新缩放，使其平均值($\mu$)为0，标准差($\sigma$)为1。转换公式如下：

$X_{scaled} = \frac{X - \mu}{\sigma}$

此方法应用广泛，特别适用于数据服从高斯（正态）分布的情况。它通常比最小-最大缩放对异常值不那么敏感。

在 Julia 中，您可以使用 `MLJ.jl` 中的 `Standardizer` 模型来执行此转换。`MLJ.jl` (Machine Learning Julia) 是一个用于 Julia 机器学习 (machine learning)的全面框架，其模型通常遵循 `fit!`/`transform` 模式。

我们来看一个例子。首先，确保您已加载 `MLJ` 和 `DataFrames`：

```julia
using MLJ, DataFrames
import StableRNGs.StableRNG # for 重现性

# 示例数据
rng = StableRNG(123)
X_df = DataFrame(
    age = rand(rng, 25:65, 10),
    income = rand(rng, 30000:150000, 10)
)

# 转换为 MLJ 可以使用的表格
X = MLJ.table(X_df)
```

现在，让我们对 `age` 和 `income` 特征应用标准化：

```julia
# 初始化 Standardizer 模型
scaler_model = Standardizer()

# 将模型封装到 machine 中并拟合数据
scaler_machine = machine(scaler_model, X)
fit!(scaler_machine)

# 转换数据
X_scaled = MLJ.transform(scaler_machine, X)

# X_scaled 现在是包含标准化特征的表格
# 您可以将其转换回 DataFrame 进行查看
X_scaled_df = DataFrame(X_scaled)
println(X_scaled_df)
```

```
Output:
10×2 DataFrame
 Row │ age        income    
     │ Float64    Float64   
─────┼────────────────────
   1 │ -0.0706283   1.2155
   2 │  1.48319    -0.893144
   3 │ -1.62449    -1.28456
   4 │ -1.03332     0.370213
   5 │  1.12564     1.33237
   6 │  0.686526   -1.07185
   7 │ -0.854546    0.781878
   8 │  0.865303    0.16541
   9 │  0.108148   -0.80376
  10 │ -0.525828    0.18795
```

MLJ 中的 `Standardizer` 会自动识别连续特征进行标准化。

为了可视化效果，请考虑标准化前后具有不同尺度的两个特征：



![特征缩放效果](plots/6850-0.json)



> 该图显示了两个最初尺度不同的特征在标准化后如何被带到可比较的尺度，并以零为中心。

### 最小-最大缩放（归一化 (normalization)）

最小-最大缩放将特征重新缩放到固定范围，通常是 [0, 1]。转换公式如下：

$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$

当您需要将数据限制在有界区间内，或者算法不假设数据的任何特定分布时，这会很有用。然而，它可能对异常值敏感，因为 $X_{min}$ 和 $X_{max}$ 用于计算中。

虽然 MLJ 的核心没有像其他一些框架那样直接的 `MinMaxScaler`，但您可以自行实现它，或者使用扩展的 MLJ 生态系统中的模型（如 `MLJScikitLearnInterface.jl`）。为了教学目的，我们来看看如何使用基本的 Julia 函数将其应用于单个特征：

```julia
# 单个特征的示例（例如 X_df 中的 income）
feature_to_scale = X_df.income

function min_max_scale(col)
    min_val = minimum(col)
    max_val = maximum(col)
    if min_val == max_val # 如果所有值都相同，避免除以零
        return zeros(length(col)) 
    end
    return (col .- min_val) ./ (max_val - min_val)
end

income_min_max_scaled = min_max_scale(feature_to_scale)
# println(income_min_max_scaled)
```

这将 `income` 特征缩放到 [0,1] 范围。对于完整的表格和集成到 MLJ 工作流程中，您通常会将此类逻辑封装到自定义转换器中，或者使用提供此功能的包。`MLJ.ContinuousEncoder` 也可以用于将特征转换到 `[0,1]` 区间，尽管其主要机制（经验 CDF）与标准最小-最大缩放不同。

### 关于拟合缩放器的重要说明

机器学习 (machine learning)中的一种常见做法是**仅在训练数据上**拟合缩放器（或任何预处理器）。然后，使用已*拟合*的缩放器来转换训练数据、验证数据和任何新的测试数据。这可以防止来自验证集或测试集的信息“泄露”到训练过程中，从而可能导致过于乐观的性能估计。MLJ 的 `machine` 抽象在与数据分区一起使用时有助于正确管理这一点。

## 分类特征编码

机器学习 (machine learning)算法通常需要数值输入。分类特征，例如“颜色”（值为“红色”、“绿色”、“蓝色”等）或“城市”（例如“纽约”、“伦敦”、“东京”），需要转换为数值表示。

Julia 的 `CategoricalArrays.jl` 包常用于处理分类数据。`MLJ.jl` 与此集成得很好，您通常会在应用编码器之前使用 `coerce` 函数来确保您的分类列具有正确的科学类型（例如 `Multiclass` 或 `OrderedFactor`）。

```julia
# 示例分类数据
X_cat_df = DataFrame(
    id = 1:5,
    color = ["Red", "Green", "Blue", "Green", "Red"],
    grade = ["A", "C", "B", "A", "D"] # 假设这是序数型
)

# 将 'color' 强制转换为 Multiclass（名义型），将 'grade' 强制转换为 OrderedFactor（序数型）
X_cat_coerced = coerce(X_cat_df, :color => Multiclass, :grade => OrderedFactor)
```

### 独热编码

独热编码是名义分类特征（其中类别没有固有顺序）的常用技术。它为每个独特的类别创建一个新的二元（0或1）特征。

例如，如果“颜色”特征有“红色”、“绿色”、“蓝色”等类别：

- “红色”可能变为 `[1, 0, 0]`
- “绿色”可能变为 `[0, 1, 0]`
- “蓝色”可能变为 `[0, 0, 1]`

这避免了暗示类别之间的任何序数关系。主要缺点是，如果分类变量有许多独特的值，它会显著增加特征数量（维度）。

MLJ 为此提供了 `OneHotEncoder`：

```julia
# 假设 X_cat_coerced 来自前面的示例
encoder_model = OneHotEncoder()
encoder_machine = machine(encoder_model, MLJ.table(X_cat_coerced))
fit!(encoder_machine)

X_encoded = MLJ.transform(encoder_machine, MLJ.table(X_cat_coerced))
X_encoded_df = DataFrame(X_encoded)
# println(X_encoded_df) # 检查独热编码输出
```

`OneHotEncoder` 默认转换科学类型为 `Multiclass` 的特征。`id` 列（连续型）和 `grade` 列（序数型）通常会被传递或以不同方式处理（例如，如果指定，`grade` 由 `ContinuousEncoder` 处理）。您可以使用 `OneHotEncoder(features=[:feature1, :feature2])` 指定要编码的特征。

### 序数编码

对于序数分类特征（其中类别有自然顺序，如“低”、“中”、“高”或教育水平），分配数值排名（例如，0、1、2）是合适的。这通常被称为标签编码或整数编码。

在 MLJ 中，如果特征被正确强制转换为 `OrderedFactor`，`ContinuousEncoder` 模型可以用于将这些有序类别转换为数值，通常是从1开始的整数，或缩放到 `[0,1]`。

```julia
# 使用 X_cat_coerced，其中 'grade' 是 OrderedFactor
# 默认级别基于出现顺序或指定级别
# 对于 'grade'：A、C、B、D。如果顺序为 A < B < C < D，则强制转换应指定级别。
# X_cat_coerced = coerce(X_cat_df, :grade => OrderedFactor(levels=["D", "C", "B", "A"]))

# ContinuousEncoder 将转换 OrderedFactor 特征
# 对于指定为 OrderedFactor 的特征，ContinuousEncoder 将它们映射为整数
ord_encoder_model = ContinuousEncoder()
ord_encoder_machine = machine(ord_encoder_model, MLJ.table(X_cat_coerced))
fit!(ord_encoder_machine)

X_ord_encoded = MLJ.transform(ord_encoder_machine, MLJ.table(X_cat_coerced))
X_ord_encoded_df = DataFrame(X_ord_encoded)
# println(X_ord_encoded_df.grade)
```

`ContinuousEncoder` 将 `OrderedFactor` 特征转换为 `Continuous` 特征。除非您对 `Multiclass` 特征使用 `OneHotEncoder`，否则它默认会直接传递 `Multiclass` 特征。`Standardizer` 默认也会对 `Multiclass` 特征进行独热编码。这通常是关于在管道中组合这些转换器。

## 连续特征分箱（离散化）

分箱，或称离散化，是指将连续的数值特征转换为离散的分类特征（箱）。这有多种用处：

- 它可以帮助捕获非线性关系。例如，收入可能对目标变量没有线性影响，但收入*区间*可能有。
- 它可以减少数据中小波动或噪声的影响。
- 某些算法可能更适合或要求分类输入。

### 等宽分箱

等宽分箱将特征的范围划分为指定数量的箱，每个箱具有相同的宽度。
宽度计算公式为 $(X_{max} - X_{min}) / N_{bins}$。

MLJ 为此目的提供了 `UnivariateDiscretizer`。它实质上创建了 `N_{bins}` 个类别。

```julia
# 示例连续数据（例如 X_df 中的 'age' 特征）
age_feature = X_df.age
data_for_binning = DataFrame(age = age_feature)

# 用于等宽分箱的 UnivariateDiscretizer 模型
# 假设我们想要 3 个箱
binner_model = UnivariateDiscretizer(n_classes=3) # n_classes 指定箱的数量
binner_machine = machine(binner_model, data_for_binning)
fit!(binner_machine)

age_binned = MLJ.transform(binner_machine, data_for_binning)
age_binned_df = DataFrame(age_binned)
# println(age_binned_df.age) # 显示每个原始年龄的 CategoricalValue
```

输出 `age_binned_df.age` 将包含代表箱的分类值。

以下是等宽分箱后连续特征分布如何变化的视觉表示：



![等宽分箱对年龄分布的影响](plots/6850-1.json)



> 第一个直方图显示了年龄的原始分布。第二个显示了将相同数据分组到三个等宽箱后的情况。

### 等频分箱（分位数分箱）

等频分箱将连续特征划分为箱，使得每个箱包含大约相同数量的观测值。箱的边缘由分位数确定（例如，4个箱使用四分位数，10个箱使用十分位数）。这种方法对于偏斜的数据分布可能更有效。

`MLJ.UnivariateDiscretizer` 主要处理等宽分箱。对于基于分位数的分箱，您可以使用 `StatsBase.jl` 中的函数（如 `quantile` 来查找箱边缘），然后应用这些分割，或者查看像 `FeatureTransforms.jl` 这样提供更复杂离散化选项的包。

例如，使用 `StatsBase.quantile` 定义边缘：

```julia
using StatsBase

# 使用之前的数据 age_feature
quantiles = quantile(age_feature, [0, 0.25, 0.5, 0.75, 1.0]) # 用于 4 个箱（四分位数）
# 然后您将使用这些 `quantiles` 作为自定义箱边缘。
# 这通常需要手动映射或在 MLJ 中使用自定义转换器。
```

### 选择箱的数量

箱的数量是一个重要参数 (parameter)。箱太少可能导致信息丢失，而箱太多可能无法提供所需的平滑或泛化效果。选择通常取决于领域知识、实验或启发式方法，如斯特奇斯公式或弗里德曼-迪亚科尼斯规则，尽管这些只是起始点。

## 实践中的转换

数据转换，如缩放、编码和分箱，是为机器学习 (machine learning)准备数据的基本步骤。在 `MLJ.jl` 中，这些转换通常被封装为模型，并且可以集成到更大的机器学习管道中。这使您能够定义一系列预处理步骤以及最终的预测模型，确保数据在训练、评估和预测过程中得到一致的转换。您将在第5章了解更多关于构建此类管道的信息。

请记住，所需的具体转换将取决于您的数据以及您计划使用的机器学习算法。始终将从训练集中学到的转换一致地应用于任何新数据，以保持完整性并避免数据泄露。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, and Jerome Friedman (2009)
  Publisher: Springer
  为机器学习算法提供了严格的统计和计算基础，涵盖了缩放和转换等数据预处理技术的理论原理。
- [MLJ.jl Documentation](https://mlj.sciml.ai/stable/) — The MLJ Developers (2025)
  Machine Learning Julia框架的官方文档，详细介绍了用于数据转换的`Standardizer`、`OneHotEncoder`、`ContinuousEncoder`和`UnivariateDiscretizer`模型的使用方法。
- [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125965/) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 850
  提供机器学习概念的实用指南，包括特征缩放、分类编码和分箱的详细解释和应用，这些技术具有普遍适用性，尽管代码示例是Python。
