---
course: "julia-for-machine-learning"
chapter: "julia-unsupervised-learning-dimensionality"
lesson: "julia-pca-dimensionality-reduction"
sourceId: 6878
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality/julia-pca-dimensionality-reduction"
title: "主成分分析 (PCA) 用于降维"
description: "应用主成分分析 (PCA) 来减少数据集中的特征数量，同时保留重要信息。"
order: 3
plots: ["plots/6878-0.json"]
sourceHash: "269a63a2189324d152f19270185583c9e18c3afb0e29648eaeaa37447d62ea23"
sourceCorrections: []
---

处理数据集时，特别是那些拥有众多特征的数据集，你可能会遇到常被称为“维度灾难”的难题。高维空间 (high-dimensional space)的行为方式与我们对2D或3D空间的直观认识不同，这会增加机器学习 (machine learning)任务的难度。模型可能变得过于复杂，计算时间可能显著增加，并且模式也更难识别。主成分分析 (PCA) 是一种常用的技术，它通过减少数据集中的特征数量来解决这个问题，同时尽可能多地保留原始数据的变异性。

PCA 通过将原始特征转换为一组新的特征（称为主成分）来实现降维。这些主成分是原始特征的线性组合，且被设计为彼此不相关。名称中的“主”字表示了它们的重要性：第一主成分捕获数据中尽可能大的方差，第二主成分捕获次大的方差（在与第一主成分正交或垂直的前提下），以此类推。通过选择这些主成分的一个子集，通常是那些解释了大部分方差的主成分，你可以构建数据的低维表示。

想象你的数据点散布在一个多维云中。PCA 本质上是尝试找到最适合查看这个云的轴向。第一主成分是云团分布最广的方向。第二主成分是次广的方向，与第一主成分垂直。

PCA 的数学基础在于数据协方差矩阵的特征分解（或者数据矩阵的奇异值分解，后者在数值上通常更稳定）。
设 $X$ 为你的数据矩阵，其中行是样本，列是特征。

1. **标准化：** 通常建议先标准化你的数据（使每个特征的均值为零，标准差为一）。这一步很重要，因为尺度较大的特征可能对主成分产生不成比例的影响。
2. **协方差矩阵：** 标准化后，计算数据的协方差矩阵 $\Sigma$。这个矩阵描述了不同特征如何一起变化。对于包含 $n$ 个样本的标准化数据 $X_{std}$，其计算方式为：
   
   $$
   \Sigma = \frac{1}{n-1} X_{std}^T X_{std}
   $$
   
3. **特征分解：** 下一步是找出协方差矩阵 $\Sigma$ 的特征向量 (vector) $v$ 和特征值 $\lambda$。它们满足以下方程：
   
   $$
   \Sigma v = \lambda v
   $$
   
   特征向量表示主成分的方向。对应的特征值表示每个主成分所解释的方差量。
4. **投影：** 特征向量按其特征值降序排列。为了将维度从原始的 $d$ 维降低到 $k$ 维（其中 $k < d$），你选择前 $k$ 个特征向量（那些与最大特征值相关的）。这 $k$ 个特征向量构成一个新的基，通常表示为矩阵 $W_k$。原始标准化数据 $X_{std}$ 随后被投影到这个新的 $k$ 维子空间，得到转换后的数据 $X_{pca}$：
   
   $$
   X_{pca} = X_{std} W_k
   $$
   

结果是一个包含 $k$ 个特征的新数据集 $X_{pca}$。这些新特征是主成分，它们按从原始数据中捕获的方差量进行排序。

### 在 Julia 中使用 MultivariateStats.jl 实现 PCA

Julia 的生态系统为统计分析提供了强大的工具。对于 PCA，`MultivariateStats.jl` 包是常用的。让我们来看一个例子。

首先，确保你已安装 `MultivariateStats.jl`。如果没有，你可以使用 Julia 的包管理器添加它：

```julia
using Pkg
Pkg.add("MultivariateStats")
Pkg.add("Statistics") # 用于计算均值和标准差，如果手动标准化
Pkg.add("Random")     # 用于生成样本数据
```

现在，让我们将 PCA 应用到一个小型合成数据集上。我们将生成包含3个特征的数据，并将其降维到2个主成分。

```julia
using MultivariateStats
using Statistics
using Random

# 生成一些合成的3D数据
Random.seed!(42) # 用于重现性
n_samples = 100
X1 = randn(n_samples)
X2 = 0.5 * X1 + 0.5 * randn(n_samples)
X3 = 0.3 * X1 - 0.2 * X2 + 0.3 * randn(n_samples)

# MultivariateStats.jl 要求数据特征在行，样本在列 (d x n)
data = hcat(X1, X2, X3)' 

# 1. 标准化数据 (对 PCA 很重要)
# 对于特征在行的数据：
means = mean(data, dims=2)  # 计算每个特征（行）的均值
stds = std(data, dims=2)    # 计算每个特征（行）的标准差
data_std = (data .- means) ./ stds

# 2. 拟合 PCA 模型
# 我们可以指定输出维度数量 (maxoutdim)
# 或者，指定保留的方差比例 (例如, pratio=0.95)
# 这里，我们降维到2个维度。
M = fit(PCA, data_std, maxoutdim=2)

# 3. 将数据转换到新的低维空间
data_pca = transform(M, data_std)

# 转换后的数据 (data_pca) 现在有2个特征（行）和 n_samples 列。
# 如果你更喜欢样本作为行，特征作为列以便后续操作：
data_pca_samples_as_rows = data_pca'

println("原始数据维度 (特征 x 样本)：", size(data))
println("标准化数据维度：", size(data_std))
println("PCA 模型主成分 (投影矩阵 W)：")
# 投影矩阵包含特征向量作为列
# 它们定义了原始特征空间中主成分的方向
println(projection(M)) 
println("转换后数据维度 (新特征 x 样本)：", size(data_pca))
println("每个成分的解释方差比：", principalratio(M))
println("累积解释方差：", cumsum(principalratio(M)))
```

在此示例中，`data` 是一个 $3 \times 100$ 矩阵（3个特征，100个样本）。经过 PCA 后，`data_pca` 变为一个 $2 \times 100$ 矩阵。`projection(M)` 函数返回矩阵 $W_k$，其中包含了用于投影的特征向量 (vector)。`principalratio(M)` 提供了每个选定主成分所解释的方差比例，而 `cumsum(principalratio(M))` 则显示累积解释方差。

### 选择主成分的数量 ($k$)

应用 PCA 时，一个常见的问题是要保留多少个主成分。没有一个单一的确定性答案，但这里有两种广泛使用的方法：

1. **累积解释方差：** 确定你想要保留的总方差的阈值。例如，你可能旨在保留足够的成分来解释原始方差的90%、95%或99%。你可以计算解释方差比的累积和（如 Julia 示例中使用 `cumsum(principalratio(M))` 所示），然后选择满足你阈值的最小 $k$。
2. **碎石图：** 碎石图可视化了每个主成分的特征值（或解释方差的比例），按降序排列。通常，你会观察到前几个成分的解释方差急剧下降，随后较后面的成分趋于平稳。坡度变化的点常被称为“肘点”。一个启发式方法是保留肘点之前的成分，因为这些成分被认为捕获了最主要的方差。

这里是一个碎石图可能显示的内容示例，它说明了五个主成分解释的方差：



![碎石图：主成分解释方差](plots/6878-0.json)



> 碎石图显示了单个主成分（条形图）解释的方差比例和累积方差（折线图）。PC1 解释了 45% 的方差，PC2 解释了 30%，以此类推。“肘点”可能在 PC2 或 PC3 之后。例如，要捕获至少 90% 的方差，你会选择前三个主成分。

### PCA 的优点与考量

PCA 具有多个优点：

- **降维：** 它通过减少特征数量来简化你的数据集。这可以带来更简单的模型，这些模型更不易过拟合 (overfitting)，并且训练速度更快。
- **降噪：** 通过丢弃低方差成分，PCA 有时可以过滤掉数据中存在的噪声，因为这些低方差成分可能代表随机波动而非潜在结构。
- **可视化：** 它允许你将高维数据投影到2D或3D空间，从而可以可视化复杂数据集并可能识别模式或聚类。
- **多重共线性缓解：** 由于主成分是正交的（不相关的），PCA 可以帮助处理原始特征高度相关（多重共线性）的情况，这对某些建模技术而言可能是个问题。

然而，也有一些需要注意的地方：

- **可解释性：** 新的主成分是原始特征的线性组合。尽管它们在方差捕获方面在数学上是最佳的，但在原始问题领域中，它们可能并不总是具有清晰、直观的含义。
- **信息损失：** 降维本身就涉及一些信息损失。目标是通过保留捕获大部分方差的成分来最大程度地减少损失，但某些细节将不可避免地被舍弃。
- **尺度敏感性：** PCA 对特征的尺度很敏感。如果特征具有不同的尺度，方差较大的特征将主导主成分。如 Julia 示例所示，标准化数据是减轻此问题的一个常见且重要的预处理步骤。
- **线性假设：** PCA 是一种线性降维技术。它假设数据的潜在结构可以通过特征的线性组合很好地表示。如果你的数据具有复杂的非线性结构，PCA 可能不是最有效的降维技术。对于此类情况，非线性方法（如稍后将简要提及的 t-SNE 或核 PCA）可能更适合。

PCA 是数据分析和机器学习 (machine learning)中的一个基本技术，在处理具有大量特征的数据集时特别有用。它提供了一种系统的方法来降低复杂度，同时努力保留数据中最重要的信息。随着你工作的继续，你会发现 PCA 应用于各种情况，从为监督学习 (supervised learning)算法预处理数据到探索性数据分析和可视化。

## 参考资料

- [Analysis of a complex of statistical variables into principal components](https://doi.org/10.1037/h0071325) — Harold Hotelling (1933)
  Journal: Journal of Educational Psychology; Publisher: American Psychological Association; Volume: 24; Pages: 417-441, 498-520; DOI: [10.1037/h0071325](https://doi.org/10.1037/h0071325)
  这篇开创性论文介绍了主成分分析方法，作为降低多变量数据维度的工具。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://web.stanford.edu/~hastie/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer
  一本经典的统计学习方法教科书，其中包含广泛的内容，包括关于主成分分析等降维技术的详细章节（第2版）。
- [Principal Component Analysis (PCA) in MultivariateStats.jl](https://juliastats.org/MultivariateStats.jl/stable/pca/) — JuliaStats contributors (2024)
  Publisher: JuliaStats
  Julia `MultivariateStats.jl` 包的官方文档，提供了主成分分析实现的详细使用示例和API参考。
- [Machine Learning: A Probabilistic Perspective](https://mitpress.mit.edu/9780262018029/machine-learning/) — Kevin P. Murphy (2012)
  Publisher: The MIT Press
  这本全面的教科书提供了机器学习的深入理论处理，包括对主成分分析的详细数学和概率讨论。
