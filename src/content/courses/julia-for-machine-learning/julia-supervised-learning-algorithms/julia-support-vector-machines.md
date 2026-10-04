---
course: "julia-for-machine-learning"
chapter: "julia-supervised-learning-algorithms"
lesson: "julia-support-vector-machines"
sourceId: 6865
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-3-julia-supervised-learning-algorithms/julia-support-vector-machines"
title: "使用 Julia 包实现支持向量机 (SVM)"
description: "学习在 Julia 机器学习生态系统中实现支持向量机用于分类问题。"
order: 4
plots: ["plots/6865-0.json"]
sourceHash: "27cecbeb56ab77fc9ad66a8bc58e137d7a9b8be7545fc6214797a110f29efbae"
sourceCorrections: []
---

支持向量 (vector)机（SVM）是一类多功能且强大的监督学习 (supervised learning)算法，用于分类和回归任务，尽管它们最常与分类相关联。支持向量机背后的基本思想是找到一个最优超平面，它能在特征空间中最佳地分离属于不同类别的数据点。这种“最优性”是通过最大化边距来实现的，边距是超平面与任何类别最近数据点之间的距离。这些最近的点被称为支持向量，因为它们是支持或定义超平面的核心元素。

支持向量机在高维空间 (high-dimensional space)中特别有效，即使维度数量大于样本数量，它们也能表现良好。它们还通过使用不同的核函数提供了灵活性，使其能够对非线性决策边界进行建模。

### 支持向量 (vector)机的核心思想

在查看 Julia 实现之前，我们先巩固几个核心思想：

- **超平面**：在 $p$ 维空间中，超平面是一个维度为 $p-1$ 的平坦仿射子空间。对于二维（$p=2$）中的二分类任务，超平面是一条直线。在三维（$p=3$）中，它是一个平面。对于 $p > 3$，它是一个超平面。支持向量机寻找能产生最佳分离的超平面。
- **支持向量**：这些是距离决策边界（超平面）最近的数据点。它们是最难分类的点，并直接影响超平面的位置和方向。如果这些点移动，超平面也会随之移动。
- **边距**：边距是超平面与两侧最近支持向量之间的距离。支持向量机的目标是最大化此边距。通常，更大的边距与在未见过数据上更好的泛化性能相关联。
- **核函数**：对于在其原始特征空间中非线性可分离的数据，支持向量机采用“核技巧”。核函数是将输入数据转换为更高维空间 (high-dimensional space)（在那里可能实现线性分离）的函数。常见的核函数包括：
  - **线性**：适用于线性可分离数据。$K(x_i, x_j) = x_i^T x_j$。
  - **多项式**：$K(x_i, x_j) = (\gamma x_i^T x_j + r)^d$。
  - **径向基函数（RBF）**：$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$。这是一个非常常用且灵活的核函数。
  - **Sigmoid**：$K(x_i, x_j) = \tanh(\gamma x_i^T x_j + r)$。
- **正则化 (regularization)（C 参数 (parameter)）**：参数 $C$（通常称为成本或惩罚参数）控制着最大化边距和最小化训练数据上的分类误差之间的平衡。
  - 较小的 $C$ 值会促使模型产生更大的边距，即使这意味着会错误分类一些训练点。这可以带来更简单的决策边界，并可能提高泛化能力。
  - 较大的 $C$ 值会更严重地惩罚错误分类，导致边距变小，决策边界会尝试正确分类所有训练点。这有时可能导致过拟合 (overfitting)。

下图显示了支持向量机模型中涉及的主要组成部分：

> 上图显示了输入数据、支持向量机的目标以及超平面、支持向量、边距和核函数等主要组成部分之间的关系。

### 使用 MLJ.jl 在 Julia 中实现支持向量 (vector)机

在 Julia 中，支持向量机实现可通过外部包获得，并且 MLJ.jl 提供了一致的接口来使用它们。提供支持向量机最常用的包是 `LIBSVM.jl`，它是广泛使用的 LIBSVM C++ 库的封装。您通常会通过 `MLJLIBSVMInterface.jl` 与它交互。

我们来一步步实现一个用于分类任务的支持向量机。

首先，请确保您已拥有必要的包。如果您的 Julia 环境中尚未安装 `MLJLIBSVMInterface`，您需要添加它：

```julia
# 在 Julia REPL 中，按 ']' 进入 Pkg 模式
# pkg> add MLJLIBSVMInterface
# 按 Backspace 退出 Pkg 模式
```

现在，我们来设置我们的 Julia 脚本：

```julia
using MLJ
using DataFrames, Random, StableRNGs

# 从 MLJLIBSVMInterface 加载支持向量分类器 (SVC) 模型类型
# modest=false 返回模型类型本身，而不是实例。
SVC = @load SVC pkg=LIBSVM modest=false
# 对于专用线性支持向量机，如果数据线性可分离，它会更快：
LinearSVC = @load LinearSVC pkg=LIBSVM modest=false

# 为了结果可复现，使用一个稳定的 RNG
rng = StableRNG(123)

# 生成用于分类的合成 2D 数据
# X 将是特征，y 将是分类标签
X_raw, y = make_blobs(150, 2; centers=2, cluster_std=0.9, rng=rng, as_table=false)
X = DataFrame(X_raw, :auto) # 将数据转换为 DataFrame 以供 MLJ 使用
```

`make_blobs` 函数生成各向同性高斯斑点，用于聚类或分类。在这里，我们正在二维空间中创建两组不同的点。下图显示了此类生成数据的一个示例。



![用于 SVM 分类的示例 2D 数据](plots/6865-0.json)



> 一个包含两个不同类别的合成 2D 数据的散点图，适用于训练支持向量机分类器。

#### 使用线性核函数训练支持向量机

如果您怀疑数据是线性可分离的，或者需要一个基准，那么线性核函数是一个不错的开始。`LinearSVC` 为此进行了优化，但您也可以将 `SVC` 与线性核函数一起使用。

```julia
# 实例化一个线性支持向量机模型
# LinearSVC 对于线性问题通常更快。
# 它内部使用 LIBSVM 的线性求解器。
linear_svm_model = LinearSVC(cost=1.0) # 'cost' 是 C 参数

# 或者，使用 SVC：
# linear_svm_model = SVC(kernel=LIBSVM.Kernel.LINEAR, cost=1.0)

# 创建一个 MLJ 机器
mach_linear_svm = machine(linear_svm_model, X, y)

# 训练机器
fit!(mach_linear_svm, verbosity=0)

# 进行预测
y_pred_linear = predict_mode(mach_linear_svm, X)

# 评估（评估指标将在另一部分详细介绍）
# 例如，计算错误分类率：
accuracy_linear = accuracy(y_pred_linear, y)
println("线性 SVM 准确度: $(round(accuracy_linear, digits=3))")
```

这里的 `cost` 参数 (parameter)是正则化 (regularization)参数 $C$。`cost=1.0` 是一个常见的默认值。

#### 使用 RBF 核函数训练支持向量机

对于非线性可分离数据，RBF 核函数因其灵活性而成为常用选择。

```julia
# 实例化一个使用 RBF 核函数的支持向量机模型
rbf_svm_model = SVC(kernel=LIBSVM.Kernel.RADIAL, # RADIAL 即 RBF
                    cost=1.0,      # 正则化参数 C
                    gamma=0.5)     # RBF 的核函数系数

# `LIBSVM.Kernel` 提供对核函数类型的访问：
# LIBSVM.Kernel.LINEAR, LIBSVM.Kernel.POLY, LIBSVM.Kernel.RADIAL, LIBSVM.Kernel.SIGMOID

# 创建一个 MLJ 机器
mach_rbf_svm = machine(rbf_svm_model, X, y)

# 训练机器
fit!(mach_rbf_svm, verbosity=0)

# 进行预测
y_pred_rbf = predict_mode(mach_rbf_svm, X)

accuracy_rbf = accuracy(y_pred_rbf, y)
println("RBF 核 SVM 准确度: $(round(accuracy_rbf, digits=3))")
```

在这个 `SVC` 模型中：

- `kernel=LIBSVM.Kernel.RADIAL` 指定 RBF 核函数。
- `cost=1.0` 是正则化参数 $C$。
- `gamma=0.5` 特定于 RBF 和多项式等核函数。对于 RBF 核函数，$K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$，gamma 定义了单个训练示例的影响程度。
  - **低 gamma 值** 意味着更大的相似半径，因此距离较远的点也被认为是相似的。这会产生更平滑的决策边界。
  - **高 gamma 值** 意味着更小的相似半径，因此只有彼此靠近的点才被认为是相似的。这可能导致更复杂、“曲折”的边界，可能会使训练数据过拟合 (overfitting)。

您可以使用 `params(model_name)` 检查模型的所有可调超参数 (hyperparameter)：

```julia
# println(params(rbf_svm_model))
```

核函数的选择以及 $C$ 和 $\gamma$ 等超参数的值对于支持向量机的性能非常重要。这些通常使用交叉验证和超参数调优等技术确定，本章后面会详细介绍。

### 支持向量 (vector)机的优点和需要考虑的因素

**优点：**

- **在高维空间 (high-dimensional space)中有效**：即使特征数量很大，支持向量机也能很好地工作。
- **内存效率高**：它们在决策函数中只使用训练点的一个子集（支持向量）。
- **多功能**：可以为决策函数指定不同的核函数。常见的核函数为各种数据类型提供灵活性。
- **泛化能力好**：边距最大化目标通常能使模型在未见过数据上获得良好的泛化能力，尤其是在经过良好调优时。

**需要考虑的因素：**

- **计算成本**：对于非常大的数据集，训练可能很慢，尤其是在使用非线性核函数时。对于某些实现，其复杂度可能在 $O(n^2 p)$ 到 $O(n^3 p)$ 之间，这里的 $n$ 是样本数量，$p$ 是特征数量。
- **核函数和超参数 (parameter) (hyperparameter)选择**：性能高度依赖于核函数的选择及其参数（例如 $C$, $\gamma$）。这通常需要仔细调优。
- **可解释性**：支持向量机，尤其是使用非线性核函数时，生成的模型不如决策树那样容易解释。
- **缩放**：支持向量机对特征缩放很敏感。通常建议在训练支持向量机之前缩放数据（例如，缩放到零均值和单位方差）。

### 何时考虑使用支持向量 (vector)机

支持向量机是以下情况的不错选择：

- **分类任务**，当需要类别之间有明确的分离边距时。
- 具有**高维数据**的问题，例如文本分类或生物信息学。
- 数据可能非线性可分离，且**非线性核函数**有益的情况。
- 当您需要一个能提供良好**泛化性能**的模型时，前提是经过适当调优。

虽然支持向量机可能并非总是最快的训练算法，但它们查找复杂决策边界的能力以及坚实的理论支撑使它们成为机器学习 (machine learning)实践者工具包中的一个有价值的工具。与任何模型一样，它们的性能应使用适当的指标和验证策略进行严格评估，本章后续部分将对此进行介绍。

## 参考资料

- [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) — Corinna Cortes, Vladimir Vapnik (1995)
  Journal: Machine Learning; Publisher: Springer; Volume: 20; Pages: 273–297; DOI: [10.1007/BF00994018](https://doi.org/10.1007/BF00994018)
  这篇论文介绍了支持向量机 (SVM) 算法，奠定了其理论基础。
- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  一本全面的统计学习方法教科书，其中详细介绍了支持向量机和核方法。
- [LIBSVM: A Library for Support Vector Machines](https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf) — Chih-Chung Chang, Chih-Jen Lin (2011)
  Journal: ACM Transactions on Intelligent Systems and Technology; Publisher: ACM; Volume: 2; Pages: 1-27 (Article 27); DOI: [10.1145/1961189.1961199](https://doi.org/10.1145/1961189.1961199)
  描述了广泛使用的 LIBSVM 软件包，它通过 MLJLIBSVMInterface 为 Julia 中的 SVM 提供了底层实现。
- [MLJ.jl: Unified Machine Learning in Julia](https://mlj.sciml.ai/stable/) — Anthony F. Blaom and the MLJ Developers (2024)
  MLJ.jl 是 Julia 的机器学习框架的官方文档，提供了模型接口和使用细节，包括支持向量机。
