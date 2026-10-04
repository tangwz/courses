---
course: "julia-for-machine-learning"
chapter: "julia-unsupervised-learning-dimensionality"
lesson: "julia-clustering-pca-practice"
sourceId: 6884
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality/julia-clustering-pca-practice"
title: "动手实践：应用聚类与PCA"
description: "在Julia中使用数据集进行聚类与降维，并评估结果。"
order: 6
plots: ["plots/6884-0.json", "plots/6884-1.json"]
sourceHash: "6bdf681527710c2761bd5569a6e4a47594896ee34184b804738b589ad1f85e7b"
sourceCorrections: []
---

使用Julia应用聚类算法和主成分分析（PCA）。涉及处理一个常用数据集，进行K-Means聚类，使用PCA降低数据维度，并评估聚类结果的质量。这种实践操作将巩固你对这些无监督学习 (supervised learning) (unsupervised learning)技术在典型机器学习 (machine learning)流程中如何实现和运用的理解。

首先，确保你已安装所需的Julia包。如果没有，可以使用Julia的包管理器添加它们：

```julia
using Pkg
Pkg.add(["MLJ", "MLJModels", "DataFrames", "RDatasets", "Plots", "Clustering", "MultivariateStats", "Random", "Printf"])
```

安装完成后，我们可以将它们加载到Julia会话中。

```julia
using MLJ
using DataFrames
using RDatasets
using Plots
using Clustering # 用于轮廓系数
using Distances  # 用于轮廓系数的距离矩阵
using MultivariateStats # 如果不使用MLJ的封装，则直接用于PCA
using Random
using Printf

# 设置随机种子以保证结果可复现性
Random.seed!(1234)

# 从MLJModels加载特定模型
KMeans = @load KMeans pkg=Clustering verbosity=0
PCA = @load PCA pkg=MultivariateStats verbosity=0
```

我们使用`Random.seed!(1234)`来确保涉及随机性的操作（例如K-Means初始化）在每次运行代码时都能产生相同的结果。`MLJ`的`@load`宏用于导入模型，而`verbosity=0`则在加载时抑制输出。

### 加载和准备鸢尾花数据集

我们将使用经典的鸢尾花数据集，该数据集可通过`RDatasets.jl`轻松获取。该数据集包含150朵鸢尾花的测量数据，每朵花属于三个物种之一。对于我们的无监督学习 (supervised learning) (unsupervised learning)任务，我们将只使用特征测量值，忽略物种标签，假装我们不知道它们。

```julia
# 加载鸢尾花数据集
iris = RDatasets.dataset("datasets", "iris")

# 从数据集中分离特征(X)。对于无监督任务，我们将忽略物种(Species)列。
X = select(iris, Not(:Species))

# 显示特征的前几行和摘要
println("First 5 rows of features:")
show(stdout, "text/plain", first(X, 5))
println("\n\nSummary of features:")
show(stdout, "text/plain", describe(X))
```

输出将显示四个特征：`SepalLength`（萼片长度）、`SepalWidth`（萼片宽度）、`PetalLength`（花瓣长度）和`PetalWidth`（花瓣宽度）。这些是我们将用于聚类和降维的属性。

### 对原始特征应用K-Means聚类

K-Means是一种算法，旨在将$n$个观测值划分成$k$个簇，其中每个观测值都属于离其最近的均值（聚类中心）所在的簇。我们对4维鸢尾花特征数据应用K-Means。我们将选择$k=3$个簇，因为我们（外部）知晓鸢尾花有三种物种。

```julia
# 实例化K-Means模型，指定3个簇
kmeans_model = KMeans(k=3)

# 创建一个将模型与数据绑定的MLJ机器
mach_kmeans = machine(kmeans_model, X)

# 拟合K-Means模型
fit!(mach_kmeans)

# 获取每个数据点的簇分配
assignments = predict(mach_kmeans, X)
# MLJ的predict函数返回CategoricalArray。在某些情况下，我们可能需要整数标签。
assignments_int = MLJ.levelcode.(assignments)

# 获取聚类中心
centroids = report(mach_kmeans).centroids
println("\n\n聚类中心 (4D):")
show(stdout, "text/plain", centroids)
```

`report(mach_kmeans).centroids`为我们提供了原始4维空间中三个聚类中心的坐标。`centroids`中的每一行都代表一个中心。

直接在4D中可视化聚类是不可行的。然而，我们可以创建两个特征（例如`PetalLength`和`PetalWidth`）的散点图，并根据其分配的簇给点着色。

```julia
# 使用两个特征可视化聚类：花瓣长度 vs 花瓣宽度
scatter(X.PetalLength, X.PetalWidth, group=assignments,
        xlabel="花瓣长度", ylabel="花瓣宽度",
        title="鸢尾花上的K-Means聚类 (花瓣特征)",
        legend=:topleft, palette=[:blue, :green, :orange]) # 使用调色板以获得不同的颜色
```

> 一个散点图，显示根据花瓣长度和花瓣宽度绘制的鸢尾花数据点，并根据其K-Means分配的簇进行着色。

此图提供了聚类的部分视图。相同颜色的点被K-Means分组在一起。

### 使用主成分分析（PCA）进行降维

我们的鸢尾花数据集有四个特征。PCA可以帮助我们将其降至较低的维度数量（例如两个），同时保留数据中的大部分方差。这对于可视化很有帮助，有时还能提升后续机器学习 (machine learning)算法的性能。

```julia
# 实例化PCA模型，降至2个维度
pca_model = PCA(maxoutdim=2) # 或者 pratio=0.95 以保留95%的方差

# 为PCA创建一个MLJ机器
mach_pca = machine(pca_model, X)

# 拟合PCA模型
fit!(mach_pca)

# 将原始数据转换为主成分，得到2个主成分
X_pca = transform(mach_pca, X) # X_pca 将是一个表格

# 显示PCA转换后的数据的前几行
println("\n\nPCA转换后的数据（2D）的前5行:")
show(stdout, "text/plain", first(X_pca, 5))

# 报告中包含如解释方差等信息
pca_report = report(mach_pca)
@printf("\n2个成分解释的累计方差: %.2f%%\n", pca_report.cumulative_variance[2] * 100)
```

`transform`函数将原始4D数据投影到前两个主成分上。PCA机器的报告告诉我们这两个成分捕获了原始数据多少方差。通常，高百分比（例如，>80-90%）表明降维效果良好。

现在，我们来可视化由PCA得到的这个新的2D空间中的数据。

```julia
# 如果需要，将X_pca表格转换为矩阵，以便使用Plots.jl更容易绘图
# X_pca 是一个表格，所以通过名称访问列（例如：:x1, :x2, ...）
# 名称通常是 x1, x2, ... 这样的形式
col_names = Tables.columnnames(X_pca)

scatter(Tables.getcolumn(X_pca, col_names[1]), Tables.getcolumn(X_pca, col_names[2]),
        xlabel="主成分 1", ylabel="主成分 2",
        title="PCA后的鸢尾花数据 (2D)", legend=false,
        color=:gray, markersize=4)
```



![投影到2个主成分上的鸢尾花数据](plots/6884-0.json)



> 经过PCA转换到二维空间后的鸢尾花数据集数据点。每个点代表一朵鸢尾花。

### 对PCA转换后的数据进行聚类

在数据降至二维后，我们现在可以再次应用K-Means聚类。这是常见做法：先降维，再聚类。

```julia
# 对2D PCA转换后的数据应用K-Means (k=3)
mach_kmeans_pca = machine(kmeans_model, X_pca) # kmeans_model 已是KMeans(k=3)
fit!(mach_kmeans_pca)

# 获取PCA数据上的簇分配
assignments_pca = predict(mach_kmeans_pca, X_pca)
assignments_pca_int = MLJ.levelcode.(assignments_pca)

# 获取2D PCA空间中的中心点
centroids_pca = report(mach_kmeans_pca).centroids
println("\n\n聚类中心 (2D PCA空间):")
show(stdout, "text/plain", centroids_pca)

# 在2D PCA图上可视化聚类
scatter(Tables.getcolumn(X_pca, col_names[1]), Tables.getcolumn(X_pca, col_names[2]), group=assignments_pca,
        xlabel="主成分 1", ylabel="主成分 2",
        title="PCA降维后的鸢尾花数据的K-Means聚类",
        legend=:topleft, palette=[:blue, :green, :orange]) # 如果进行比较，则匹配调色板
```



![PCA转换后的鸢尾花数据上的K-Means聚类](plots/6884-1.json)



> PCA转换后的鸢尾花数据点，根据应用于此2D数据的K-Means聚类分配进行着色。不同的颜色代表不同的簇。

如果簇与主成分对齐 (alignment)良好，此图应显示更清晰的视觉分离。

### 评估聚类性能

我们的聚类效果如何？对于无监督学习 (supervised learning) (unsupervised learning)，评估可能有些难度，因为我们没有真实标签（尽管对于鸢尾花数据集，我们私下里知道）。内部评估指标评估聚类结构本身的质量。轮廓系数是一个常用指标。它衡量一个对象与它所属的簇（内聚性）的相似程度，并与它到其他簇（分离性）的相似程度进行比较。分数范围从-1到1，高值表示对象与它所属的簇匹配良好，而与相邻簇匹配不佳。

我们将使用`Clustering.jl`中的`silhouettes`函数。它需要数据为矩阵形式，整数簇分配以及一个距离矩阵。

```julia
# 将特征转换为矩阵格式以计算距离
X_matrix = MLJ.matrix(X)               # 原始4D数据
X_pca_matrix = MLJ.matrix(X_pca)       # PCA降维后的2D数据

# 计算原始4D数据上K-Means的轮廓系数
# assignments_int 来自原始X上的K-Means
dist_matrix_4d = pairwise(Euclidean(), X_matrix', dims=2)
sils_4d = silhouettes(assignments_int, dist_matrix_4d)
mean_silhouette_4d = mean(sils_4d)
@printf("\n平均轮廓系数 (4D数据上的K-Means): %.3f\n", mean_silhouette_4d)

# 计算PCA降维后的2D数据上K-Means的轮廓系数
# assignments_pca_int 来自X_pca上的K-Means
dist_matrix_2d = pairwise(Euclidean(), X_pca_matrix', dims=2)
sils_2d = silhouettes(assignments_pca_int, dist_matrix_2d)
mean_silhouette_2d = mean(sils_2d)
@printf("平均轮廓系数 (2D PCA数据上的K-Means): %.3f\n", mean_silhouette_2d)
```

比较`mean_silhouette_4d`和`mean_silhouette_2d`可以提供一些认识。有时，对PCA降维后的数据进行聚类会产生更好（或更稳定）的轮廓系数，特别是当移除的成分大部分是噪声时。在其他情况下，PCA造成的信息损失可能会降低聚类质量。

对于鸢尾花数据集，对PCA降维后的数据进行聚类通常会得到不错的轮廓系数，因为前两个主成分很好地捕获了物种分离。

### 了解其他算法：关于DBSCAN的说明

尽管本次实践主要集中于K-Means，但Julia的生态系统，特别是通过`MLJ`和`Clustering.jl`，支持其他无监督算法。例如，DBSCAN（基于密度的含噪声应用空间聚类）非常适合寻找非球形簇和识别噪声点。你可以类似地加载和使用它：

```julia
# DBSCANModel = @load DBSCAN pkg=Clustering verbosity=0
# dbscan_model = DBSCANModel(eps=0.5, min_neighbors=5) # 参数 eps 和 min_points 需要调整
# mach_dbscan = machine(dbscan_model, X_pca) # 或 X
# fit!(mach_dbscan)
# assignments_dbscan = predict(mach_dbscan, X_pca)
# ... 然后可视化和评估。
```

在鸢尾花数据集上（特别是PCA降维版本）研究DBSCAN的不同参数 (parameter)（`eps`和`min_neighbors`）会是一个富有启发性的后续练习。

### 实践环节总结

在此动手环节中，你成功地：

1. 加载并准备了用于无监督学习 (supervised learning) (unsupervised learning)的鸢尾花数据集。
2. 对原始4维特征集应用了K-Means聚类。
3. 使用主成分分析（PCA）将数据集维度降至两个成分。
4. 在PCA降维后的2D空间中可视化了数据。
5. 对PCA转换后的数据应用K-Means聚类并可视化了这些簇。
6. 使用轮廓系数评估了两种聚类方法的性能。

这些步骤代表了探索性数据分析和无监督机器学习 (machine learning)中的常见工作流程。通过在Julia中完成这些实例操作，你获得了使用其强大工具来执行这些任务的实践经验，为解决更复杂的问题奠定了基础。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://link.springer.com/book/10.1007/978-0-387-84858-7) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; DOI: [10.1007/978-0-387-84858-7](https://doi.org/10.1007/978-0-387-84858-7)
  为主成分分析（PCA）和K-Means聚类提供了全面的理论基础，对于理解无监督学习概念具有价值。
- [MLJ.jl Documentation](https://juliaai.github.io/MLJ.jl/dev/) — MLJ.jl Contributors (2025)
  Julia机器学习框架的官方文档，对于理解其API和模型加载过程至关重要。
- [Clustering.jl Documentation](https://juliastats.org/Clustering.jl/stable/) — JuliaML Contributors (2025)
  提供了K-Means和DBSCAN等聚类算法的详细用法和解释，包括轮廓系数功能。
- [Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis](https://www.sciencedirect.com/science/article/pii/0377042787901257) — Peter J. Rousseeuw (1987)
  Journal: Journal of Computational and Applied Mathematics; Publisher: Elsevier; Volume: 20; Pages: 53-65; DOI: [10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427%2887%2990125-7)
  这篇开创性论文引入了轮廓系数，这是一个广泛用于聚类评估的指标。
- [Hands-On Julia for Data Science: Explore the power of Julia 1.x to perform data science and machine learning tasks](https://www.packtpub.com/product/hands-on-julia-for-data-science/9781801815124) — Rezaul Karim (2022)
  Publisher: Packt Publishing
  一本使用Julia实现机器学习技术（包括无监督方法）的实用指南。
