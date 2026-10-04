---
course: "julia-for-machine-learning"
chapter: "julia-unsupervised-learning-dimensionality"
lesson: "julia-dbscan-clustering"
sourceId: 6876
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality/julia-dbscan-clustering"
title: "基于密度的聚类：DBSCAN"
description: "使用DBSCAN基于密度对数据点进行聚类，发现不同形状的簇。"
order: 2
plots: ["plots/6876-0.json"]
sourceHash: "e77cd640b9d5921c79573095e672ac6798cf5373f30229cf632c8fc0618b81b0"
sourceCorrections: []
---

K-Means聚类在已知簇数量$k$时，对于识别球形簇很有效，但许多数据集中有任意形状的簇，并可能包含不属于任何特定群体的异常点。基于密度的带噪声应用空间聚类（DBSCAN）提供了一种在这些情况下表现很好的替代方法。它将紧密排列的点归为一类，将孤立于低密度区域的点标记 (token)为异常点。DBSCAN的一个重要优点是，它不需要你事先指定簇的数量。

### DBSCAN的核心思想

DBSCAN的逻辑围绕着两个主要参数 (parameter)展开：

1. **Epsilon ($\epsilon$ 或 `eps`)**: 这个参数定义了数据点周围邻域的半径。如果你想象在一个点周围画一个半径为$\epsilon$的圆，所有落在这个圆内的其他点都被视为它的邻居。
2. **最小点数（`min_pts`）**: 这是在一个点的$\epsilon$-邻域内必须存在的最小数据点数量（包括点本身），该点才会被视为“核心点”。

根据这些参数，DBSCAN将数据集中的每个点分为三种类型之一：

- **核心点**: 如果一个点的$\epsilon$-邻域包含至少`min_pts`个点，则该点是核心点。核心点通常位于簇的密集内部。
- **边界点**: 如果一个点本身不是核心点（即，它的$\epsilon$-邻域中少于`min_pts`个点），但落在某个核心点的$\epsilon$-邻域内，则该点是边界点。边界点位于簇的边缘。
- **噪声点（异常点）**: 如果一个点既不是核心点也不是边界点，则该点是噪声点。这些点孤立在低密度区域中。

簇是通过连接彼此靠近的核心点形成的。具体来说：

- 如果点$q$在核心点$p$的$\epsilon$-邻域内，则称$q$从$p$*直接密度可达*。
- 如果存在一个点链$p_1, p_2, \ldots, p_n$，使得$p_1 = p$， $p_n = q$，并且每个$p_{i+1}$都从$p_i$直接密度可达（其中链中的所有点，除了$q$以外，都必须是核心点），则点$q$从点$p$*密度可达*。
- 如果存在一个核心点$o$，从该核心点$o$，$p$和$q$都密度可达，则点$p$和点$q$*密度相连*。

DBSCAN中的一个簇是密度相连点的最大集合。算法遍历数据点，当遇到一个未被访问的核心点时，它从该点开始，通过寻找所有密度可达点来扩充一个新簇。被归类为噪声的点不属于任何簇。

### 在Julia中实现DBSCAN

在Julia中，`Clustering.jl`包提供了DBSCAN的高效实现。我们来看看如何使用它。首先，请确保你已安装该包：

```julia
using Pkg
Pkg.add("Clustering")
Pkg.add("Plots") # 用于可视化
```

现在，让我们生成一些形成非球形簇的样本数据，这正是DBSCAN的优势所在，并应用该算法。

```julia
using Clustering
using Plots
using Random # 用于可重现示例

# 生成样本数据：两轮新月
Random.seed!(123) # 用于可重现性
n_points_per_moon = 150
noise_level = 0.05

# 第一轮新月
t1 = range(0, pi, length=n_points_per_moon)
x1 = cos.(t1) .+ Random.randn(n_points_per_moon) .* noise_level
y1 = sin.(t1) .+ Random.randn(n_points_per_moon) .* noise_level

# 第二轮新月（偏移并翻转）
t2 = range(0, pi, length=n_points_per_moon)
x2 = 1 .- cos.(t2) .+ Random.randn(n_points_per_moon) .* noise_level
y2 = 0.5 .- sin.(t2) .- Random.randn(n_points_per_moon) .* noise_level

# 合并成一个数据集（特征作为列）
data = vcat(hcat(x1, y1)', hcat(x2, y2)')' # 转置以使特征作为列

# 添加一些随机噪声点
n_noise = 20
noise_x = (rand(n_noise) .* 3 .- 0.5)
noise_y = (rand(n_noise) .* 2 .- 0.5)
noise_data = hcat(noise_x, noise_y)'

full_data = hcat(data, noise_data) # 数据矩阵：2行（特征），N列（样本）

# 应用DBSCAN
# 参数：eps（半径），min_pts（邻域中的最小点数）
# 数据矩阵应该是 d x n，其中 d 是维度，n 是点数。
# 我们的 full_data 当前是 n x d（点作为行，特征作为列），
# 所以我们为 Clustering.jl 的 dbscan 转置它。
eps_val = 0.25
min_pts_val = 5

result = dbscan(full_data', eps_val, min_neighbors=min_pts_val)

# `result.assignments` 包含每个点的簇分配。
# 0 通常表示噪声点。
# `result.counts` 给出每个簇中的点数。
# `result.is_core` 是一个布尔向量，表示一个点是否是核心点。

println("簇分配: ", result.assignments)
println("簇计数: ", result.counts)

# 可视化结果
# 将簇分配映射到颜色。噪声点（分配0）获得特定颜色。
cluster_colors = [ :gray, :blue, :red, :green, :purple, :orange ] # 如果预期有更多簇，则添加更多
point_colors = map(id -> id == 0 ? cluster_colors[1] : cluster_colors[id+1], result.assignments)
point_markers = map(id -> id == 0 ? :x : :circle, result.assignments)

scatter(full_data[1, :], full_data[2, :],
        markercolor=point_colors,
        markershape=point_markers,
        legend=false,
        title="DBSCAN聚类结果 (eps=$eps_val, min_pts=$min_pts_val)",
        xlabel="特征 1", ylabel="特征 2")
```

运行这段代码将把DBSCAN应用于生成的数据集。`Clustering.jl`中的`dbscan`函数接受数据矩阵（特征为行，样本为列）、$\epsilon$和`min_neighbors`（对应于`min_pts`）作为主要参数 (parameter)。输出的`result`对象包含`assignments`（一个数组，其中每个元素是相应数据点的簇ID，0通常表示噪声）、`counts`（每个簇中的点数）和`is_core`（一个布尔数组）。

### 选择`eps`和`min_pts`

DBSCAN的性能对`eps`和`min_pts`的选择相当敏感。

- **`min_pts`**: `min_pts`的一个通用指南是根据数据的维度($D$)来设置。一个常见的起始点是`min_pts >= D + 1`。对于二维数据，`min_pts = 4`或`min_pts = 5`通常是一个不错的选择。较大的`min_pts`值倾向于产生更重要的簇并将更多点归类为噪声，使算法对异常点更具抵抗力。如果`min_pts`太小（例如1或2），每个点都可能成为核心点，或者稀疏的簇可能合并。
- **`eps`**: 选择`eps`通常更具挑战性。

  - **K-距离图**: 一个有用的启发式方法是k-距离图。对于选定的$k$（通常k = \text{min\_pts} - 1），计算每个点到其第$k$个最近邻居的距离。将这些距离按升序排列并绘制出来。图表通常会显示一个“膝盖”或“肘部”点。这个肘部处的距离值可以作为`eps`的一个良好选择。这个点代表着一个阈值，在该阈值处距离急剧增加，表明从密集区域到稀疏区域的过渡。

  ```julia
  # k-距离图示例（使用 k = min_pts_val - 1）
  # 这需要一种计算 k-最近邻距离的方法。
  # 我们可以为此使用 NearestNeighbors.jl。
  # Pkg.add("NearestNeighbors")
  using NearestNeighbors

  k = min_pts_val - 1
  kdtree = KDTree(full_data) # 构建KD树以进行高效的邻居搜索
  # `knn` 返回索引和距离。我们需要距离。
  # 对于每个点，找到它的 k 个最近邻居（不包括它自己）。
  # 由于 knn 会包含树中的点本身，所以我们请求 k+1 个邻居
  # 如果 k > 0，则取第 k 个距离，如果 min_pts=1，则处理 k=0 的情况。
  if k > 0
      idxs, dists = knn(kdtree, full_data, k + 1, true) # true 表示对结果进行排序
      # dists 是一个向量的向量；我们需要每个点的第 k 个距离
      k_distances = [d[k+1] for d in dists] # k+1 是因为 knn 结果是1-索引的，d[1] 是点本身
                                            # 我们需要第 k 个*其他*点，所以索引是 k+1
      sort!(k_distances)
      plot(k_distances,
          xlabel="点（按距离排序）",
          ylabel="第 $k$ 个最近邻距离",
          title="k-距离图 (k=$k)",
          legend=false)
  else
      println("对于基于 min_pts - 1 的 k-距离图，k 必须 > 0。")
  end
  ```

  然后你将在生成的k-距离图中寻找一个“肘部”来估算`eps`。

  - **领域知识**: 如果你对数据特征的尺度或簇之间的典型间隔有了解，这可以指导你选择`eps`。
  - **迭代**: 寻找最佳`eps`通常需要一些实验。尝试几个值，并评估所得簇的质量，可以凭视觉判断，或者在没有真实标签的情况下使用簇评估指标。

### DBSCAN的优缺点

**优点：**

- **处理任意形状**: 与K-Means不同，DBSCAN可以识别非球形或复杂形状的簇。
- **异常点检测**: 它内在识别并标记 (token)噪声点，使其对异常点表现良好。
- **无需预先指定簇数量**: 簇的数量由算法根据数据的密度结构来确定。

**缺点：**

- **参数 (parameter)敏感性**: DBSCAN聚类的质量在很大程度上取决于`eps`和`min_pts`参数。找到最优值可能不是一件简单的事。
- **处理不同密度时的困难**: 如果簇的密度显著不同，DBSCAN可能会遇到困难，因为单个`eps`和`min_pts`设置可能不适合所有簇。一些密集簇可能会合并，或者稀疏簇可能会被遗漏或分散。
- **维度灾难**: 在非常高维的空间中，由于点之间的距离趋于一致，密度概念的意义降低。这会使DBSCAN难以有效区分密集区域。

### DBSCAN结果的可视化

可视化输出是理解DBSCAN如何划分数据的好方法。对于二维或三维数据，散点图非常有效。



![DBSCAN聚类示例：两轮新月](plots/6876-0.json)



> DBSCAN识别出两个月牙形簇并将一些点标记 (token)为噪声的示例。

DBSCAN提供了一种强大的、基于密度的聚类方法，在处理复杂数据结构和存在噪声时特别有用。它能够在不预先指定簇数量的情况下发现不同形状的簇，使其成为无监督学习 (supervised learning) (unsupervised learning)工具箱中的一个有价值的工具。但是，仔细考虑其参数 (parameter)对于获得有意义的结果很重要。

## 参考资料

- [A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise](http://www.aaai.org/Library/KDD/1996/kdd96-037.php) — Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu (1996)
  Journal: Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD-96); Publisher: AAAI Press; Pages: 226-231; DOI: [10.1145/233255.233267](https://doi.org/10.1145/233255.233267)
  介绍了DBSCAN算法的原始论文，定义了密度可达性和密度连接等核心概念。
- [DBSCAN](https://juliaclustering.github.io/Clustering.jl/dev/dbscan/) — Clustering.jl Developers (2024)
  Publisher: JuliaStats
  Julia `Clustering.jl`包中DBSCAN实现的官方文档，对实际应用有指导价值。
