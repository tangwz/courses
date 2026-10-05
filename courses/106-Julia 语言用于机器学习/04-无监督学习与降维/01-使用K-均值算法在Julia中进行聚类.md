# 使用K-均值算法在Julia中进行聚类

来源：[原文](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality/julia-kmeans-clustering)

[返回章节目录](README.md) · [返回课程目录](../README.md)

K-均值聚类是一种被广泛使用的算法，用于将数据集划分为预设数量的不同且不重叠的子群或簇。它是一种迭代算法，旨在通过最小化每个数据点到其所属簇中心（或算术平均值）的平方距离之和来找出局部最优解。这种方法对于识别无标签数据中的潜在群组结构很有效，这是无监督学习 (supervised learning) (unsupervised learning)中的一项常见任务。

### K-均值算法说明

K-均值算法通过一个直接的迭代过程，将每个数据点分配到 $k$ 个簇中的一个。主要思想是不断调整簇的中心和点的分配，直到达到稳定的状态。

主要步骤如下：

1. **初始化**：选择 $k$ 个初始簇中心。一个常见方法是从数据集中随机选择 $k$ 个数据点作为初始簇中心。像“k-means++”这样更高级的方法旨在实现更合理的初始位置。
2. **分配步骤**：将数据集中的每个数据点分配到最近的簇中心。“接近程度”通常使用欧几里得距离来衡量。如果 $c_j$ 是簇 $j$ 的中心，且 $x_i$ 是一个数据点，当距离 $d(x_i, c_j)$ 最小化时，$x_i$ 被分配到簇 $S_j$。
3. **更新步骤**：重新计算每个簇的中心。一个簇的新中心是当前分配给该簇的所有数据点的平均值。对于簇 $S_j$，新中心 $c_j$ 计算如下：
   $c_j = \frac{1}{|S_j|} \sum_{x_i \in S_j} x_i$
   其中 $|S_j|$ 是簇 $S_j$ 中的点数量。
4. **收敛检查**：重复分配和更新步骤，直到簇分配不再发生显著变化，或簇中心的位置稳定，或达到最大迭代次数。

K-均值算法的迭代优化过程如下所示：

> K-均值算法迭代地将数据点分配给簇并更新簇中心，直到收敛。

K-均值算法试图最小化的目标函数是簇内平方和（WCSS），也称为惯性：
$WCSS = \sum_{j=1}^{k} \sum_{x_i \in S_j} ||x_i - c_j||^2$
其中 $k$ 是簇的数量，$S_j$ 是簇 $j$ 中的点集，$c_j$ 是簇 $j$ 的中心。

### 使用 Clustering.jl 在 Julia 中实现 K-均值

在Julia中，`Clustering.jl` 包提供了K-均值及其他聚类算法的高效实现。如果您尚未安装，可以使用Julia的包管理器添加它：

```julia
using Pkg
Pkg.add("Clustering")
Pkg.add("Plots") # 用于可视化
Pkg.add("Random") # 用于生成样本数据
```

我们来看一个基本例子。首先，我们将生成一些具有自然分组的合成二维数据。

```julia
using Clustering, Plots, Random

# 设置随机种子以复现结果
Random.seed!(1234)

# 生成包含三个不同群组的合成数据
# 群组 1
X1 = randn(2, 50) .* 0.5 .+ [2.0; 2.0]
# 群组 2
X2 = randn(2, 50) .* 0.5 .+ [4.0; 4.0]
# 群组 3
X3 = randn(2, 50) .* 0.5 .+ [3.0; 0.0]

# 将群组合并为单个数据集
# Clustering.jl 要求数据列为特征，行为观测值
X_combined = hcat(X1, X2, X3)
data_points = permutedims(X_combined) # 转置，使观测值为行

# 执行 K-均值聚类
# 我们指定 k=3，因为我们知道合成数据有三个群组
k = 3
result = kmeans(data_points', k; display=:iter) # data_points' 表示特征作为列

# 访问结果
assignments = result.assignments  # 每个点的簇分配
centroids = result.centers       # 簇中心的坐标 (特征为行)
total_cost = result.totalcost    # 最终聚类的 WCSS
iterations = result.iterations   # 执行的迭代次数
converged = result.converged     # 指示算法是否收敛的布尔值

println("点数量: ", size(data_points, 1))
println("簇分配 (前 10 个): ", assignments[1:10])
println("簇中心:\n", permutedims(centroids)) # 显示簇中心，特征作为列
println("总 WCSS: ", total_cost)
println("迭代次数: ", iterations)
println("已收敛: ", converged)
```

在 `kmeans` 函数中，`data_points'` 意味着我们提供的数据以特征为列、观测值为行，这是某些Julia机器学习 (machine learning)包的常见约定。`display=:iter` 选项会显示算法的进度。`result` 对象是一个 `KmeansResult` 结构体，包含有关聚类结果的详细信息。

### 确定簇的数量 ($k$)

K-均值算法的主要难点之一是您需要预先指定簇的数量 $k$。在许多实际情况下，最优的 $k$ 是未知的。有几种方法可以辅助做出此决定：

1. **肘部法则**：此方法涉及对一系列 $k$ 值运行K-均值算法，并计算每个值的WCSS。随着 $k$ 的增加，WCSS将减小，因为点平均而言会更接近其各自的簇中心。当WCSS与 $k$ 绘制成图时，图表常显示一个“肘部”点，在该点之后，增加更多簇在减小WCSS方面提供的回报会递减。此肘部点被认为是 $k$ 的一个良好选择。

我们来使用合成数据计算不同 $k$ 值的WCSS：

```julia
# data_points' 表示特征为列，观测值为行
data_for_kmeans = data_points' # 来自上一个例子

max_k = 10
wcss_values = Float64[]

for k_val in 1:max_k
    r = kmeans(data_for_kmeans, k_val)
    push!(wcss_values, r.totalcost)
end

# 下面的 Plotly 图表将此可视化。
# 您也可以使用 Plots.jl 快速绘图：
# plot(1:max_k, wcss_values, xlabel="簇的数量 (k)", ylabel="WCSS", marker=:o, legend=false)
```



[交互图表：确定最佳 k 的肘部法则](https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality/julia-kmeans-clustering#plot-18179qn)

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
        5,
        6,
        7,
        8,
        9,
        10
      ],
      "y": [
        275.3,
        102.1,
        38.7,
        32.5,
        27.1,
        23.8,
        20.5,
        18.2,
        16.3,
        14.9
      ],
      "type": "scatter",
      "mode": "lines+markers",
      "name": "WCSS",
      "marker": {
        "color": "#228be6"
      },
      "line": {
        "color": "#228be6"
      }
    }
  ],
  "layout": {
    "title": {
      "text": "确定最佳 k 的肘部法则"
    },
    "xaxis": {
      "title": {
        "text": "簇的数量 (k)"
      }
    },
    "yaxis": {
      "title": {
        "text": "簇内平方和 (WCSS)"
      }
    },
    "height": 400,
    "autosize": true
  }
}
```

</details>



> WCSS 通常随 $k$ 的增加而减小。“肘部”点（在此示例中约为 $k=3$）表明了合适的簇数量，在该点之后，增加更多簇会带来递减的回报。

2. **轮廓分析**：此方法测量一个数据点与其自身簇的相似程度，并与与其他簇的相似程度进行比较。轮廓得分的范围是 -1 到 1。较高的值表示簇之间分离良好。我们将在本章稍后更细致地介绍像轮廓得分这样的评价指标。
3. **领域知识**：通常，关于数据或问题背景的先验知识可以表明一个自然的簇数量。

### 初始化及其他注意事项

- **簇中心初始化**：K-均值对簇中心的初始位置敏感。糟糕的初始化可能导致算法收敛到WCSS的次优局部最小值。为了缓解这种情况，`Clustering.jl` 的 `kmeans` 函数默认使用“k-means++”初始化方法。这种方法通常比纯随机初始化带来更好、更一致的结果。您可以控制初始化的部分（例如，`init=:kmcen` 用于k-均值中心选择或提供您自己的初始簇中心）。
- **多次运行**：对于重要应用，K-均值有时会使用不同的随机初始化运行多次，并选择具有最低 WCSS 的聚类结果。`Clustering.jl` 的默认设置（k-means++）通常使这变得不那么必要，但如果需要，可以使用 `nruns` 参数 (parameter)。
- **特征缩放**：K-均值使用欧几里得距离，它对特征的尺度敏感。如果您的特征具有不同的单位或取值范围差异很大（例如，以年为单位的年龄和以千美元为单位的收入），那么在应用 K-均值之前，通常最好对数据进行缩放（例如，使用第二章中的标准化或最小-最大缩放）。这保证了所有特征对距离计算的贡献更加均衡。

### 可视化 K-均值结果

对簇进行可视化可以提供重要的了解，特别是对于二维或三维数据。我们来使用 `Plots.jl` 绘制之前示例中的 3 簇结果。

```julia
# 使用 k=3 示例的结果：
# data_points (观测值为行，特征为列)
# assignments (每个点的簇 ID)
# centroids (特征为行，簇为列)

# 准备绘图数据
x_coords = data_points[:, 1]
y_coords = data_points[:, 2]

# 簇中心 (如果特征是行，绘图时需要转置)
# result.centers 是特征为行，簇为列。
# 因此，centroids_plot 是簇为行，特征（坐标）为列。
centroids_plot = permutedims(result.centers)

# 创建散点图
p = scatter(x_coords, y_coords, group=assignments,
            xlabel="特征 1", ylabel="特征 2",
            title="K-均值聚类 (k=3)",
            legend=:outertopright, palette=:viridis) # :viridis 只是一个选项

# 将簇中心添加到图中
scatter!(p, centroids_plot[:, 1], centroids_plot[:, 2],
         markershape=:xcross, markersize=8, markercolor=:red,
         label="簇中心", seriesalpha=1)

# 在典型 Julia 环境中显示图表：
# display(p)
# 如果使用自动显示图表的笔记本或环境，这可能不需要。
```

> 二维数据上的 K-均值聚类示例。点按其所属簇着色，簇中心用红色叉号标记 (token)。`Plots.jl` 库与 GR 或 PlotlyJS 等后端配合使用可以生成此类可视化。

### K-均值的优点与局限

K-均值受欢迎是有充分理由的，但需要了解它的特点：

**优点：**

- **简单性与速度**：它相对容易理解和实现。它在计算上也很高效，特别是对于大型数据集，其时间复杂度通常接近数据点数量的线性关系。
- **可伸缩性**：对大型数据集有良好的伸缩性，特别是经过优化的实现。
- **易于理解**：由簇中心定义的聚类结果通常易于理解。

**局限性：**

- **需要指定 $k$**：簇的数量 $k$ 必须预先选择，如果不是从领域专业知识中得知，这会有些难度。
- **对初始化敏感**：尽管 k-means++ 有所帮助，但算法仍然可能根据初始簇中心的位置收敛到局部最优解。
- **假定簇为球形**：K-均值在簇大致呈球形、大小相等且密度相似时表现最佳。它可能在形状不规则、大小不一或密度不同的簇上表现不佳。
- **异常值的影响**：异常值会明显扭曲簇中心的位置并使所得簇变形。
- **“硬”分配**：每个点都被明确地分配给单个簇。对于位于簇边界之间的点，或者对于具有重叠结构的数据集（其中概率分配（如高斯混合模型中）可能更适合），这可能不是最佳选择。

尽管存在局限性，但 K-均值因其简单性和效率，通常是聚类任务的一个良好起点。了解其行为和假设对于有效地应用它并解读其结果是必要的。本章稍后，我们将研究 DBSCAN，这是一种基于密度的聚类算法，它能应对其中一些局限，例如找出任意形状的簇且无需预先指定 $k$。

## 参考资料

- [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://hastie.su.domains/ElemStatLearn/) — Trevor Hastie, Robert Tibshirani, Jerome Friedman (2009)
  Publisher: Springer; Pages: 511-517
  这是统计机器学习的经典参考书，全面阐述了 K-Means 算法、其目标函数及其应用注意事项。
- [k-means++: The advantages of careful seeding](https://doi.org/10.1145/1283383.1283494) — David Arthur, Sergei Vassilvitskii (2007)
  Journal: Proceedings of the Eighteenth Annual ACM-SIAM Symposium on Discrete Algorithms (SODA); Publisher: Society for Industrial and Applied Mathematics; Pages: 1027–1035; DOI: [10.1145/1283383.1283494](https://doi.org/10.1145/1283383.1283494)
  提出了 k-means++ 初始化算法，通过更具策略性地选择初始中心点，显著提升了 K-Means 聚类的质量。
- [Clustering.jl Documentation](https://juliastats.org/Clustering.jl/stable/) — JuliaData (2025)
  `Clustering.jl` 包的官方文档，提供了其 K-Means 实现、API 以及在 Julia 中进行聚类的各种选项的详细信息。

---

[上一节](../03-%E5%AE%9E%E7%8E%B0%E7%9B%91%E7%9D%A3%E5%AD%A6%E4%B9%A0%E7%AE%97%E6%B3%95/07-%E7%9B%91%E7%9D%A3%E6%A8%A1%E5%9E%8B%E7%9A%84%E8%AE%AD%E7%BB%83%E4%B8%8E%E8%AF%84%E4%BC%B0%EF%BC%9A%E5%8A%A8%E6%89%8B%E5%AE%9E%E8%B7%B5.md) · [下一节](02-%E5%9F%BA%E4%BA%8E%E5%AF%86%E5%BA%A6%E7%9A%84%E8%81%9A%E7%B1%BB%EF%BC%9ADBSCAN.md)
