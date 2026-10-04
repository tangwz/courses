---
course: "julia-for-machine-learning"
chapter: "julia-unsupervised-learning-dimensionality"
lesson: "julia-evaluating-clustering"
sourceId: 6882
sourceUrl: "https://apxml.com/zh/courses/julia-for-machine-learning/chapter-4-julia-unsupervised-learning-dimensionality/julia-evaluating-clustering"
title: "评估聚类效果"
description: "学习评估聚类结果质量的方法和指标（例如，轮廓分数、戴维斯-布丁指数）。"
order: 5
plots: ["plots/6882-0.json"]
sourceHash: "fc4a2ae5c95be58413e63fb7b5c1b21492f62b8f7d0d3f9be596cbd1b9ec6c34"
sourceCorrections: []
---

当数据已通过 K-Means 或 DBSCAN 等算法聚类后，评估这些分组的质量便成为必不可少的步骤。与提供真实标签进行比较的监督学习 (supervised learning)不同，无监督聚类通常缺少此类直接参照。因此，评估聚类表现需要一套不同的工具和角度。目的是根据数据的某些内在属性或（如果存在）外部参照来衡量所得聚类的质量。

评估聚类质量通常有两类度量方法：内部和外部。

### 内部评估方法

内部评估方法不依赖任何外部信息来评估聚类结构的好坏。它们仅使用数据集中本身的信息，关注诸如簇的紧凑性（同一簇中点之间的接近程度）和簇的分离度（不同簇之间的区分度）等特征。

一个好的聚类算法应生成满足以下条件的簇：

- 同一簇内的点彼此相似（高簇内相似度）。
- 不同簇的点彼此不相似（低簇间相似度）。

我们来看一些常用的内部评估指标。

#### 轮廓系数（或轮廓分数）

轮廓系数是一种常用的指标，它量化 (quantization)你的簇的清晰度。对于每个数据点 $i$，其轮廓值 $s(i)$ 计算如下：


$$
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
$$


其中：

- $a(i)$: 点 $i$ 到同一簇内所有其他点的平均距离。这衡量了内聚性。
- $b(i)$: 点 $i$ 到最近邻簇（不包含 $i$ 且与 $i$ 平均距离最小的簇）中所有点的平均距离。这衡量了分离度。

轮廓系数范围从 -1 到 +1：

- **+1**: 表示该点远离相邻簇，并且非常靠近其自身簇中的点。这是理想情况。
- **0**: 表示该点位于或非常接近两个相邻簇之间的决策边界。
- **-1**: 表示该点可能被分配到了错误的簇。

所有数据点的平均轮廓系数提供了聚类质量的总体度量。平均轮廓分数越高，通常表示聚类效果越好。

在 Julia 中，您可以使用 `Clustering.jl` 包计算轮廓分数。

```julia
using Clustering, Distances, Statistics

# 假设 X 是您的数据矩阵（特征按行，观测值按列）
# 例如，X = rand(2, 100) 表示 100 个二维点
# 假设 'assignments' 是每个点的簇标签数组
# 例如，assignments = kmeans(X, 3).assignments

# 示例数据和分配
X = hcat(randn(2, 30), randn(2,40) .+ 5, randn(2,30) .- 3)
assignments = vcat(fill(1, 30), fill(2, 40), fill(3, 30))

if size(X, 2) != length(assignments)
    error("数据点与分配计数不匹配。")
end

# 计算成对距离（例如，欧几里得距离）
# 确保 X 是特征按行、观测值按列的形式，以便进行成对计算
dist_matrix = pairwise(Euclidean(), X, dims=2)

# 计算每个点的轮廓分数
sils = silhouettes(assignments, dist_matrix)

# 整个聚类的平均轮廓分数
avg_silhouette_score = mean(sils)
println("平均轮廓分数: ", avg_silhouette_score)

# 您也可以查看单个轮廓分数
# 例如，查找轮廓分数较低的点：
# low_silhouette_points_indices = findall(s -> s < 0.1, sils)
```

以下图表说明了轮廓系数与簇结构的关系：

> 该图展示了三个点。点 A 具有高轮廓值，与相邻簇分离良好，且与自身簇内聚。点 B 靠近边界。点 C 可能被分到错误的簇，因为它比其被分配到的簇更接近相邻簇。

轮廓分数的一个常见用途是帮助确定合适的簇数量 $k$。您可以对不同 $k$ 值运行聚类算法（例如 K-Means），并选择产生最高平均轮廓分数的 $k$ 值。

#### 戴维斯-布丁指数 (DBI)

戴维斯-布丁指数 (DBI) 是另一种内部评估指标。它被定义为每个簇与其最相似簇的平均相似度度量。相似度是簇内距离与簇间距离之比。DBI 值越低表示聚类效果越好，因为簇越紧凑且分离良好。

对于每个簇 $C_i$，我们计算 $R_i$：


$$
R_i = \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \right)
$$


其中：

- $\sigma_i$ 是簇 $C_i$ 中所有点到其簇中心 $c_i$ 的平均距离（衡量簇 $i$ 内部的离散程度）。
- $d(c_i, c_j)$ 是簇 $C_i$ 和簇 $C_j$ 的中心之间的距离。

戴维斯-布丁指数是所有簇的这些 $R_i$ 值的平均值：


$$
\text{DBI} = \frac{1}{k} \sum_{i=1}^{k} R_i
$$


DBI 分数越低，表示模型中簇之间的分离度越好。与轮廓分数不同，`Clustering.jl` 中没有直接用于 DBI 的函数，因此其计算可能需要自定义实现，或者如果 `MLJ.jl` 等更全面的框架中针对特定模型类型提供了相应功能，则可以采用这些功能。

#### 卡林斯基-哈拉巴斯指数（方差比准则）

卡林斯基-哈拉巴斯指数（也称为方差比准则）根据所有簇的簇间离散度之和与簇内离散度之和的比率来评估簇的有效性。卡林斯基-哈拉巴斯指数分数越高，通常表示模型中的簇定义越清晰。计算公式如下：


$$
\text{CH} = \frac{\text{SS}_B / (k-1)}{\text{SS}_W / (N-k)}
$$


其中：

- $\text{SS}_B$ 是总体簇间方差。
- $\text{SS}_W$ 是总体簇内方差。
- $k$ 是簇的数量。
- $N$ 是观测值的总数。

CH 值越大表示聚类效果越好，意味着簇更密集且分离良好。与 DBI 类似，卡林斯基-哈拉巴斯指数的直接函数可能不在基本聚类包中，但可以在更全面的机器学习 (machine learning)框架中实现或找到。

### 外部评估方法

外部评估方法在数据具有真实类别标签时使用。这种情形在对已知分类的数据集进行聚类算法基准测试时很常见，或者在使用合成数据时。这些指标将聚类结果与真实的底层结构进行比较。

#### 调整兰德指数 (ARI)

兰德指数 (RI) 通过考虑所有样本对，并计算在预测聚类和真实聚类中分配到相同或不同簇的对数，来衡量两种数据聚类之间的相似度。调整兰德指数 (ARI) 是兰德指数的一个版本，它针对随机性进行了调整。
ARI 范围从 -1 到 1：

- **1**: 真实标签与聚类结果完美一致。
- **0**: 聚类效果不比随机分配好。
- **负值**: 聚类效果比随机分配差。

在 Julia 中，`Clustering.jl` 提供了 `randindex` 来计算 RI 和 ARI。

```julia
using Clustering, Statistics

# 假设 'true_labels' 是真实类别标签
# 假设 'predicted_assignments' 是您的算法生成的簇标签
# 示例：
true_labels = vcat(fill(1, 50), fill(2, 50), fill(3,50))
# 我们来创建一些预测的分配，例如来自 K-Means
# 为了说明，一些略有缺陷的分配：
predicted_assignments = vcat(fill(1, 45), fill(2,5), fill(2, 48), fill(3,2), fill(3,40), fill(1,10))
# 确保长度匹配
if length(true_labels) != length(predicted_assignments)
   idx = min(length(true_labels), length(predicted_assignments))
   true_labels = true_labels[1:idx]
   predicted_assignments = predicted_assignments[1:idx]
end

# randindex 返回 (RI, ARI)
ri, ari = randindex(predicted_assignments, true_labels)
println("调整兰德指数 (ARI): ", ari)
```

#### 归一化 (normalization)互信息 (NMI)

归一化互信息 (NMI) 是一种信息论度量，它量化 (quantization)了真实标签与预测簇分配之间共享的信息量。它的归一化范围在 0 到 1 之间：

- **1**: 真实标签与聚类之间完美相关。
- **0**: 没有互信息；聚类与真实标签无关。

`Clustering.jl` 也为此目的提供了 `mutualinfo`。

```julia
using Clustering, Statistics

# 使用与 ARI 示例中相同的 true_labels 和 predicted_assignments
# true_labels = ...
# predicted_assignments = ...

nmi_score = mutualinfo(predicted_assignments, true_labels, norm=true) # norm=true 用于 NMI
println("归一化互信息 (NMI): ", nmi_score)
```

其他外部指标，如同质性、完整性和 V 度量，也能让人更清楚地了解聚类与真实类别标签的匹配程度，侧重于每个簇是否只包含单一类别的成员（同质性），以及给定类别的所有成员是否都分配到同一个簇（完整性）。

### 评估的实际考量

选择“正确”的评估指标通常取决于聚类任务的具体目标和数据的特点。

#### 确定簇的数量 ($k$)

对于 K-Means 等算法，选择簇的数量 $k$ 是一个重要步骤。内部评估指标可以指导这一选择：

- **肘部法则与 WCSS**: 绘制簇内平方和 (WCSS) 与不同 $k$ 值之间的关系图。WCSS 是样本到其最近簇中心的平方距离之和。寻找一个“肘部”点，在该点之后，增加更多簇并不会显著减少 WCSS。
- **平均轮廓法**: 计算不同 $k$ 值的平均轮廓分数。通常选择使平均轮廓分数最大化的 $k$ 值。这通常优于肘部法则，因为它同时考虑了内聚性和分离度。

这是一个使用轮廓分数帮助选择 $k$ 的可视化示例：



![平均轮廓分数 vs. 簇的数量 (k)](plots/6882-0.json)



> 示例图展示了不同 $k$ 值下的平均轮廓分数。在此示例中，$k=3$ 产生了最高的平均轮廓分数，表明它可能是最优的簇数量。

#### 可视化检查

如果可能（例如，对于 2D 或 3D 数据，或在降维后），直观检查簇可以提供有价值的定性理解，作为定量指标的补充。簇是否分离良好？它们在问题背景下是否合理？

#### 领域知识

始终要考虑领域知识。有时，从数学角度来看的“最佳”聚类，从领域角度来看可能不是最有意义或最可操作的。例如，您可能有外部原因期望客户数据中存在特定数量的细分。

评估聚类通常是一个迭代过程。您可以尝试不同的算法，调整参数 (parameter)（如 $k$），并使用多种评估指标来增强对结果的信心。很少有一个完美的答案，因此定量度量与定性判断相结合通常是最有效的方法。

## 参考资料

- [Introduction to Data Mining](https://www.pearson.com/us/higher-education/program/Tan-Introduction-to-Data-Mining-1st-Edition/PGM175058.html) — Pang-Ning Tan, Michael Steinbach, and Vipin Kumar (2005)
  Publisher: Pearson Education
  提供数据挖掘的全面概述，包括聚类算法和各种评估技术（如内部和外部度量）以及选择聚类数量的方法的详细讨论。
- [Silhouettes: A graphical aid to the interpretation and validation of cluster analysis](https://doi.org/10.1016/0377-0427%2887%2990125-7) — Peter J. Rousseeuw (1987)
  Journal: Journal of Computational and Applied Mathematics; Publisher: Elsevier; Volume: 20; Pages: 53-65; DOI: [10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427%2887%2990125-7)
  提出轮廓系数的原始研究论文，该系数用于根据簇内和簇间内聚性与分离性评估聚类质量。
- [A cluster separation measure](https://doi.org/10.1109/TPAMI.1979.4766909) — David L. Davies, Donald W. Bouldin (1979)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: PAMI-1; Pages: 224-227; DOI: [10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909)
  定义戴维森-堡丁指数的开创性论文，该指数是一种通过考虑簇内散度和簇间距离来评估聚类效果的内部度量。
- [Clustering.jl Documentation](https://juliastats.org/Clustering.jl/stable/) — JuliaStats (2022)
  Julia Clustering.jl 包的官方文档，提供各种聚类算法和评估指标（包括轮廓分数、兰德指数和互信息）的 API 细节。
