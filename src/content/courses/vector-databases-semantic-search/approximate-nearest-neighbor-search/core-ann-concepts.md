---
course: "vector-databases-semantic-search"
chapter: "approximate-nearest-neighbor-search"
lesson: "core-ann-concepts"
sourceId: 3541
sourceUrl: "https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-3-approximate-nearest-neighbor-search/core-ann-concepts"
title: "ANN 的核心思想"
description: "了解搜索准确度（召回率）与速度/资源消耗之间的平衡。"
order: 2
plots: ["plots/3541-0.json"]
sourceHash: "27c7036b80cb7401489ba195030cd8d8bf01544fe61fa78804f91737674770a9"
sourceCorrections: []
---

绝对*精确*的最近邻搜索在高维向量 (vector)空间中可能非常缓慢。想象一下，在一个包含数百万或数十亿条数据的数据库中，尝试计算查询向量与*每一个*向量之间的距离——这种计算开销很快就会变得不切实际，无法用于即时应用。这时，近似最近邻 (ANN) 搜索就发挥作用了。

ANN 背后的基本理念简单而有效：**以少量搜索准确度为代价，换取速度和效能的显著提升。** ANN 算法并非旨在确保完全最相似的对应项，而是力求以极快的速度找到*大部分*最相似的对应项，或者与查询*非常接近*的点。

### 准确度与速度的取舍

可以把它想象成在线搜索信息。一个完美的搜索引擎会阅读并理解互联网上的每份文档，以给出唯一的最佳答案（就像精确最近邻搜索）。而搜索引擎使用巧妙的索引机制来快速返回*高度相符*的结果，即使无法保证绝对的最佳结果排在首位（就像 ANN 搜索）。对于许多应用，特别是语义搜索或推荐系统，其相似度本身有些不明确，极快地找到“足够好”的邻居，远比缓慢地找到计算上完美的邻居更有用。

这引出了 ANN 的核心思想：**取舍关系图**。通常无法同时让准确度和速度都达到最高。提升其中一个，通常会以牺牲另一个为代价。我们需要方法来评估和管理这种均衡。

### 评估取舍：衡量参数 (parameter)

在使用 ANN 时，我们通过以下几个衡量参数来评估表现：

1. **召回率 (Recall)：** 这是 ANN 应用中搜索准确度的主要衡量标准。Recall@k 通常衡量真实的 *k* 个最近邻（由精确搜索得出）中有多少百分比包含在 ANN 算法返回的前 *k* 个结果中。例如，召回率为 0.95 意味着 ANN 搜索找到了 95% 的实际最近邻。召回率越高，准确度越高，越接近精确搜索的结果。
2. **延迟 (Latency)：** 这是完成单次搜索查询所需的耗时。更低的延迟对交互式系统非常重要。ANN 算法旨在相比精确搜索大幅降低延迟。通常以毫秒 (ms) 为单位衡量。
3. **吞吐量 (throughput) (QPS)：** 每秒查询次数 (QPS) 衡量系统能同时应对多少搜索查询。更高的 QPS 表明在压力下有更好的效能和可扩展性。
4. **索引构建时间：** 构建用于加快搜索的专用数据结构（即 ANN 索引）需要耗时。这通常是一次性或不常发生的离线开销。
5. **内存占用：** ANN 索引结构会占用内存 (RAM)。索引的大小是一个重要的因素，特别是对于大规模数据集，因为它关系到硬件需求和费用。

### 呈现均衡关系

召回率与延迟之间的关系通常如下所示：



![典型的 ANN 取舍关系图](plots/3541-0.json)



> 这是一种常见情况，即提高目标召回率通常会引起查询延迟增加。具体的图线主要受算法、数据集和参数 (parameter)配置的影响。

### 可调整性：找到适合点

重要的是，ANN 算法并非固定不变；它们提供参数 (parameter)，使您能够**调节**这种取舍。通过在索引构建期间或查询时（取决于算法和具体实现）调整这些参数，您可以调整这种均衡。您可以将索引设置为：

- **高召回率：** 优先考虑找到尽量多的实际邻居，接受可能更高的延迟和内存占用。适用于离线分析或准确度重要的任务。
- **低延迟：** 优先考虑快速响应时间，接受略低的召回率。非常适合即时推荐系统或交互式搜索系统。
- **均衡策略：** 找到一个折中方案，能在特定使用场景下提供良好召回率和可接受的表现。

“了解准确度（召回率）与表现（延迟、资源）之间的这种核心取舍，是在后续章节中学习 HNSW、IVF 或 LSH 等具体 ANN 算法前不可或缺的。这些算法提供不同的策略和参数，用于高效地处理这种折中。ANN 通过认识到在许多情况下，快速提供的接近完美的结果远胜于速度过慢的完美结果，从而使大规模向量 (vector)搜索变为可能。”

## 参考资料

- [Approximate Nearest Neighbor Search in High Dimensions](https://link.springer.com/chapter/10.1007/978-3-319-94770-9_3) — Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, Ludwig Schmidt (2018)
  Journal: Algorithms and Data Structures. WADS 2017. Lecture Notes in Computer Science, vol 10999; Publisher: Springer; Volume: 10999; Pages: 30-69; DOI: [10.1007/978-3-319-94770-9_3](https://doi.org/10.1007/978-3-319-94770-9_3)
  对近似最近邻搜索的更新综合调查，详细说明了其动机、挑战和各种算法方法。
- [Mining of Massive Datasets](http://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeff Ullman (2020)
  Publisher: Cambridge University Press
  第三章介绍了高效查找相似项的问题，激发了高维数据的近似方法。
- [Benchmarking Approximate Nearest Neighbor Search Algorithms for High-Dimensional Vectors](http://www.vldb.org/pvldb/vol10/p1817-aumueller.pdf) — Bernhard Aumüller, Erik Heinlein, Alexander Reiser, Thomas Seidl (2017)
  Journal: Proceedings of the VLDB Endowment; Publisher: VLDB Endowment; Volume: 10; Pages: 1817-1828; DOI: [10.14778/3137628.3137637](https://doi.org/10.14778/3137628.3137637)
  讨论了评估和比较ANN算法的基本指标，如召回率、延迟和每秒查询数（QPS），这与理解准确性-速度权衡直接相关。
