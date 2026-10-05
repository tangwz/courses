# 算法概述：IVF

来源：[原文](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-3-approximate-nearest-neighbor-search/ann-algorithm-ivf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

倒排文件索引（IVF）方法为近似最近邻搜索提供了一种独特的策略。它与基于图的方法不同，不构建连接数据点的复杂图。相反，IVF将向量 (vector)空间划分为不同的区域（或称单元），并将搜索工作仅集中于给定查询最相关的单元。这种方法受传统文本信息检索技术的启发。

### 核心思想：分而治之

想象你的高维向量 (vector)空间。IVF首先将此空间划分为预设数量的区域，通常使用聚类算法。每个区域都有一个代表向量，称为*质心*。当一个新的查询向量到来时，IVF不是将其与数据集中的每一个向量进行比较，而是首先确定查询向量最接近哪些区域（单元）。随后，它*只*在这些选定的单元内进行更集中的、通常是详尽的搜索。这与暴力搜索相比，极大地减少了所需的距离计算次数。

### 索引阶段：聚类和构建倒排文件

构建IVF索引包含两个主要步骤：

1. **聚类：** 数据集向量 (vector)使用像K-means这样的算法被分组为 $k$ 个簇。$k$ 的选择（在库参数 (parameter)中常被称为 `nlist`）很要紧，它决定了空间划分的粒度。每个簇在已划分的向量空间中代表一个单元。每个簇的质心作为其代表。
2. **创建倒排文件：** 一个“倒排文件”数据结构被创建。这个结构本质上是一个字典或哈希映射，其中键是簇ID（从0到 $k-1$），值是包含属于该特定簇的所有数据点标识符（有时也包含向量本身）的列表。

因此，索引后，你将拥有 $k$ 个代表分区的质心，以及一个能快速告诉你哪些向量驻留在哪个分区（单元）中的倒排列表。

### 查询阶段：探测单元

当对查询向量 (vector) $q$ 执行最近邻搜索时：

1. **识别候选单元：** 计算查询向量 $q$ 与所有 $k$ 个簇质心之间的距离。
2. **选择探测：** 找出最接近 $q$ 的 $n_{\text{probe}}$ 个质心。参数 (parameter) $n_{\text{probe}}$ 控制将搜索多少个单元。这是一个重要的调整参数。
3. **在选定单元内搜索：** 从倒排文件中检索与选定的 $n_{\text{probe}}$ 个单元关联的向量ID列表。在查询向量 $q$ 与*只*在这些选定单元内的向量之间执行距离计算。
4. **聚合和排序：** 收集在已搜索单元中找到的最近邻，并根据距离返回前N个结果。

这个过程大幅加快了搜索速度，因为昂贵的距离计算仅限于 $n_{\text{probe}}$ 个最有希望的单元内的向量，而不是整个数据集。

### 调整参数 (parameter)：`nlist` 和 `nprobe`

IVF的性能和准确性很大程度上取决于两个参数：

- **`nlist`（簇/单元数量）：** 这是在K-means聚类步骤中使用的 $k$ 值。

  - *更大*的 `nlist` 会创建更多、更小的单元。这可以使*在一个*单元内的搜索更快（每个单元的向量 (vector)更少），但需要检查更多单元（更高的 `nprobe`）以保持高准确性，因为真实邻居可能分散在相邻的小单元中。将查询与所有质心进行比较的初始步骤也稍微耗时更长。
  - *更小*的 `nlist` 会创建更少、更大的单元。在单元内搜索耗时更长，但你可能通过探测更少单元（更低的 `nprobe`）来获得良好的准确性。将查询与质心进行比较更快。
    最佳 `nlist` 通常取决于数据集大小和分布，通常建议的值范围从 $\sqrt{N}$ 到 $4\sqrt{N}$，其中 $N$ 是数据集大小，但通常需要进行经验性测试。
- **`nprobe`（要探测的单元数量）：** 这决定了查询时实际搜索的单元数量。

  - *更大*的 `nprobe` 增加了找到真实最近邻的可能性（更高的召回率），因为它覆盖了更多潜在相关空间。然而，这会以增加搜索延迟为代价，因为需要比较更多的向量。
  - *更小*的 `nprobe` 导致更快的搜索，但增加了遗漏位于未探测单元中的相关邻居的风险（更低的召回率）。

`nprobe` 直接控制着给定IVF索引的搜索速度和准确性（召回率）之间的权衡。找到正确的平衡对于满足应用程序要求是不可或缺的。



[交互图表：IVF：nprobe与召回率和延迟](https://apxml.com/zh/courses/vector-databases-semantic-search/chapter-3-approximate-nearest-neighbor-search/ann-algorithm-ivf#plot-7efvgx)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "IVF：nprobe与召回率和延迟",
    "xaxis": {
      "title": "nprobe（已搜索单元数量）"
    },
    "yaxis": {
      "title": "指标值",
      "range": [
        0,
        1.1
      ]
    },
    "yaxis2": {
      "title": "延迟 (ms)",
      "overlaying": "y",
      "side": "right",
      "range": [
        0,
        55
      ]
    },
    "legend": {
      "x": 0.05,
      "y": 0.95
    }
  },
  "data": [
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "召回率",
      "x": [
        1,
        5,
        10,
        20,
        40,
        60
      ],
      "y": [
        0.65,
        0.85,
        0.92,
        0.96,
        0.98,
        0.99
      ],
      "line": {
        "color": "#228be6"
      }
    },
    {
      "type": "scatter",
      "mode": "lines+markers",
      "name": "延迟 (ms)",
      "x": [
        1,
        5,
        10,
        20,
        40,
        60
      ],
      "y": [
        5,
        10,
        15,
        25,
        35,
        50
      ],
      "yaxis": "y2",
      "line": {
        "color": "#fd7e14"
      }
    }
  ]
}
```

</details>



> IVF索引中 `nprobe` 参数、搜索召回率和查询延迟之间的权衡示例。增加 `nprobe` 通常能提高召回率，但会增加延迟。

### 将IVF与量化 (quantization)结合

为进一步优化IVF，尤其是在单元内的内存使用和速度方面，它通常与向量 (vector)量化技术结合，例如乘积量化（PQ）或标量量化（SQ）。量化将倒排列表中存储的全精度向量压缩为低内存表示（例如，每个向量使用更少的字节）。

在搜索阶段（上述第3步），查询向量与探测单元内压缩向量之间的距离计算可以更快地执行，通常使用基于压缩码的近似距离计算。这引入了额外的近似层，但可以大幅减少内存占用和计算成本，使IVF对于极大型数据集变得可行。具体变体通常表示为IVF-PQ或IVF-SQ。

### IVF可视化：单元探测

下图展示了查询IVF索引的思想。空间被划分（由线条表示），质心标记 (token)每个单元。当查询到来时，最接近的单元被识别（在此示例中 `nprobe=3`），并且只搜索这些单元内的向量 (vector)。

> IVF查询的图示。查询向量（红色星形）与质心（蓝色圆形）进行比较。最接近的 `nprobe=3` 个单元（1、2、4）被选中，用于搜索其中实际的数据点（小的灰色点）。与所有其他质心（3、5、6）的距离更大，因此它们的单元被跳过。

### IVF的优点和缺点

**优点：**

- **内存效率：** 可以非常节省内存，尤其当与量化 (quantization)（如IVF-PQ）结合时。主要内存开销来自在倒排列表中存储质心和（可能量化的）向量 (vector)。
- **良好的速度/准确性平衡：** 提供了一个易于理解的机制（`nprobe`）来平衡搜索速度和召回率。
- **解耦索引：** 聚类步骤（通常是最耗时的部分）在索引构建期间离线完成。索引一旦构建完成，查询相对较快。

**缺点：**

- **索引构建时间：** 初始K-means聚类可能很慢，尤其对于非常大的数据集。
- **对数据分布的敏感性：** 如果数据分布不适合聚类（例如，非均匀密度、复杂簇形状），性能可能会下降。
- **参数 (parameter)调整：** 找到最佳的 `nlist` 和 `nprobe` 值通常需要在特定数据集和查询负载上进行细致的实验和评估。
- **静态特性：** 索引一旦构建，添加新的数据点通常需要定期重新训练或重建索引结构，尽管有些实现提供了增量添加功能，但可能伴随性能权衡。

IVF代表了一种基础且广泛使用的方法，用于近似最近邻搜索。它有效地划分了搜索空间，并提供了明确的参数来调整性能和准确性之间的平衡，使其成为向量数据库工具集中一种有价值的算法，经常与HNSW等基于图的方法一起使用或作为其替代。

## 参考资料

- [Searching in one billion vectors](https://inria.hal.science/inria-00514462/file/) — Hervé Jégou, Matthijs Douze, Cordelia Schmid (2011)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 33; Pages: 117-128; DOI: [10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
  介绍了IVFADC方法，该方法将倒排文件索引与乘积量化和非对称距离计算相结合，用于大规模数据集中的高效相似性搜索。
- [Product Quantization for Nearest Neighbor Search](https://inria.hal.science/inria-00514462/file/) — Hervé Jégou, Matthijs Douze, Cordelia Schmid (2011)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE Computer Society; Volume: 33; Pages: 117-128; DOI: [10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
  提出了乘积量化技术，用于压缩高维向量并有效估计其距离，是IVF-PQ等方法的基础。
- [FAISS Wiki: IVFPQ](https://github.com/facebookresearch/faiss/wiki/IVFPQ) — Meta AI Research (2025)
  FAISS库中IVF和IVF-PQ实现的官方文档，提供了技术细节和参数说明，以便实际使用。

---

[上一节](03-%E7%AE%97%E6%B3%95%E6%A6%82%E8%A7%88%EF%BC%9AHNSW.md) · [下一节](05-%E7%AE%97%E6%B3%95%E6%A6%82%E8%BF%B0%EF%BC%9ALSH.md)
