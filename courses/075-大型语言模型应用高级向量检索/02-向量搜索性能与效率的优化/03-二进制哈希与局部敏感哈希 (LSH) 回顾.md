# 二进制哈希与局部敏感哈希 (LSH) 回顾

来源：[原文](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance/binary-hashing-lsh-refresher)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然量化 (quantization)技术通过压缩向量 (vector)来优化存储并加速距离计算，但哈希方法为近似最近邻搜索提供了一种截然不同的方法。局部敏感哈希（LSH）等技术并非直接压缩向量，而是旨在利用专用哈希函数将相似向量分组到相同的“桶”中。二进制哈希和LSH的原理及其在优化向量搜索系统中的作用将被讨论。

### 从向量 (vector)到比特：二进制哈希

其核心是，二进制哈希旨在将高维浮点向量转换为紧凑的二进制码，通常表示为比特串。主要动力是效率：对二进制码的操作，尤其是计算汉明距离（不同比特的数量），能够比在原始向量上计算欧几里得或余弦距离快得多。此外，二进制码所需的存储空间也少得多。

一个简单、具有说明性的方法可能涉及定义一组阈值。对于向量 $v = [v_1, v_2, ..., v_d]$，我们可以生成一个二进制哈希 $h = [b_1, b_2, ..., b_d]$，其中如果 $v_i > \text{阈值}_i$ 则 $b_i = 1$，否则 $b_i = 0$。另一种基本技术涉及将向量投影到一组随机向量上，并根据投影的符号分配一个比特。

然而，简单的二进制哈希方法通常难以有效地保持原始向量的*局部性*。两个在原始高维空间 (high-dimensional space)中距离接近的向量，可能会得到非常不同的二进制码，反之亦然。这一局限性促成了更精巧的技术的发展，这些技术专门为相似性搜索设计。

### 局部敏感哈希 (LSH)

局部敏感哈希 (LSH) 通过采用具有特定、数学定义的属性的哈希函数来解决朴素哈希的不足：**碰撞概率**。LSH函数族的设计方式是：

1. *相互接近*的向量 (vector)在原始向量空间中，有*很高概率*哈希到相同的值（发生碰撞）。
2. *相距较远*的向量在原始向量空间中，有*很低概率*哈希到相同的值。

更正式地说，对于距离度量 $D$，一个哈希函数族 $\mathcal{H}$ 称为 $(r_1, r_2, p_1, p_2)$-敏感的，如果对于任意两个向量 $x, y$：

- 如果 $D(x, y) \le r_1$，那么 $P_{h \in \mathcal{H}}[h(x) = h(y)] \ge p_1$
- 如果 $D(x, y) \ge r_2$，那么 $P_{h \in \mathcal{H}}[h(x) = h(y)] \le p_2$
  其中 $r_1 < r_2$ 且 $p_1 > p_2$。目标是使 $p_1$ 和 $p_2$ 之间的差距最大化。

#### 常用相似度度量的LSH族

不同的LSH族适用于不同的距离或相似度度量：

- **随机投影（用于余弦相似度/角距离）：** 这可能是最适合大型语言模型应用中常用的稠密嵌入 (embedding)的LSH族，这些应用经常依赖余弦相似度。核心思想是生成几个通过原点的随机超平面。每个超平面将空间分成两半。向量的哈希码（或其一部分）由它落在这些超平面的哪一侧决定。对于由随机法向量 $u$ 定义的单个超平面，哈希函数可以是：如果 $u \cdot v \ge 0$ 则 $h(v) = 1$，如果 $u \cdot v < 0$ 则 $h(v) = 0$。向量之间角度越小（余弦相似度越高）的向量越有可能落在同一随机超平面的同一侧，从而有更高的碰撞概率。多个这样的哈希函数（使用不同的随机向量 $u$）结合起来创建一个更长的哈希码，提高区分度。

> 向量 v1 和 v2 距离接近（夹角小），并落在随机超平面的同一侧 (h(v1)=h(v2)=1)，这使得碰撞更有可能。向量 v3 距离 v1/v2 较远，并落在另一侧 (h(v3)=0)。

- **最小哈希 (MinHash)（用于Jaccard相似度）：** 主要用于比较集合（例如，文档中的词语集合）。虽然对于来自大型语言模型的稠密向量嵌入来说不那么常见，但它是一种经典的LSH技术。它根据项目随机排列后两个集合产生相同最小元素的概率来估算Jaccard相似度。
- **p-稳定分布（用于$L_p$距离，例如欧几里得距离）：** 该族使用向随机直线投影，然后进行量化 (quantization)。哈希函数是根据从p-稳定分布中采样构建的（例如用于$L_2$/欧几里得距离的高斯分布）。在$L_p$距离上更近的向量，在投影和量化后，更有可能映射到接近的值。

#### 使用LSH进行近似最近邻搜索

使用LSH实现ANN搜索通常涉及以下步骤：

1. **索引构建：**

   - 根据距离度量选择合适的LSH族（例如，用于余弦相似度的随机投影）。
   - 生成多个哈希表。每个表使用一组不同的哈希函数（例如，通过连接$k$个基本LSH函数，如随机超平面测试）。
   - 使用每个表的函数哈希数据库中的每个向量，并将向量（或其ID）存储到相应的桶中。
2. **查询：**

   - 使用每个表的相同哈希函数集哈希查询向量。
   - 识别查询向量在每个哈希表中落入的桶。
   - 收集所有表中这些桶中存在的唯一向量（候选对象）。
3. **精细化：**

   - 计算查询向量与上一步检索到的每个候选向量之间的精确距离（例如，余弦相似度）。
   - 返回距离最优的前$k$个候选对象。

#### 配置与权衡

LSH的性能在很大程度上取决于其参数 (parameter)：

- **哈希表的数量 ($L$)：** 增加 $L$ 提高了找到近邻的概率（召回率更高），因为在一个表中遗漏的邻居可能在另一个表中被找到。然而，这也增加了查询时间，因为需要检查更多的桶。
- **每个表的哈希函数数量 ($k$)：** 增加 $k$ 使哈希码更长、更具区分度。这减少了桶的平均大小（每个桶的假阳性更少），但也降低了真实近邻的碰撞概率（如果 $L$ 没有相应增加，可能会导致召回率降低）。通常需要仔细调整。

找到 $L$ 和 $k$ 的最佳值对于平衡召回率、查询延迟和内存使用非常重要。

### LSH在现代向量 (vector)搜索中的作用

尽管LSH为ANN搜索提供了概率性方法，并且是早期一项重要的进展，但其在现代嵌入 (embedding)模型生成的稠密向量上的实际运用面临竞争。

- **与HNSW/IVF的比较：** HNSW等图基方法在典型稠密向量基准测试中，在给定查询速度下通常能获得更高的召回率，与LSH相比。IVF索引与乘积量化 (quantization)（IVFPQ）结合通常在存储压缩向量方面提供更好的内存效率，与LSH为获得高召回率可能需要的大型哈希表相比。
- **调优复杂性：** 调整LSH参数 (parameter)（$L$、$k$，以及哈希函数本身的参数）有时不如调整HNSW（例如 `efConstruction`、`efSearch`）或IVF（`nprobe`）参数直观。
- **优势：** LSH在某些场景下仍然具有优势。与复杂的图结构相比，它可能更容易分发或更新。它的性能有时对数据的固有维度不那么敏感，与某些基于树的方法相比。一些专业数据库系统运用LSH变体。

理解LSH仍然很有价值。它提供了理论见解，关于高维搜索的挑战，并代表了一种独特的算法方法，与分区/图方法（IVF、HNSW）和向量压缩（量化）不同。尽管在最先进的大型语言模型应用中，你可能更常遇到HNSW或量化，但LSH是优化向量搜索工具包的一部分，尤其是在内存或特定性能权衡很重要时，或者在处理其他方法可能不容易应用的非标准距离度量时。它提供了一个理解为什么近似是必需的途径，以及不同的策略如何应对“维度灾难”。

## 参考资料

- [Similarity Search in High Dimensions via Hashing](http://portal.acm.org/citation.cfm?id=671516) — Aristides Gionis, Piotr Indyk, Rajeev Motwani (1999)
  Journal: Proceedings of the 25th International Conference on Very Large Data Bases (VLDB '99); Publisher: Morgan Kaufmann Publishers Inc.; Pages: 518-529; DOI: [10.5555/635606.635655](https://doi.org/10.5555/635606.635655)
  介绍了早期的局部敏感哈希理论和算法，包括MinHash，用于高维相似性搜索。
- [Mining of Massive Datasets](https://www.mmds.org/) — Jure Leskovec, Anand Rajaraman, Jeffrey David Ullman (2020)
  Publisher: Cambridge University Press; Pages: Chapter 3: Finding Similar Items
  一本标准教材，提供了局部敏感哈希、MinHash及其他相似性搜索方法的可理解解释。（第三版）
- [Similarity Estimation Techniques from Rounding Algorithms](https://dl.acm.org/doi/10.1145/509907.509965) — Moses S. Charikar (2002)
  Journal: Proceedings of the 34th Annual ACM Symposium on Theory of Computing (STOC); Publisher: ACM; Pages: 380-388; DOI: [10.1145/509907.509965](https://doi.org/10.1145/509907.509965)
  提出了符号随机投影（SRP）作为一种局部敏感哈希技术，能有效估算高维空间中的余弦相似度。
- [Locality-Sensitive Hashing Scheme Based on p-Stable Distributions](https://doi.org/10.1145/997817.997857) — Mayur Datar, Nicole Immorlica, Piotr Indyk, Vahab S. Mirrokni (2004)
  Journal: Proceedings of the Twentieth Annual Symposium on Computational Geometry (SoCG'04); Publisher: ACM; Pages: 253-262; DOI: [10.1145/997817.997857](https://doi.org/10.1145/997817.997857)
  提出了一种基于p稳定分布的局部敏感哈希族，适用于Lp距离度量，例如欧几里得距离。

---

[上一节](02-%E5%AE%9E%E7%8E%B0%E4%BC%98%E5%8C%96%E4%B9%98%E7%A7%AF%E9%87%8F%E5%8C%96%20%28OPQ%29.md) · [下一节](04-%E9%AB%98%E7%BA%A7%E8%BF%87%E6%BB%A4%E7%AD%96%E7%95%A5%EF%BC%9A%E9%A2%84%E8%BF%87%E6%BB%A4%E4%B8%8E%E5%90%8E%E8%BF%87%E6%BB%A4.md)
