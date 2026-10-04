# 实现优化乘积量化 (OPQ)

来源：[原文](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance/implementing-opq)

[返回章节目录](README.md) · [返回课程目录](../README.md)

乘积量化 (quantization)（PQ）通过独立压缩向量 (vector)子空间，可以显著减少内存占用。然而，它的效果取决于向量最初如何划分。标准PQ通常将向量划分为连续的块。如果数据分布在这些任意边界上存在强相关性，或者方差只集中在少数几个子空间中，这种划分策略可能不是最优的，会导致更高的量化误差和降低的搜索准确性。

优化乘积量化（OPQ）通过引入一个预处理步骤解决了这一限制：它在应用PQ*之前*学习向量空间的最佳旋转。目标是将数据变换，使方差在PQ将创建的子空间中分布更均匀。这种旋转有助于使数据分布更好地与子空间划分对齐 (alignment)，从而最小化量化过程中信息丢失。

### 旋转预处理步骤

OPQ 的核心是一个正交矩阵 $R$。正交矩阵代表空间中的旋转（或反射），保留向量 (vector)之间的距离和角度。在向量 $x$ 使用PQ量化 (quantization)之前，它首先通过这个矩阵进行变换：


$$
x' = Rx
$$


然后，这个旋转后的向量 $x'$ 被划分为子向量，并对这些子向量应用标准PQ。为什么这有帮助呢？想象你的数据点在空间中形成一个拉长的云团。如果使用标准PQ沿着原始坐标轴划分这个云团，一些子量化器可能会处理非常分散的数据（高方差），而另一些则处理紧密聚集的数据（低方差），导致码本使用效率低下。OPQ学习一个旋转 $R$，理想地调整这个云团的方向，以便当 $x' = Rx$ 被划分时，每个结果子空间捕获相似的方差量，从而使每个子空间的PQ码本得到更有效的使用。

> 对比显示OPQ在标准PQ处理前增加了一个旋转步骤。

### 学习最佳旋转

旋转矩阵 $R$ 不是随意设定的；它是从有代表性的训练数据样本中学习得到的。目的是找到正交矩阵 $R$，在对旋转后的向量 (vector) $Rx$ 应用PQ后，最小化整体量化 (quantization)误差。

这个学习过程通常是迭代的：

1. **初始化**：从一个初始旋转矩阵 $R$ 开始（通常是单位矩阵）。
2. **PQ码本更新**：保持 $R$ 不变，对当前旋转的训练数据 ($Rx$) 训练PQ码本。这本质上是将标准PQ训练步骤应用于变换后的数据。
3. **旋转矩阵更新**：保持PQ码本不变，找到正交矩阵 $R$，使旋转后的向量 $Rx$ 与其使用当前码本的量化表示之间的误差最小化。这个子问题通常可以高效地解决，例如，使用与应用于量化误差协方差矩阵的奇异值分解（SVD）相关的技术，或者通过解决正交Procrustes问题。
4. **迭代**：重复步骤2和3，直到量化误差收敛或达到最大迭代次数。

这种迭代过程共同微调 (fine-tuning)旋转和PQ码本，产生一个组合变换，可以更好地保留压缩表示中的原始向量信息。

### OPQ的索引和搜索

**索引**: 一旦旋转矩阵 $R$ 和相应的PQ码本学习完成，索引步骤包括：

1. 对每个数据库向量 (vector)应用旋转：$x_{db}' = Rx_{db}$。
2. 对旋转后的向量 $x_{db}'$ 应用训练好的PQ编码器，以获得其压缩码。
   这些表示旋转向量的PQ码存储在索引中。

**搜索**: 在查询时，必须对查询向量 $q$ 应用相同的旋转：

1. 计算旋转后的查询向量：$q' = Rq$。
2. 使用 $q'$ 和数据库向量的存储PQ码进行近似距离计算。这通常涉及计算 $q'$ 的子向量与数据库码对应的PQ码本质心之间的距离。

Faiss 等库抽象了大部分这种复杂性，允许您在索引配置字符串中将OPQ指定为一个预处理步骤。

```python
# Faiss 示例，演示 OPQ 的集成

import faiss
import numpy as np

d = 128      # 向量维度
nb = 100000  # 数据库大小
nt = 20000   # 训练集大小
nq = 1000    # 查询数量

# 生成一些随机数据
xt = np.random.rand(nt, d).astype('float32') * 10
xb = np.random.rand(nb, d).astype('float32') * 10
xq = np.random.rand(nq, d).astype('float32') * 10

# PQ 参数
m = 16  # 子空间数量
bits = 8 # 每个码的比特数

# IVF 参数（常与 OPQ/PQ 一起使用）
nlist = 256 # 粗量化单元数量

# 集成 OPQ 的 Faiss 索引工厂字符串
# 格式: "OPQ<m_opq>,[IVF<nlist>,]PQ<m>x<bits>"
# m_opq 指定用于 OPQ 训练旋转的子空间数量（可与 m 相同）
index_string = f"OPQ{m},IVF{nlist},PQ{m}x{bits}"

# 构建索引
print(f"正在构建 Faiss 索引: {index_string}")
index = faiss.index_factory(d, index_string, faiss.METRIC_L2)

# 检查索引是否需要训练
if not index.is_trained:
    print("正在训练索引...")
    # 训练过程学习 IVF 质心、OPQ 旋转矩阵 R，
    # 以及基于旋转数据的 PQ 码本。
    index.train(xt)
else:
    print("索引不需要单独的训练步骤。") # 例如，对于 Flat 索引

print("正在添加数据库向量...")
# 在添加过程中，向量会自动处理：
# 1. 旋转：x' = Rx
# 2. 根据 x' 分配到 IVF 单元
# 3. 根据残差 (x' - 质心) 进行 PQ 编码
index.add(xb)

print(f"索引包含 {index.ntotal} 个向量。")

# 设置 IVF 搜索参数（要访问的单元数量）
index.nprobe = 16 # nprobe 值越高意味着准确性越好，搜索速度越慢

print("正在执行搜索...")
k = 5 # 查找 5 个最近邻
# 在搜索过程中，查询向量 q 会：
# 1. 旋转：q' = Rq
# 2. 根据 q' 选择 IVF 单元
# 3. 使用所选单元中的 PQ 码计算距离
distances, indices = index.search(xq, k)

print(f"搜索完成。第一个查询的示例结果：")
print(f"索引: {indices[0]}")
print(f"距离: {distances[0]}")
```

### 权衡和考量

实现 OPQ 需要平衡其益处和成本：

- **提高准确性**：OPQ 的主要优点是，在相同压缩水平（内存使用）下，其搜索准确性（召回率）可能高于标准PQ。通过减少量化 (quantization)误差，计算出的近似距离更接近真实距离。
- **增加索引时间**：学习旋转矩阵 $R$ 会为索引训练阶段增加显著的计算成本。这涉及迭代优化以及可能对大型矩阵进行SVD计算，使得OPQ索引构建比标准PQ慢。
- **查询转换开销**：在搜索开始之前，旋转矩阵 $R$ 必须应用于每个查询向量 (vector)。虽然矩阵乘法通常很快，但与标准PQ相比（标准PQ中查询向量直接使用或只需要子向量提取），它为每次查询增加了少量延迟。
- **复杂性**：实现和调整过程本身就比标准PQ更复杂。

OPQ 在以下情况下特别有益：

- 向量维度显示出显著相关性。
- 在严格内存限制下最大化召回率是主要目标。
- 额外的索引构建时间可以接受。

对于维度已经合理去相关或索引构建时间受到高度限制的数据集，OPQ 的收益可能不足以抵消其成本，与标准PQ等更简单的技术相比。正如许多优化技术一样，通过实验（第五章涵盖）评估它在您的特定数据和应用要求上的有效性是必不可少的。

## 参考资料

- [Product Quantization for Nearest Neighbor Search](https://doi.org/10.1109/TPAMI.2010.57) — Hervé Jégou, Matthijs Douze, Cordelia Schmid (2011)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Volume: 33; Pages: 117-128; DOI: [10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
  这项开创性工作介绍了乘积量化（PQ），即OPQ扩展的核心技术，提供了一种用于高维向量压缩和近似最近邻搜索的方法。
- [Faiss: A Library for Efficient Similarity Search](https://proceedings.neurips.cc/paper_files/paper/2019/file/e6f4b669e46a9b49ad617b35f6e890c0-Paper.pdf) — Hervé Jégou, Matthijs Douze, Armand Joulin (2019)
  Journal: Advances in Neural Information Processing Systems (NeurIPS 2019 Systems and Demonstrations)
  这篇论文描述了Faiss库，它提供了各种近似最近邻搜索算法的高效实现，包括OPQ，如章节中的代码示例所示。

---

[上一节](01-%E9%87%8F%E5%8C%96%E6%8A%80%E6%9C%AF%EF%BC%9A%E6%A0%87%E9%87%8F%E9%87%8F%E5%8C%96%E4%B8%8E%E4%B9%98%E7%A7%AF%E9%87%8F%E5%8C%96.md) · [下一节](03-%E4%BA%8C%E8%BF%9B%E5%88%B6%E5%93%88%E5%B8%8C%E4%B8%8E%E5%B1%80%E9%83%A8%E6%95%8F%E6%84%9F%E5%93%88%E5%B8%8C%20%28LSH%29%20%E5%9B%9E%E9%A1%BE.md)
