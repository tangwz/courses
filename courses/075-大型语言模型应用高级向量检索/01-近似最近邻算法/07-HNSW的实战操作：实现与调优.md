# HNSW的实战操作：实现与调优

来源：[原文](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-1-ann-algorithms/hands-on-implementing-tuning-hnsw)

[返回章节目录](README.md) · [返回课程目录](../README.md)

HNSW是一种用于近似最近邻搜索的算法，其基础是分层图结构和贪婪搜索启发式算法。实践中实现HNSW索引，涉及使用一个常用的Python库来构建索引、执行搜索，并调整其参数 (parameter)以平衡性能和准确性，满足特定要求。我们假设您有一个可用的Python环境，并熟悉用于数值运算的NumPy。

### 环境设置

首先，确保您已安装所需的库。我们将使用`hnswlib`，这是一个流行且高效的HNSW实现。您还需要`numpy`来创建向量 (vector)数据。

```bash
pip install hnswlib numpy
```

本示例将使用合成数据。让我们生成一些随机的128维向量，模拟语言模型中常见的嵌入 (embedding)输出。

```python
import hnswlib
import numpy as np
import time

# 定义数据参数
dim = 128  # 向量的维度
num_elements = 10000  # 数据集中的向量数量

# 生成随机数据（替换为您实际的嵌入）
np.random.seed(42) # 用于复现性
data = np.float32(np.random.random((num_elements, dim)))

# 为每个向量生成唯一ID
data_labels = np.arange(num_elements)

# 生成少量查询向量
num_queries = 5
query_data = np.float32(np.random.random((num_queries, dim)))
```

### 构建HNSW索引

初始化和填充HNSW索引需要指定空间（距离度量）、维度，然后添加数据点。

```python
# 初始化HNSW索引
# 可能的空间选项：'l2'，'ip'（内积），'cosine'（余弦）
space_name = 'l2' # 欧氏距离
index = hnswlib.Index(space=space_name, dim=dim)

# 在添加数据之前设置索引参数
# M：每层每个节点的最大连接数（默认16）
# ef_construction：构建过程中邻居动态列表的大小（默认200）
# 更高的ef_construction值会带来更好的索引质量，但构建时间更长
M = 16
ef_construction = 200
index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

# 将数据添加到索引
# ID应为唯一整数
print("正在向索引添加数据...")
start_time = time.time()
index.add_items(data, data_labels)
build_time = time.time() - start_time
print(f"索引构建完成，耗时 {build_time:.2f} 秒。")

# 可选：控制用于索引的线程数
# index.set_num_threads(4) # 使用4个线程
```

这里，`init_index`准备索引结构。`max_elements`应至少等于您打算添加的元素数量。`M`控制图连接的密度，影响内存使用和搜索质量。`ef_construction`影响构建过程中邻居搜索的完整性；更高的值通常会带来更好的召回率，但会增加索引时间。

### 执行近似最近邻搜索

索引构建完成后，您可以使用`knn_query`方法执行搜索。

```python
# 设置搜索时参数ef_search
# ef_search：搜索过程中邻居动态列表的大小
# 更高的ef_search值会带来更好的召回率，但搜索时间更长
# 必须 >= k（请求的邻居数量）
ef_search = 50
k = 10 # 要检索的最近邻数量

print(f"\n正在对 {num_queries} 个查询执行 k-NN 搜索 (k={k}, ef_search={ef_search})...")

index.set_ef(ef_search) # 为查询会话设置ef_search

start_time = time.time()
labels, distances = index.knn_query(query_data, k=k)
search_time = time.time() - start_time

print(f"搜索完成，耗时 {search_time:.4f} 秒。")
print(f"平均查询时间：{search_time / num_queries:.4f} 秒。")

# 显示第一个查询的结果
print("\n第一个查询的结果：")
print("标签：", labels[0])
print("距离：", distances[0])
```

`knn_query`方法返回两个数组：包含近似最近邻整数ID的`labels`，以及包含相应计算距离（例如，本例中的L2距离）的`distances`。这里重要的参数 (parameter)是`ef_search`（通过`set_ef`设置）。它控制搜索遍历期间候选列表的大小。增加`ef_search`会使搜索更全面，可能会提高准确性（召回率），但代价是更高的延迟。它必须至少为`k`。

### HNSW参数 (parameter)调整：召回率/速度的权衡

使用HNSW这类ANN算法的主要难点在于平衡准确性（召回率）和性能（查询延迟、构建时间、内存使用）。主要的调整选项有：

1. **`M` (构建时间):** 定义图层中节点的最大出站连接数（0层除外）。更高的`M`值会创建更密集的图，可能提高召回率并有助于更好的导航，但会明显增加内存占用和构建时间。典型值范围为8到64。
2. **`ef_construction` (构建时间):** 控制索引构建的质量。更高的值意味着在插入元素时对邻居进行更全面的搜索，从而可能形成结构更好的图（更高的召回率），但构建时间会更长。典型值范围为100到2000或更高，具体取决于数据集和所需的质量。
3. **`ef_search` (搜索时间):** 控制查询时搜索速度和召回率之间的权衡。它决定了图遍历期间使用的优先队列的大小。更高的值会增加找到真正最近邻的几率，但会减慢查询速度。这通常是索引构建完成后最常调整的参数。它必须至少为`k`。

**实用的调优策略：**

1. **建立真实值：** 为了准确调优，您需要真实值：查询集的实际$k$个最近邻。这通常需要运行精确的暴力搜索。这对于较小的数据集或有代表性的样本是可行的。

```python
# 示例：查找精确邻居（计算成本高昂！）
# from sklearn.neighbors import NearestNeighbors
# print("正在计算真实值（这可能需要一些时间）...")
# nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(data)
# true_distances, true_indices = nbrs.kneighbors(query_data)
```

2. **定义衡量指标：** 衡量：
   - **召回率@k:** HNSW在其前$k$个结果中找到的真实$k$个最近邻的比例，对所有查询取平均值。
     
     $$
     \text{召回率@k} = \frac{1}{N_q} \sum_{i=1}^{N_q} \frac{| \text{ANN}_k(q_i) \cap \text{True}_k(q_i) |}{k}
     $$
     
     其中 $N_q$ 是查询数量，$\text{ANN}_k(q_i)$ 是HNSW为查询 $q_i$ 返回的$k$个邻居，且 $\text{True}_k(q_i)$ 是真实的$k$个邻居。
   - **查询延迟：** 每个查询的平均耗时。
   - **索引大小：** 存储索引所需的内存。
   - **构建时间：** 构建索引所花费的时间。
3. **迭代调优：**
   - 固定`M`和`ef_construction`（从M=16，ef\_construction=200等默认值开始）。构建索引。
   - 调整`ef_search`（例如，从$k$到几百）。对于每个值，衡量召回率@k和平均查询延迟。
   - 绘制召回率@k与延迟的关系图。该曲线会显示当前索引配置的权衡关系。
   - 如果即使在`ef_search`很高的情况下也无法达到所需的召回率，或者如果延迟过高，请使用增加的`ef_construction`或`M`重建索引。请注意，增加这些参数会增加构建时间并可能增加索引大小（尤其是`M`）。
   - 重复`ef_search`的遍历和评估。

**可视化示例：**

让我们模拟不同`ef_search`值的结果，并绘制权衡关系图。假设我们已针对某些真实值计算了召回率。

````python
# 模拟调优结果（替换为实际测量值）
ef_values = [10, 20, 50, 100, 200, 400]
recalls = [0.75, 0.85, 0.92, 0.96, 0.98, 0.99] # 示例召回率值
latencies_ms = [0.5, 0.8, 1.5, 2.8, 5.0, 9.5] # 示例延迟（毫秒）

import json

chart_data = {
  "layout": {
    "title": "HNSW调优：召回率@10 vs. 查询延迟",
    "xaxis": {"title": "平均查询延迟 (毫秒)"},
    "yaxis": {"title": "召回率@10", "range": [0.7, 1.0]},
    "legend": {"title": "ef_search"},
    "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "x": latencies_ms,
      "y": recalls,
      "mode": "lines+markers",
      "type": "scatter",
      "name": "召回率",
      "text": [f"ef={ef}" for ef in ef_values], # 鼠标悬停时显示的文本
      "marker": {"color": "#339af0", "size": 8},
      "line": {"color": "#339af0", "width": 2}
    }
  ]
}

# 显示图表（在markdown中嵌入JSON）
print("```plotly")
print(json.dumps(chart_data))
print("```")
````



[交互图表：HNSW调优：召回率@10 vs. 查询延迟](https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-1-ann-algorithms/hands-on-implementing-tuning-hnsw#plot-1czmub9)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "layout": {
    "title": "HNSW调优：召回率@10 vs. 查询延迟",
    "xaxis": {
      "title": "平均查询延迟 (毫秒)"
    },
    "yaxis": {
      "title": "召回率@10",
      "range": [
        0.7,
        1.0
      ]
    },
    "legend": {
      "title": "ef_search"
    },
    "margin": {
      "l": 50,
      "r": 50,
      "t": 50,
      "b": 50
    },
    "width": 600,
    "height": 400
  },
  "data": [
    {
      "x": [
        0.5,
        0.8,
        1.5,
        2.8,
        5.0,
        9.5
      ],
      "y": [
        0.75,
        0.85,
        0.92,
        0.96,
        0.98,
        0.99
      ],
      "mode": "lines+markers",
      "type": "scatter",
      "name": "召回率",
      "text": [
        "ef=10",
        "ef=20",
        "ef=50",
        "ef=100",
        "ef=200",
        "ef=400"
      ],
      "marker": {
        "color": "#339af0",
        "size": 8
      },
      "line": {
        "color": "#339af0",
        "width": 2
      }
    }
  ]
}
```

</details>



> 该图表显示了增加`ef_search`通常会提高召回率，但也会增加查询延迟。最佳的`ef_search`取决于应用对准确性和速度的特定要求。

### 索引的保存与加载

对于生产环境的使用，您会希望保存已调优的索引并加载它，而无需重建。

```python
# 将索引保存到磁盘
index_path = 'my_hnsw_index.bin'
print(f"\n正在将索引保存到 {index_path}...")
index.save_index(index_path)
print("索引已保存。")

# 加载索引（在新会话或脚本中）
# 需要知道创建时使用的dim和space_name
loaded_index = hnswlib.Index(space=space_name, dim=dim)
print(f"\n正在从 {index_path} 加载索引...")
loaded_index.load_index(index_path)
print("索引已加载。")

# 您现在可以设置ef并查询加载的索引
loaded_index.set_ef(ef_search)
labels_loaded, distances_loaded = loaded_index.knn_query(query_data, k=k)

# 验证结果是否相同
assert np.array_equal(labels, labels_loaded), "加载的索引结果不同！"
print("加载的索引查询成功并与原始结果匹配。")
```

本次动手练习体现了使用HNSW的基本步骤：初始化、构建、查询，以及调整`M`、`ef_construction`，特别是`ef_search`等参数 (parameter)，以便为您的向量 (vector)搜索应用在搜索准确性和性能效率之间达到期望的平衡。请记住，最佳参数取决于数据集，并需要根据您的具体应用需求进行仔细评估。

## 参考资料

- [hnswlib - High-performance nearest neighbor search library](https://github.com/nmslib/hnswlib) — nmslib (2024)
  `hnswlib`库的官方GitHub仓库，作为安装、API使用和本实践部分讨论的参数配置的主要文档。
- [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) — Erik Bernhardsson and contributors (2024)
  一个近似最近邻算法的比较基准测试框架，为包括HNSW在内的各种ANN方法提供性能和准确性权衡的实际见解。

---

[上一节](06-%E9%80%89%E6%8B%A9%E5%90%88%E9%80%82%E7%9A%84%E8%BF%91%E4%BC%BC%E6%9C%80%E8%BF%91%E9%82%BB%E7%AE%97%E6%B3%95%EF%BC%9A%E6%80%A7%E8%83%BD%E5%8F%96%E8%88%8D.md) · [下一节](../02-%E5%90%91%E9%87%8F%E6%90%9C%E7%B4%A2%E6%80%A7%E8%83%BD%E4%B8%8E%E6%95%88%E7%8E%87%E7%9A%84%E4%BC%98%E5%8C%96/01-%E9%87%8F%E5%8C%96%E6%8A%80%E6%9C%AF%EF%BC%9A%E6%A0%87%E9%87%8F%E9%87%8F%E5%8C%96%E4%B8%8E%E4%B9%98%E7%A7%AF%E9%87%8F%E5%8C%96.md)
