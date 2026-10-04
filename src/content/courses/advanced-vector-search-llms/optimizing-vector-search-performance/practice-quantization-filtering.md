---
course: "advanced-vector-search-llms"
chapter: "optimizing-vector-search-performance"
lesson: "practice-quantization-filtering"
sourceId: 3870
sourceUrl: "https://apxml.com/zh/courses/advanced-vector-search-llms/chapter-2-optimizing-vector-search-performance/practice-quantization-filtering"
title: "实践：应用量化与过滤"
description: "动手实践，应用量化和过滤技术以提升搜索性能。"
order: 8
plots: ["plots/3870-0.json"]
sourceHash: "a2e5607e976380c1733c1365f5d3cf3e96165637a8b6a6b64a6aef300660b660"
sourceCorrections: []
---

产品量化 (quantization)（PQ）等量化方法能够压缩向量 (vector)以节省内存并加速搜索。元数据过滤则可缩小搜索范围，提升相关性并可能加快速度。本练习提供在模拟环境中动手应用这些方法的实践。目标是展示这些优化方法如何相互作用并影响搜索结果。

我们假设你拥有一个包含 `numpy`（用于数值运算）和 `faiss` 等向量搜索库的开发环境。尽管具体代码在不同库之间有所差异，但这里展示的原理普遍适用。

### 场景设置

设想我们正在为一个技术文章语料库构建一个语义搜索系统。每篇文章都有一个代表其内容的向量 (vector)嵌入 (embedding)，以及相关的元数据，比如出版年份和主分类（例如，“人工智能”、“数据科学”、“软件工程”）。

我们的目标是找到与查询向量语义相似的文章，但仅限于特定分类且在某一年份之后出版的。我们需要这种搜索既快速又节省内存。

让我们模拟一些数据：

```python
import numpy as np
import faiss
import time

# --- 配置 ---
d = 128      # 向量维度
n_total = 100000 # 索引中的向量总数
n_query = 10     # 查询向量数
k = 5        # 要检索的最近邻数量

# --- 生成示例数据 ---
print("正在生成示例数据...")
np.random.seed(42)
db_vectors = np.random.random((n_total, d)).astype('float32')
query_vectors = np.random.random((n_query, d)).astype('float32')

# 生成随机元数据（示例）
categories = ['AI', 'Data Science', 'Software Engineering', 'Cloud Computing']
years = [2020, 2021, 2022, 2023]
metadata = [
    {
        'id': i,
        'category': np.random.choice(categories),
        'year': np.random.choice(years)
    } for i in range(n_total)
]

# 创建一个映射，通过ID快速查找元数据
metadata_map = {item['id']: item for item in metadata}

print(f"生成了 {n_total} 个维度为 {d} 的向量。")
print(f"示例元数据[0]: {metadata[0]}")

# --- 目标过滤条件（示例）---
target_category = 'AI'
target_min_year = 2022
```

### 基线：穷举搜索（无优化）

首先，我们使用精确搜索索引（Faiss 中的 `IndexFlatL2`）来建立一个基线。这提供了完美的召回率，但对于大型数据集可能速度较慢且占用大量内存。

```python
# --- 基线：平面索引（精确搜索）---
print("\n--- 基线：平面索引 ---")
index_flat = faiss.IndexFlatL2(d)
index_flat.add(db_vectors)

start_time = time.time()
D_flat, I_flat = index_flat.search(query_vectors, k)
end_time = time.time()

print(f"平面索引搜索时间: {end_time - start_time:.4f} 秒")
print(f"示例结果（索引）: {I_flat[0]}")
# 注意：此基线尚未包含过滤。
# 过滤将发生在检索到这些 K 个结果之后（后置过滤）。
```

### 应用产品量化 (quantization)（PQ）

现在，让我们应用PQ来压缩向量 (vector)并加快搜索速度。我们将使用 `IndexIVFPQ` 结构，它结合了倒排文件（IVF）进行分区，以及PQ用于每个分区内的压缩。

`IndexIVFPQ` 的重要参数 (parameter)：

- `nlist`：IVF 分区的数量（Voronoi 单元）。一个常用的起始值是 `sqrt(n_total)`。
- `M`：PQ 的子量化器数量。它将向量空间划分。必须是 `d` 的一个因子。
- `nbits`：每个子量化器编码的位数。通常是 8（每个子空间产生 256 个质心）。

```python
# --- 应用量化：IVFPQ ---
print("\n--- 优化1：IVFPQ 索引 ---")

# 参数
nlist = int(np.sqrt(n_total)) # IVF 的聚类数量
M = 8                        # PQ 的子量化器数量 (d=128 可被 M=8 整除)
nbits = 8                    # 每个子向量索引的位数

# 构建量化器（用于 IVF 聚类）和索引
quantizer = faiss.IndexFlatL2(d)
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, M, nbits)

# 在数据库向量上训练索引
print(f"正在训练 IVFPQ 索引 (nlist={nlist}, M={M}, nbits={nbits})...")
# 训练需要有代表性的样本，如果可行，通常就是数据本身
# 对于非常大的数据集，请使用子集进行训练。
index_ivfpq.train(db_vectors)
print("训练完成。")

# 向索引添加向量
print("正在向 IVFPQ 索引添加向量...")
index_ivfpq.add(db_vectors)
print("添加完成。")

# --- 使用 IVFPQ 执行搜索 ---
# 'nprobe' 控制要搜索的 IVF 分区数量。值越高 = 越精确，越慢。
index_ivfpq.nprobe = 10 # 示例值，需要调整

start_time = time.time()
D_ivfpq, I_ivfpq = index_ivfpq.search(query_vectors, k)
end_time = time.time()

print(f"IVFPQ 搜索时间 (nprobe={index_ivfpq.nprobe}): {end_time - start_time:.4f} 秒")
print(f"示例结果（索引）: {I_ivfpq[0]}")

# 计算召回率（近似值，与第一个查询的平面索引结果进行比较）
# 注意：准确的召回率需要来自*整个*数据集的真实标签，
# 这计算成本很高。这是一个简化检查。
common_results = np.intersect1d(I_flat[0], I_ivfpq[0])
recall_approx = len(common_results) / k
print(f"第一个查询的近似召回率@{k} (对比平面索引): {recall_approx:.2f}")
```

你应该会看到相对于平面索引有显著的速度提升，但可能会牺牲一定的召回率。调整 `nprobe` 可以让你在速度和准确性之间进行权衡。

### 实施过滤策略

现在，让我们加入我们的元数据过滤器：`category == 'AI'` 和 `year >= 2022`。

#### 策略1：后置过滤

这是最简单的方法。我们首先执行ANN搜索（可能检索超过 `k` 个结果以弥补过滤损失），然后根据元数据过滤结果。

```python
# --- 优化2a：IVFPQ + 后置过滤 ---
print("\n--- 优化2a：IVFPQ + 后置过滤 ---")

k_oversample = k * 5 # 最初检索更多结果以应对过滤损失

index_ivfpq.nprobe = 10 # 如果需要，重置 nprobe

start_time_search = time.time()
D_oversample, I_oversample = index_ivfpq.search(query_vectors, k_oversample)
end_time_search = time.time()

# --- 过滤步骤 ---
start_time_filter = time.time()
final_results_post = []
final_distances_post = []

for i in range(n_query): # 对于每个查询
    filtered_indices = []
    filtered_distances = []
    for j in range(k_oversample):
        doc_id = I_oversample[i, j]
        if doc_id < 0: continue # Faiss 可能会返回 -1 表示无效结果

        meta = metadata_map.get(doc_id)
        if meta and meta['category'] == target_category and meta['year'] >= target_min_year:
            filtered_indices.append(doc_id)
            filtered_distances.append(D_oversample[i, j])
            if len(filtered_indices) == k: # 一旦获得足够结果就停止
                 break
    final_results_post.append(filtered_indices)
    final_distances_post.append(filtered_distances)

end_time_filter = time.time()

total_time = (end_time_search - start_time_search) + (end_time_filter - start_time_filter)
print(f"IVFPQ 搜索时间: {end_time_search - start_time_search:.4f} 秒")
print(f"后置过滤时间: {end_time_filter - start_time_filter:.4f} 秒")
print(f"总时间（搜索 + 后置过滤）: {total_time:.4f} 秒")
print(f"示例过滤结果（索引）: {final_results_post[0]}")
```

后置过滤直观易懂，但可能通过获取和检查那些本可以更早排除的候选项的元数据而执行不必要的工作。开销随 `k_oversample` 的增加而增加。

#### 策略2：预过滤（模拟）

高效的预过滤通常需要与索引结构本身或底层数据库集成。像 Faiss 这样的向量 (vector)搜索库提供了诸如 `IDSelector` 这样的机制，允许只搜索指定ID的子集。我们可以通过首先识别符合我们元数据标准的文档ID，然后指示搜索库（如果可能）仅考虑这些ID来模拟预过滤的*效果*。

```python
# --- 优化2b：IVFPQ + 预过滤（通过 IDSelector 模拟）---
print("\n--- 优化2b：IVFPQ + 预过滤（模拟）---")

start_time_id_select = time.time()
# 步骤1：识别符合元数据过滤条件的ID
eligible_ids_list = [
    item['id'] for item in metadata
    if item['category'] == target_category and item['year'] >= target_min_year
]
eligible_ids = np.array(eligible_ids_list, dtype='int64') # Faiss 通常需要 int64 类型的 ID

# 如果符合条件的 ID 数量非常少，精确搜索可能会更快
if len(eligible_ids) == 0:
    print("没有文档符合过滤条件。")
    # 处理空结果情况
elif len(eligible_ids) < 1000: # 阈值取决于系统/数据
     print(f"找到 {len(eligible_ids)} 个匹配 ID。可能在子集上使用精确搜索。")
     # 为简洁起见，省略了子集上的精确搜索实现
else:
    print(f"找到 {len(eligible_ids)} 个匹配 ID。使用 IDSelector 和 IVFPQ。")

    # 步骤2：使用 Faiss IDSelector
    id_selector = faiss.IDSelectorArray(eligible_ids)

    # 步骤3：使用选择器执行搜索
    # 注意：`search` 函数需要传入上下文/参数，包括选择器
    # Faiss 语法: index.search(query, k, params=SearchParameters)，其中 params 包含选择器

    # 简化调用结构用于说明（实际 API 可能略有不同）
    search_params = faiss.SearchParametersIVF(sel=id_selector, nprobe=index_ivfpq.nprobe)

    end_time_id_select = time.time()

    start_time_search = time.time()
    # 实际搜索调用可能需要根据 Faiss 版本/特定索引类型进行调整
    # 这里我们使用通用搜索方法，假设它遵循 params 对象
    D_pre, I_pre = index_ivfpq.search(query_vectors, k, params=search_params)
    end_time_search = time.time()

    total_time = (end_time_id_select - start_time_id_select) + (end_time_search - start_time_search)
    print(f"ID 选择时间: {end_time_id_select - start_time_id_select:.4f} 秒")
    print(f"IVFPQ 搜索时间（带预过滤）: {end_time_search - start_time_search:.4f} 秒")
    print(f"总时间（ID 选择 + 搜索）: {total_time:.4f} 秒")
    print(f"示例过滤结果（索引）: {I_pre[0]}")

# 警告：预过滤的性能优势很大程度上取决于过滤器的选择性
# 以及索引内部 IDSelector 实现的效率。
# 如果过滤器选择了数据集的很大一部分，那么收益可能很小。
```

预过滤，当在索引内部高效实现时，可以通过避免对不相关向量的计算来显著减少搜索时间。然而，这需要索引结构或数据库有效支持此功能。识别符合条件的ID的初始成本也需要考虑。

### 性能对比

让我们直观地展示我们观察到的权衡。我们将绘制不同方法的近似延迟与召回率（或其代表）图。



![延迟与召回率的权衡（示意图）](plots/3870-0.json)



> 不同优化策略的示意性比较。数值为示例，实际性能很大程度上取决于数据、硬件和参数 (parameter)调整。预过滤假设高效实现。

此图表展示了典型的行为：

- **平面索引：** 召回率最高（按定义为 1.0），延迟最高。
- **IVFPQ：** 延迟较低但召回率降低。增加 `nprobe` 可提高召回率但会增加延迟。
- **后置过滤：** 在基础IVFPQ搜索中为过滤步骤增加了少量延迟开销。召回率取决于基础搜索召回率和 `k_oversample`。
- **预过滤：** 如果过滤器具有选择性且实现高效，则可能为与相应IVFPQ搜索相同的召回率提供最低延迟。

### 总结

本次实践展示了量化 (quantization)（特别是IVFPQ）如何显著减少与穷举搜索相比的搜索延迟，尽管会带来由 `nprobe` 等参数 (parameter)控制的准确性权衡。我们还实现并比较了后置过滤和（模拟的）预过滤。后置过滤简单但可能效率低下，而预过滤，当得到有效支持时，可以提前修剪搜索空间，从而带来更好的性能，尤其是在使用选择性过滤器时。

选择正确的组合需要理解你的应用程序对速度、内存使用和准确性的具体需求，并根据实证评估仔细调整 `nprobe`、`M`、`nbits` 和 `k_oversample` 等参数。这里讨论的技术是构建依赖向量 (vector)搜索的响应迅速且可扩展的大语言模型（LLM）应用的核心部分。

## 参考资料

- [Product Quantization for Nearest Neighbor Search](https://inria.hal.science/inria-00514462/file/) — Hervé Jégou, Matthijs Douze, Cordelia Schmid (2011)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 33; Pages: 117-128; DOI: [10.1109/TPAMI.2010.57](https://doi.org/10.1109/TPAMI.2010.57)
  本文介绍了乘积量化，一种压缩高维向量并加速近似最近邻搜索的基础技术。
- [Filtering with Approximate Nearest Neighbor Search](https://doi.org/10.1016/j.bdr.2018.06.002) — Xiaohui Li, Jihong Guan, Yongdong Cheng (2018)
  Journal: Big Data Research; Publisher: Elsevier; Volume: 13; Pages: 1-9; DOI: [10.1016/j.bdr.2018.06.002](https://doi.org/10.1016/j.bdr.2018.06.002)
  本文探讨了将过滤与近似最近邻搜索相结合的策略，涵盖了预过滤和后过滤技术，以提高效率和相关性。
