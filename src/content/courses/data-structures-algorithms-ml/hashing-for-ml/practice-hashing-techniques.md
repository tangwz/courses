---
course: "data-structures-algorithms-ml"
chapter: "hashing-for-ml"
lesson: "practice-hashing-techniques"
sourceId: 5292
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-3-hashing-for-ml/practice-hashing-techniques"
title: "实践：哈希技术的实现"
description: "动手实践应用特征哈希并了解与 LSH 相关的库或原理。"
order: 7
plots: []
sourceHash: "6cd3956cdd1009a292e4848d80b398331d898d8f7101839a57760bdf83b375c4"
sourceCorrections: []
---

使用 Python 进行的动手练习展示了哈希表、特征哈希和局部敏感哈希（LSH）的实际应用。这些练习旨在巩固对哈希在机器学习 (machine learning)任务中应用的理解，特别是在有效处理大型或高维数据方面。

### 实现基础哈希和冲突处理

首先，让我们回顾哈希表的核心思想和冲突处理。虽然 Python 的内置 `dict` 在内部处理此问题，但手动实现一个简单版本有助于说明其工作原理。我们将使用分离链接法，其中哈希表中的每个桶都保存一个哈希到相同索引的项目列表。

```python
import pprint # 用于美观打印

class SimpleHashTable:
    def __init__(self, size=10):
        self.size = size
        # 使用空列表初始化表，用于链接法
        self.table = [[] for _ in range(self.size)]

    def _hash_function(self, key):
        # 一个简单的模运算哈希函数
        return hash(key) % self.size

    def insert(self, key, value):
        hash_index = self._hash_function(key)
        bucket = self.table[hash_index]

        # 检查桶中是否已存在键
        for i, (existing_key, _) in enumerate(bucket):
            if existing_key == key:
                # 更新现有键的值
                bucket[i] = (key, value)
                return

        # 未找到键，添加新的键值对
        bucket.append((key, value))

    def search(self, key):
        hash_index = self._hash_function(key)
        bucket = self.table[hash_index]

        # 在桶的列表中查找键
        for existing_key, value in bucket:
            if existing_key == key:
                return value
        # 未找到键
        return None

# 示例用法
ht = SimpleHashTable(size=5)
ht.insert("apple", 1)
ht.insert("banana", 2)
ht.insert("cherry", 3)
ht.insert("date", 4) # 根据 hash() 输出可能发生冲突
ht.insert("elderberry", 5) # 可能发生冲突
ht.insert("fig", 6) # 可能发生冲突

print("哈希表结构:")
pprint.pprint(ht.table)

print("\n搜索结果:")
print(f"搜索 'banana': {ht.search('banana')}")
print(f"搜索 'grape': {ht.search('grape')}")
print(f"搜索 'date': {ht.search('date')}")

# 演示更新
ht.insert("apple", 100)
print(f"\n更新后搜索 'apple': {ht.search('apple')}")
print("\n更新后的哈希表结构:")
pprint.pprint(ht.table)
```

尝试使用不同的 `size` 值和输入键。注意冲突（同一内部列表中的多个项目）如何发生以及分离链接如何处理它们。当冲突频繁时，查找时间会从理想的 $O(1)$ 退化到最坏情况下的 $O(n)$（所有键都哈希到同一个桶），这强调了使用好的哈希函数和适当的表大小的重要性。

### 使用 Scikit-learn 应用特征哈希

特征哈希在将高维、稀疏的文本数据转换为固定大小的数值特征向量 (vector)方面特别有用。Scikit-learn 为此提供了 `HashingVectorizer`。让我们将其与标准的 `CountVectorizer` 进行比较。

```python
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
import numpy as np

# 示例文本数据（例如，来自用户评论或文档）
corpus = [
    'this is the first document great document',
    'this document is the second document',
    'and this is the third one',
    'is this the first document again maybe',
]

# 1. 使用 CountVectorizer（创建词汇表）
count_vectorizer = CountVectorizer()
count_features = count_vectorizer.fit_transform(corpus)
print("--- CountVectorizer ---")
print(f"词汇表大小: {len(count_vectorizer.vocabulary_)}")
# print(f"词汇表: {count_vectorizer.vocabulary_}") # 可能很大！
print(f"特征矩阵形状: {count_features.shape}")
# print(f"特征矩阵（稀疏）:\n{count_features.toarray()}") # 密集表示

# 2. 使用 HashingVectorizer（固定特征数量）
# 选择 n_features（输出空间的维度）
# 较小的 n_features 会增加冲突概率，但节省内存。
n_features_hashed = 10 # 远小于词汇表大小
hashing_vectorizer = HashingVectorizer(n_features=n_features_hashed, norm=None, alternate_sign=False)
hashed_features = hashing_vectorizer.fit_transform(corpus)

print("\n--- HashingVectorizer ---")
print(f"特征数量（固定）: {n_features_hashed}")
print(f"特征矩阵形状: {hashed_features.shape}")
# 转换为密集数组以检查。注意潜在冲突。
print(f"特征矩阵（稀疏 -> 密集）:\n{hashed_features.toarray()}")

# 请注意，当不同词语哈希到同一特征索引时，由于冲突，可能出现大于 1 的值。
# 选项 `alternate_sign=True`（默认值）通过随机分配 +1 或 -1 来帮助减轻冲突影响。
```

**观察结果：**

- `HashingVectorizer` 生成一个具有预定义列数（`n_features`）的矩阵，无论词汇表 (vocabulary)大小如何。这直接控制了内存使用。
- 它以流式方式处理文本，不构建显式词汇表，这使其适用于极大型数据集。
- 缺点是存在潜在的**哈希冲突**：不同的词语可能映射到相同的特征索引。`hashed_features.toarray()` 的输出可能在与冲突对应的位置显示大于 1 或负值（如果 `alternate_sign=True`）。这使得生成的特征比 `CountVectorizer` 的特征解释性更差。
- `n_features` 的选择涉及一个权衡：较小的值可以节省更多内存，但会增加冲突，可能损害模型性能。较大的值可以减少冲突，但会使用更多内存。

此练习说明了特征哈希如何有效编码稀疏数据，这是自然语言处理（NLP）和推荐系统中的常见情况。

### 使用 LSH 进行近似相似性搜索（MinHash 示例）

实现完整的 LSH 索引很复杂，但我们可以使用像 `datasketch` 这样的库来说明使用 MinHash 的核心思想，MinHash 适用于估计集合之间的 Jaccard 相似度。

首先，确保您已安装 `datasketch`：
`pip install datasketch`

现在，让我们为示例集合创建 MinHash 签名，并使用 LSH 索引进行近似最近邻搜索。

```python
from datasketch import MinHash, MinHashLSH

# 表示文档或用户偏好的示例集合
set1 = set(["cat", "dog", "mouse", "bird", "fish"])
set2 = set(["cat", "dog", "mouse", "lion", "tiger"]) # 类似于 set1
set3 = set(["apple", "banana", "orange", "grape", "kiwi"]) # 不相似
set4 = set(["cat", "dog", "fish", "hamster", "parrot"]) # 与 set1 中度相似

# 要索引的数据
data = {
    "set1": set1,
    "set2": set2,
    "set3": set3,
    "set4": set4
}

# MinHash 参数
# 更高的 num_perm 意味着更好的精度，但需要更多计算/存储
num_perm = 128

# 为每个集合创建 MinHash 对象
minhashes = {}
for key, item_set in data.items():
    m = MinHash(num_perm=num_perm)
    for element in item_set:
        m.update(element.encode('utf8'))
    minhashes[key] = m
    # 可选：计算精确的 Jaccard 相似度以供后续比较
    # print(f"'{key}' 的 MinHash 签名: {minhashes[key].digest()}") # 如有需要可检查签名

# 创建 LSH 索引
# Threshold 定义了候选对的 Jaccard 相似度阈值
lsh = MinHashLSH(threshold=0.5, num_perm=num_perm)

# 索引 MinHash 对象
print("\n--- 在 LSH 中索引集合 ---")
for key, m_hash in minhashes.items():
    lsh.insert(key, m_hash)
    print(f"已索引 '{key}'")

# 定义一个查询集合并创建其 MinHash
query_set = set(["cat", "dog", "mouse", "rabbit", "horse"])
query_key = "query"
m_query = MinHash(num_perm=num_perm)
for element in query_set:
    m_query.update(element.encode('utf8'))

# 查询 LSH 索引以获取近似最近邻
print(f"\n--- 查询 LSH 以获取与 '{query_key}' 相似的集合（阈值 > 0.5）---")
result = lsh.query(m_query)
print(f"找到近似邻居: {result}")

# 验证（可选）：计算精确的 Jaccard 相似度
print("\n--- 验证（精确 Jaccard 相似度）---")
print(f"查询集合: {query_set}")
for key in result:
    intersect_size = len(query_set.intersection(data[key]))
    union_size = len(query_set.union(data[key]))
    jaccard = intersect_size / union_size if union_size > 0 else 0
    print(f"  与 '{key}' ({data[key]}) 的 Jaccard 相似度: {jaccard:.4f}")

# 与 LSH 查询未找到的不相似集合进行比较
intersect_size_set3 = len(query_set.intersection(data["set3"]))
union_size_set3 = len(query_set.union(data["set3"]))
jaccard_set3 = intersect_size_set3 / union_size_set3 if union_size_set3 > 0 else 0
print(f"  与 'set3' ({data['set3']}) 的 Jaccard 相似度: {jaccard_set3:.4f}")
```

**观察结果：**

- MinHash 将集合转换为固定大小的签名（整数数组），使得两个签名在某个位置匹配的概率与原始集合的 Jaccard 相似度相关。
- LSH 索引将相似的 MinHash 签名分组到桶中。
- 查询涉及哈希查询签名，并仅检查相应桶中的项目，从而避免对所有索引项目进行完整比较。
- 结果（`result` 列表）包含其与查询集的估计 Jaccard 相似度高于指定 `threshold` 的集合的键。这是一种*近似*搜索；它可能会遗漏一些邻居（假阴性），或包含略低于阈值的项目（假阳性），但对于大型数据集而言，它比精确比较快得多。
- 调整 `num_perm` 和 `threshold` 可以控制准确性和查询速度之间的权衡。

这些实践练习说明了哈希技术如何实现以解决机器学习 (machine learning)中的特定问题，从高效特征表示到可伸缩相似性搜索。理解这些实现有助于在您的机器学习流程中选择合适的工具并有效调整其参数 (parameter)。

## 参考资料

- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: MIT Press
  涵盖哈希表、哈希函数和分离链等冲突解决方法的经典数据结构教材。
- [Feature Hashing for Large-Scale Multitask Learning](https://dl.acm.org/doi/abs/10.1145/2996914.2996923) — Kilian Weinberger, Anirban Dasgupta, John Attenberg, John Langford, Alex Smola (2009)
  Journal: Proceedings of the 26th International Conference on Machine Learning (ICML); Publisher: Association for Computing Machinery (ACM); Pages: 1136-1143; DOI: [10.1145/2996914.2996923](https://doi.org/10.1145/2996914.2996923)
  介绍了特征哈希概念的原始研究论文，该方法可高效表示高维数据，无需显式创建词汇表。
- [On the resemblance and containment of documents](https://ieeexplore.ieee.org/abstract/document/666900) — Andrei Z. Broder (1997)
  Journal: Proceedings of the Compression and Complexity of Sequences (SEQUENCES); Publisher: IEEE Computer Society; Pages: 21-29; DOI: [10.1109/SEQUEN.1997.666900](https://doi.org/10.1109/SEQUEN.1997.666900)
  引入 MinHash 的基础论文，这是一种估计集合间 Jaccard 相似度的技术，也是局部敏感哈希 (LSH) 的组成部分。
- [sklearn.feature_extraction.text.HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) — scikit-learn developers (2024)
  scikit-learn 官方文档，提供了 HashingVectorizer 在特征工程中的细节、参数和使用示例。
