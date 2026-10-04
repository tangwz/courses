---
course: "data-structures-algorithms-ml"
chapter: "hashing-for-ml"
lesson: "implementing-hashing-python"
sourceId: 5286
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-3-hashing-for-ml/implementing-hashing-python"
title: "在 Python 中实现基于哈希的数据结构"
description: "查看哈希表和集合如何在 Python 中被实现并有效运用在机器学习任务中。"
order: 5
plots: []
sourceHash: "492cf5224f16a9cf34e2da2ba1c5235fd3f14da68d1d65955834fca727da7eef"
sourceCorrections: []
---

Python 通过其字典 (`dict`) 和集合 (`set`) 类型，提供了高效的内置哈希表实现。了解它们的工作方式以及如何使用它们，对许多需要快速查找、插入和成员资格测试的机器学习 (machine learning)任务来说，具有实用价值。

### Python 的字典 (`dict`)

其本质上，Python 字典是一种哈希映射。它存储键值对，允许您非常迅速地检索与特定键关联的值，通常平均时间复杂度为 $O(1)$。

字典中的键必须是*可哈希的*，意思是它们必须具有在生命周期内不变的哈希值（与 `__hash__` 方法关联），并且可以与其他对象进行比较（与 `__eq__` 方法关联）。大多数内置的不可变类型，如字符串、数字（整型、浮点型）以及只包含可哈希类型的元组都是可哈希的。可变类型，如列表和字典，不能用作键，因为它们的内容（因此其哈希值）可能会改变。

这是一个基本示例：

```python
# 创建一个字典
feature_counts = {'word_a': 150, 'word_b': 85, 'word_c': 210}

# 按键访问值（平均时间 O(1)）
print(f"`word_b` 的计数: {feature_counts['word_b']}")

# 添加新的键值对（平均时间 O(1)）
feature_counts['word_d'] = 55

# 检查键是否存在（平均时间 O(1)）
print(f"'`word_a'` 存在吗？ {'word_a' in feature_counts}")
print(f"'`word_x'` 存在吗？ {'word_x' in feature_counts}")

# 更新值（平均时间 O(1)）
feature_counts['word_a'] += 10
print(f"`word_a` 的更新计数: {feature_counts['word_a']}")

# 获取项目数量
print(f"特征数量: {len(feature_counts)}")
```

**机器学习 (machine learning)中的字典：**

- **特征映射：** 字典非常适合创建从分类特征（如字符串）到数值表示（如整数索引）的映射，许多机器学习算法都需要这种映射。

  ```python
  categories = ['red', 'green', 'blue', 'green', 'red']
  cat_to_index = {category: i for i, category in enumerate(set(categories))}
  print(f"分类到索引映射: {cat_to_index}")
  # 输出: 分类到索引映射: {'green': 0, 'blue': 1, 'red': 2}

  indexed_features = [cat_to_index[cat] for cat in categories]
  print(f"索引特征: {indexed_features}")
  # 输出: 索引特征: [2, 0, 1, 0, 2]
  ```
- **计数频率：** 使用字典可以高效地计算文本数据中的词频（TF）或统计项目出现次数。Python 的 `collections.Counter` 是用于此目的的专用字典子类。

  ```python
  from collections import Counter

  document = "the quick brown fox jumps over the lazy dog the quick brown fox"
  words = document.split()
  word_counts = Counter(words)
  print(f"词频: {word_counts}")
  print(f"'the' 的频率: {word_counts['the']}")
  # 输出: 词频: Counter({'the': 3, 'quick': 2, 'brown': 2, 'fox': 2, 'jumps': 1, 'over': 1, 'lazy': 1, 'dog': 1})
  # 输出: 'the' 的频率: 3
  ```
- **缓存/记忆化：** 存储计算成本高的函数调用的结果，尤其是在输入可能重复的情况下。

### Python 的集合 (`set`)

集合是无序的、包含唯一可哈希元素的集合。与字典一样，它们内部使用哈希表，为添加元素以及（非常重要地）检查成员资格（测试元素是否存在于集合中）提供了平均 $O(1)$ 的时间复杂度。

```python
# 创建一个集合
unique_tags = {'ml', 'python', 'data_science', 'algorithms'}

# 添加一个元素（平均时间 O(1)）
unique_tags.add('hashing')

# 检查成员资格（平均时间 O(1)）
print(f"'python' 在集合中吗？ {'python' in unique_tags}")
print(f"'java' 在集合中吗？ {'java' in unique_tags}")

# 移除一个元素（平均时间 O(1)）
unique_tags.remove('algorithms') # 如果未找到则引发 KeyError
# unique_tags.discard('algorithms') # 如果未找到则不引发错误

print(f"当前标签: {unique_tags}")
```

**机器学习 (machine learning)中的集合：**

- **查找唯一项：** 可以轻松地从列表或其他可迭代对象中提取唯一元素。

  ```python
  feature_list = ['temperature', 'humidity', 'pressure', 'temperature', 'wind_speed']
  unique_features = set(feature_list)
  print(f"唯一特征: {unique_features}")
  # 输出: 唯一特征: {'pressure', 'humidity', 'wind_speed', 'temperature'}
  ```
- **快速成员资格测试：** 快速检查项目是否属于某个组，这对于过滤或验证非常有用。例如，检查用户 ID 是否属于已知的测试用户组。
- **集合操作：** 高效执行数学集合操作，如并集 (`|`)、交集 (`&`)、差集 (`-`) 和对称差集 (`^`)。这对于比较特征集、查找组间的共同元素等很有用。

  ```python
  set_A = {'apple', 'banana', 'cherry'}
  set_B = {'banana', 'cherry', 'date'}

  print(f"交集: {set_A & set_B}")   # 共同元素
  print(f"并集: {set_A | set_B}")        # 所有唯一元素
  print(f"差集 (A - B): {set_A - set_B}") # 存在于 A 但不存在于 B 的元素
  # 输出: 交集: {'cherry', 'banana'}
  # 输出: 并集: {'date', 'cherry', 'apple', 'banana'}
  # 输出: 差集 (A - B): {'apple'}
  ```

### 示例：简化的哈希表映射

虽然 Python 处理底层细节，但将映射可视化有助于理解。哈希函数将键转换为整数（哈希值），而取模运算符 (`%`) 将此哈希值映射到表中数组的一个索引。

> 键由哈希函数和取模运算处理，以确定它们在哈希表中的槽位索引。冲突（例如 'apple' 和 'orange' 都映射到索引 3）由 Python 内部处理。

### 与特征哈希的关系

字典明确存储从特征（键）到其索引或值的映射，而**特征哈希**（在“用于降维的特征哈希”一节中有详细介绍）直接使用哈希函数来确定输出特征向量 (vector)中的索引，而*不*存储原始特征键。

设想将单词映射到长度为 `D` 的固定大小向量中的索引。

- **字典方法：** `index = feature_map['word']`（需要构建和存储 `feature_map`）。
- **特征哈希方法：** `index = hash('word') % D`（不需要存储映射，多个特征可能映射到同一索引导致的冲突会被隐式处理）。

Python 的内置 `hash()` 函数可以用来实现一个简单版本，尽管库的实现（如 Scikit-learn 的 `HashingVectorizer`）经过优化，并处理符号哈希和其他细节。

```python
# 简化的特征哈希思想
feature_vector_size = 10
word1 = "machine"
word2 = "learning"
word3 = "algorithms" # 可能发生冲突

index1 = hash(word1) % feature_vector_size
index2 = hash(word2) % feature_vector_size
index3 = hash(word3) % feature_vector_size # 潜在冲突

print(f"'`machine'` 映射到索引: {abs(index1)}") # 使用 abs 获取非负索引
print(f"'`learning'` 映射到索引: {abs(index2)}")
print(f"'`algorithms'` 映射到索引: {abs(index3)}")

# 注意: hash() 的值可能为负，因此使用 abs()。
# 实际实现可能使用不同的哈希函数。
```

### 性能说明

虽然 `dict` 和 `set` 操作的平均时间复杂度为 $O(1)$，但需要记住：

1. **最坏情况：** 在（罕见的）最坏情况下，当许多键哈希到同一索引（高冲突）时，性能可能会下降到接近 $O(n)$，其中 $n$ 是元素数量。Python 的实现使用复杂的探测技术，在实践中能有效缓解此问题。
2. **大小调整：** 随着字典和集合的增长，底层表可能需要调整大小以保持较低的负载因子（元素与表大小的比率）并维持高性能。调整大小涉及创建一个更大的表并重新哈希/重新插入所有现有元素，这需要 $O(n)$ 的时间。这种情况不常发生，因此插入的*均摊*成本仍为 $O(1)$。
3. **内存：** 哈希表比简单的列表需要更多的内存来存储相同的元素，因为它们需要表数组本身的存储空间，而该数组通常只被部分填充。

Python 的 `dict` 和 `set` 是功能强大的工具，由高效的哈希表实现支持。恰当地运用它们可以实现快速的数据操作和查找，这在机器学习 (machine learning)流程中通常是很重要的步骤，涵盖从数据预处理、特征工程到某些算法的实现。它们也为理解高级哈希技术（如特征哈希和局部敏感哈希）提供了实践上的支持。

## 参考资料

- [Built-in Types - Dictionary and Set Types](https://docs.python.org/3/library/stdtypes.html) — Python Software Foundation (2023)
  提供 Python `dict` 和 `set` 数据结构的官方规范，包括其行为、可哈希性要求和时间复杂度。
- [Introduction to Algorithms](http://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: The MIT Press
  一本基础教材，涵盖哈希表、哈希函数和冲突解决技术的理论背景，对于理解基于哈希结构的工作表现至关重要。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: Chapter 3 (Dictionaries and Sets)
  详细说明了 Python 内置 `dict` 和 `set` 类型的内部实现、内存布局和工作表现特征，提供了对其效率的见解。
- [sklearn.feature_extraction.text.HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html) — scikit-learn developers (2024)
  scikit-learn HashingVectorizer 的官方文档，提供哈希在机器学习中用于特征工程以创建数值特征向量的实际应用示例。
