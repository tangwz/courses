---
course: "data-structures-algorithms-ml"
chapter: "dsa-ml-foundations"
lesson: "python-structures-ml"
sourceId: 5234
sourceUrl: "https://apxml.com/zh/courses/data-structures-algorithms-ml/chapter-1-dsa-ml-foundations/python-structures-ml"
title: "Python的内置结构在机器学习工作流中的应用"
description: "回顾Python核心数据结构在机器学习任务中的用法和性能特点。"
order: 3
plots: ["plots/5234-0.json"]
sourceHash: "2ed5b4099c39d074d46a338021a18d1d037c28b2638cb54d0b21ca1cdc2ef760"
sourceCorrections: []
---

虽然NumPy和Pandas等专用库是机器学习 (machine learning)中处理大规模数值数据的主要工具，但Python的内置数据结构，即列表、字典和集合，在机器学习工作流的各个环节仍发挥着重要作用。了解它们的性能特点对于编写高效的辅助代码来说很重要。

### 列表：有序、可变序列

Python列表是多功能的有序项目集合。它们使用简单，常作为数据的初始容器，之后可能会转换为更专门的结构。

**机器学习 (machine learning)中的常见用途：**

- **收集结果：** 存储不同交叉验证折叠的评估指标。
- **序列数据：** 在自然语言处理任务中表示序列，例如句子（作为单词列表），在进一步处理之前。
- **简单特征向量 (vector)：** 保存单个数据点的特征值，特别是在初始数据加载或转换阶段。
- **中间存储：** 在多步骤预处理流程中临时保存数据。

**性能考量：**

- **按索引访问：** 访问特定索引处的元素（例如，`my_list[i]`）非常快，花费常数时间，$O(1)$。
- **追加：** 在末尾添加元素（`my_list.append(x)`）通常是高效的。它花费摊销常数时间，$O(1)$。摊销意味着虽然偶尔追加可能需要更长时间（由于内部重新分配大小），但多次追加操作的平均时间是常数。
- **插入/删除：** 在任意位置（非末尾）插入或删除元素需要移动后续元素。这花费线性时间，$O(n)$，其中$n$是插入/删除点之后的元素数量。如果在大列表上频繁执行，这可能成为一个瓶颈。
- **搜索：** 检查元素是否存在于列表中（`x in my_list`）或找到其索引（`my_list.index(x)`）需要扫描列表，平均花费线性时间，$O(n)$。

对于许多涉及大型数据集的机器学习任务，列表中插入、删除和搜索操作的$O(n)$开销使得它们不如为这些操作优化的结构适用，尤其是在性能要求高时。然而，对于较小集合或主要需要追加或索引访问的场景，列表是完全足够的。

### 字典（dict）：键值映射

字典存储键值对，根据键提供快速查找。它们内部使用哈希表实现。

**机器学习 (machine learning)中的常见用途：**

- **特征表示：** 将特征名称（字符串）映射到它们的值（数字、类别）。
- **稀疏数据：** 表示稀疏向量 (vector)或矩阵，其中只存储非零元素（例如，`{feature_index: value}`）。
- **参数 (parameter)存储：** 存储模型超参数 (hyperparameter)或配置。
- **计数频率：** 统计单词或类别的出现次数（例如，`{'word1': 10, 'word2': 5}`）。
- **映射ID：** 在分类特征和数值ID之间创建映射。

**性能考量：**

- **查找、插入、删除：** 字典的主要优点是其通过键访问、插入或删除元素的平均情况下的常数时间性能，$O(1)$。这依赖于底层哈希函数的效率。
- **最坏情况：** 在涉及许多哈希冲突（多个键映射到同一个内部槽位）的罕见情况下，这些操作可能退化为线性时间，$O(n)$。然而，Python的字典实现经过高度优化以最大程度地减少这种情况。
- **内存使用：** 由于哈希表结构的开销，相同数量的元素下，字典通常比列表消耗更多内存。

当您需要基于标识符或键进行快速查找时，字典是非常有价值的，这在机器学习流程中处理命名特征或映射操作时很常见。

### 集合：无序的唯一元素集合

集合类似于字典，但只存储键（唯一元素），不存储关联的值。它们通常也使用哈希表实现。

**机器学习 (machine learning)中的常见用途：**

- **成员测试：** 快速检查集合中是否存在某个项（例如，某个特定单词是否在停用词列表中？）。
- **查找唯一项：** 确定特征列或项目列表中的唯一值。
- **集合操作：** 执行高效的交集、并集或差集操作（例如，比较特征集合）。

**性能考量：**

- **成员测试、插入、删除：** 与字典类似，集合为添加元素、删除元素和检查成员资格（`x in my_set`）提供平均情况下的常数时间，$O(1)$。
- **最坏情况：** 类似于字典，在出现大量哈希冲突时，性能可能退化为$O(n)$。
- **无序：** 集合不保持任何元素顺序。

当唯一性和快速成员检查是主要要求时，集合表现出色。它们在大集合中检查成员资格时通常比列表更节省内存，因为它们避免重复并利用哈希。

### 性能比较：成员测试

我们来通过查找性能的差异进行说明。检查集合中是否存在某个元素（`element in collection`）是一种常见操作。



![成员测试性能（时间对比大小）](plots/5234-0.json)



> 检查成员资格（`element in collection`）的性能比较。请注意Y轴采用对数刻度。列表查找时间随大小线性增加（$O(n)$），而集合和字典键查找平均而言保持大致不变（$O(1)$）。实际时间取决于硬件和具体数据。

如图所示，对于搜索或成员测试，列表所需的时间随元素数量直接增长。相比之下，集合和字典平均而言无论大小如何，都保持快速、接近常数时间的查找。在机器学习 (machine learning)中处理大型词汇表 (vocabulary)、特征集或数据集时，这种差异变得非常明显。

虽然Python的内置结构便于使用，但必须考虑它们的性能特点。对于大型数组上的数值操作，我们通常转向NumPy；对于灵活的表格数据处理，Pandas是标准工具。接下来我们将讨论这些。

## 参考资料

- [5. Data Structures](https://docs.python.org/3/tutorial/datastructures.html) — Python Software Foundation (2024)
  Python 内置数据类型、操作及其性能要点的指南。
- [Data Structures and Algorithms in Python](https://www.wiley.com/en-us/Data+Structures+and+Algorithms+in+Python-p-9781118290279) — Michael T. Goodrich, Roberto Tamassia, Michael H. Goldwasser (2013)
  Publisher: John Wiley & Sons; DOI: [10.1002/9781118671415](https://doi.org/10.1002/9781118671415)
  一本详细讲解数据结构和算法的教科书，包含 Python 实现。
- [Fluent Python: Clear, Concise, and Effective Programming](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056355/) — Luciano Ramalho (2022)
  Publisher: O'Reilly Media; Pages: 1014
  对 Python 数据模型、内置类型及其性能方面的深入研究。
- [Introduction to Algorithms](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) — Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, Clifford Stein (2022)
  Publisher: MIT Press
  数据结构和算法理论基础的参考书，涵盖哈希和动态数组等内容。
