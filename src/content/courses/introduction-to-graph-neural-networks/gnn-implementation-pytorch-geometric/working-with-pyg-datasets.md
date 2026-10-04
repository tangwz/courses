---
course: "introduction-to-graph-neural-networks"
chapter: "gnn-implementation-pytorch-geometric"
lesson: "working-with-pyg-datasets"
sourceId: 7643
sourceUrl: "https://apxml.com/zh/courses/introduction-to-graph-neural-networks/chapter-5-gnn-implementation-pytorch-geometric/working-with-pyg-datasets"
title: "使用 PyG 数据集"
description: "了解如何使用 PyG 中的内置数据集，以及如何创建自定义图数据集进行训练。"
order: 3
plots: []
sourceHash: "0db332f8a49cd4717d3f7c99cc3930ec4e73cdc934b0e444ff9777300985840a"
sourceCorrections: []
---

许多图神经网络 (neural network)应用需要处理图集合或大型标准基准数据集。虽然 `Data` 对象可以有效地表示单个图，但为集合中的每个实例手动创建这些对象会非常繁琐且容易出错。PyTorch Geometric 通过其强大的 `torch_geometric.datasets` 模块简化了这一过程，该模块提供了对各种常见图数据集的便捷访问。

此模块实现了数据下载、处理和格式化的自动化，让你只需一行代码即可加载整个数据集。这使你能够专注于模型架构，而不必处理数据准备的细节。

### 访问内置数据集

PyTorch Geometric 预装了数十个基准数据集。这些数据集涵盖了从 `Cora` 和 `PubMed` 等引用网络、社交网络、生物信息学图到 `ModelNet10` 等 3D 网格数据集的各种类型。

让我们从加载一个适用于图分类的数据集开始，图分类的任务是预测每个完整图的属性。`TUDataset` 集合是此类数据的一个常用来源，包含 `ENZYMES`、`PROTEINS` 和 `IMDB-BINARY` 等数据集。我们可以加载 `ENZYMES` 数据集，它由 600 个表示蛋白质结构的图组成。

```python
from torch_geometric.datasets import TUDataset

# 'root' 目录是存储数据集的位置。
# 如果未找到数据集，PyG 会自动下载。
dataset = TUDataset(root='data/TUDataset', name='ENZYMES')

print(f'数据集: {dataset}')
print('====================')
print(f'图的数量: {len(dataset)}')
print(f'特征数量: {dataset.num_node_features}')
print(f'类别数量: {dataset.num_classes}')
```

当你第一次运行这段代码时，PyG 会自动将原始数据下载到 `data/TUDataset/ENZYMES/raw` 目录中，然后将其转换为处理后的格式并保存到 `data/TUDataset/ENZYMES/processed`。后续运行将直接加载处理后的数据。

输出显示了一些有用的属性：

```
数据集: ENZYMES(600)
====================
图的数量: 600
特征数量: 3
类别数量: 6
```

### 数据集对象的结构

PyG 的 `Dataset` 对象在功能上非常类似于标准的 Python 列表。你可以通过 `len()` 获取其长度，并可以使用标准列表索引访问单个图。每个元素都是一个 `Data` 对象，就像我们在前一部分中看到的那样。

> PyG 数据集可以被视为单个 `Data` 对象的集合，其中每个对象代表一个图。

让我们查看 `ENZYMES` 数据集中的第一个图：

```python
data = dataset[0]

print(data)
print('==============================================================')

# 获取关于第一个图的一些统计信息。
print(f'节点数量: {data.num_nodes}')
print(f'边数量: {data.num_edges}')
print(f'是否为无向图: {data.is_undirected()}')
```

这将产生类似于以下的输出：

```
Data(x=[37, 3], edge_index=[2, 168], y=[1], G_I=0)
==============================================================
节点数量: 37
边数量: 168
是否为无向图: True
```

第一个图有 37 个节点，每个节点有 3 个特征。其标签 `y` 表示它属于 6 个酶类别中的哪一个。由于 `Dataset` 对象是可迭代的，它们可以很好地与 PyG 的数据加载器配合进行小批量训练（我们稍后将对此进行讨论）。你还可以轻松地打乱数据集，这是训练前常见的步骤。

```python
# 就地打乱数据集
shuffled_dataset = dataset.shuffle()

print(shuffled_dataset[0])
```

### 用于节点分类的数据集

图分类涉及许多小图。相比之下，节点分类通常涉及单个大图。PyTorch Geometric 提供了一个不同的类 `Planetoid`，用于处理著名的引用网络数据集，如 `Cora`、`CiteSeer` 和 `PubMed`。

当你加载其中一个数据集时，`Dataset` 对象仅包含一个表示整个图的 `Data` 对象。

```python
from torch_geometric.datasets import Planetoid

# 加载 Cora 数据集
dataset = Planetoid(root='data/Planetoid', name='Cora')

print(f'数据集: {dataset}')
print('====================')
print(f'图的数量: {len(dataset)}')
print(f'特征数量: {dataset.num_node_features}')
print(f'类别数量: {dataset.num_classes}')

# 获取唯一的 Data 对象
data = dataset[0]
print(f'\n图对象: {data}')
```

输出证实了我们只有一个大图：

```
数据集: Cora()
====================
图的数量: 1
特征数量: 1433
类别数量: 7

图对象: Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])
```

`Cora` 图有 2,708 个节点（科学论文）和 10,556 条边（引用）。每篇论文都由一个 1433 维的特征向量 (vector)表示。这里的一个显著区别是存在 `train_mask`、`val_mask` 和 `test_mask` 属性。这些是布尔张量，分别指定哪些节点应用于训练、验证和测试。这是节点分类基准中常见的半监督转导学习设置的标准做法。

> 在转导设置中，模型在训练期间可以访问整个图的特征矩阵和结构，但只能访问训练节点的标签。掩码决定了哪些节点用于计算损失 (`train_mask`)、调节超参数 (parameter) (hyperparameter) (`val_mask`) 以及最终评估 (`test_mask`)。

### 创建自定义数据集

虽然内置数据集非常适合研究和学习，但你经常需要处理自己的数据。PyTorch Geometric 提供了基类来简化这一过程。对于可以完全装入内存的小型数据集，你可以继承 `torch_geometric.data.InMemoryDataset`。

要创建自己的 `InMemoryDataset`，你需要实现四个主要方法：

1. **`raw_file_names()`**: 返回一个原始文件名的字符串列表，这些文件必须存在于 `self.raw_dir` 中才能跳过下载。
2. **`processed_file_names()`**: 返回处理后的数据集文件名的字符串。PyG 会在 `self.processed_dir` 中寻找此文件，如果找到则跳过 `process` 步骤。
3. **`download()`**: 包含将原始数据下载到 `self.raw_dir` 的逻辑。
4. **`process()`**: 核心方法，你在此读取原始数据，创建 `Data` 对象列表，并使用 `torch.save(self.collate(data_list), self.processed_paths[0])` 将最终列表保存到磁盘。

这种结构化的方法保证了你的自定义数据集是可重用、可共享的，并且表现得像内置数据集一样。它处理了检查处理后数据的后台逻辑，因此你的预处理代码只运行一次。在数据正确加载后，你就可以定义 GNN 模型并开始学习了。

## 参考资料

- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/) — PyTorch Geometric Team (2023)
  PyTorch Geometric的官方文档，提供了数据集的详细API参考和使用示例，包括内置数据集和自定义数据集的创建。
- [TUDataset Collection](https://chrsmrrs.github.io/datasets/) — Christopher Morris, Nils M. Kriege, Franka Bause, Kristian Kersting, Petra Mutzel, Marion Neumann (2023)
  一个权威的资源，包含了广泛的基准图数据集集合，常用于图分类任务，包括ENZYMES数据集。
- [Semi-Supervised Classification with Graph Convolutional Networks](https://doi.org/10.48550/arXiv.1609.02907) — Thomas N. Kipf and Max Welling (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1609.02907](https://doi.org/10.48550/arXiv.1609.02907)
  提出了图卷积网络，并为Cora等引文网络数据集建立了常见的训练/验证/测试划分，这是节点分类基准的基础。
