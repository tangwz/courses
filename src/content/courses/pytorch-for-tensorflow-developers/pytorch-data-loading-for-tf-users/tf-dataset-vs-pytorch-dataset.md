---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-data-loading-for-tf-users"
lesson: "tf-dataset-vs-pytorch-dataset"
sourceId: 5751
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/tf-dataset-vs-pytorch-dataset"
title: "数据结构：tf.data.Dataset 与 torch.utils.data.Dataset"
description: "比较 TensorFlow 和 PyTorch 中 `Dataset` 抽象在表示数据集合方面的作用。"
order: 1
plots: []
sourceHash: "9fd35c4e30626d470ec7280dc2ab84aa97c267f9b1487977e428cd34f0265fc4"
sourceCorrections: []
---

使用 TensorFlow 时，您习惯于利用 `tf.data.Dataset` 来表示一系列元素，例如张量或张量元组，并将 `map`、`batch` 和 `shuffle` 等变换直接应用于数据集对象，以构建输入管道。这会创建一个操作图，高效地处理数据。

PyTorch 则通过其 `torch.utils.data.Dataset` 类，提供了另一种同样有效的抽象来处理数据集。与直接在数据集对象上应用一系列变换方法不同，PyTorch 的 `Dataset` 是一个抽象类，您通常需要对其进行子类化以创建自己的自定义数据集。其核心思想是提供一种标准化的方式来访问单个数据样本。

任何继承自 `torch.utils.data.Dataset` 的 PyTorch 自定义数据集都必须实现两个重要方法：

1. `__len__(self)`: 此方法应返回数据集中样本的总数。PyTorch 的数据加载工具依赖此方法来确定数据集的规模。
2. `__getitem__(self, index)`: 此方法负责在给定 `index` 处获取单个数据样本（例如，图像及其对应的标签）。`index` 将是一个整数，其取值范围从 $0$ 到 `len(self) - 1`。

这种方式让您能够对数据如何按样本加载和处理进行细致的控制。虽然 `tf.data` 管道是通过在 `Dataset` 对象本身上链式调用操作来定义的，但在 PyTorch 中，数据变换通常封装在自定义 `Dataset` 的 `__getitem__` 方法中，或使用可组合的变换对象进行应用，这在后面讨论 `torchvision.transforms` 时会看到。

我们来看一个简单的例子。假设您有一个图像文件路径列表及其对应的标签。在 TensorFlow 中，您可能会使用 `tf.data.Dataset.from_tensor_slices((filepaths, labels))`，然后使用 `map` 函数来加载和预处理图像。

在 PyTorch 中，您会定义一个类：

```python
from torch.utils.data import Dataset
from PIL import Image # 用于图像加载，例如

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        参数:
            image_paths (list): 图像路径列表。
            labels (list): 对应的标签列表。
            transform (callable, optional): 可选的变换操作，将应用于
                单个样本。
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        img_path = self.image_paths[idx]
        # 示例: image = Image.open(img_path).convert("RGB")
        # 为了演示，我们假设图像已加载并预处理
        # 在实际场景中，您会从 img_path 加载图像
        # 并在此处应用任何必要的预处理。
        # 为简单起见，我们只返回路径和标签。
        image_data_placeholder = f"image_data_for_{img_path}" # 替换为实际图像加载代码
        label = self.labels[idx]

        sample = {"image": image_data_placeholder, "label": label}

        if self.transform:
            sample = self.transform(sample) # 假设变换操作能处理字典

        # 通常，__getitem__ 返回一个元组（特征，标签）
        # 或如上所示的字典。
        # 对于机器学习，特征和标签应为 PyTorch 张量。
        return sample # 或者 (torch.tensor(image_data), torch.tensor(label))
```

在这个 `CustomImageDataset` 中，`__init__` 存储数据源（文件路径和标签）以及任何变换操作。`__len__` 仅返回项目数量。`__getitem__` 则是获取和可能变换单个项目的逻辑所在。它接收一个索引 `idx`，获取对应的图像路径和标签，加载图像（为简洁起见省略了实际加载代码），并应用任何指定的变换操作。

PyTorch 支持两种主要类型的数据集：

- **映射风格数据集 (Map-style datasets)**：这是最常见的类型，是 `torch.utils.data.Dataset` 的子类，实现了 `__getitem__()` 和 `__len__()` 方法。它们表示从（整数）索引到数据样本的映射。上面示例中的 `CustomImageDataset` 就是一个映射风格数据集。
- **可迭代风格数据集 (Iterable-style datasets)**：这些是 `torch.utils.data.IterableDataset` 的子类，实现了 `__iter__()` 方法。它们适用于随机访问困难或数据流式传输的场景，因为它们表示数据样本上的可迭代对象。

对于大多数常见用例，特别是从 TensorFlow 转换过来时，如果您的数据可以方便地以列表或文件形式获取，那么映射风格数据集是直接需要考虑的对应部分。

下面的图表说明了 TensorFlow 和 PyTorch 中数据集结构方法的差异：

> TensorFlow 和 PyTorch 中数据集抽象的高层次对比。TensorFlow 的 `tf.data.Dataset` 涉及链式操作以形成数据管道。PyTorch 的 `torch.utils.data.Dataset` 通常是一个自定义类，定义了索引数据访问方式，然后由 `DataLoader` 使用。

本质上，`tf.data.Dataset` 更多是关于将一个数据转换管道定义为一个单一的、有状态的对象，该对象生成经过处理的（通常是批次的）数据。相比之下，PyTorch 的 `torch.utils.data.Dataset` 主要关注提供对单个原始或轻度处理数据样本的索引访问。批处理、随机化和并行数据加载的任务随后被委派给 `torch.utils.data.DataLoader`，我们将在下一节中讲解。这种关注点分离带来了灵活性：`Dataset` 定义了数据是什么以及如何获取一个项目，而 `DataLoader` 则定义了如何以批次形式迭代多个项目。

## 参考资料

- [Data loading and preprocessing (\`torch.utils.data\`)](https://pytorch.org/docs/stable/data.html) — PyTorch Core Team (2025)
  `torch.utils.data` 模块（包括 `Dataset` 和 `DataLoader`）的 PyTorch 官方文档。
- [\`tf.data\`: Build TensorFlow input pipelines](https://www.tensorflow.org/guide/data) — Google (2024)
  Publisher: Google
  `tf.data` 构建高效输入管道的 TensorFlow 官方指南。
- [Chapter 5: The Mechanics of Learning: From Goats to \`DataLoader\`](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications; Pages: 129-160
  提供了 PyTorch 数据加载的解释，包括 `Dataset` 和 `DataLoader`。
- [Chapter 13: Loading and Preprocessing Data with the Data API](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQF9FMwBIOVvB4x_4Ry8TMCSLRUa7QbIWmCMNkZwEszyKAx5hmbP4wjUz8ZvLiit-Z8DbZfI_gVvhHMMt_TF75StWX-A1Elfv4uxMsRviB3030dUKcA-xqTKy8izggFuUk8p8lqDA4QFmJ8Afg6lVTLKZvJenWe8Aa_rz7m0g9o7u-nIWH9CEFR2I7oCkzW7p9bP5jo9tCHB8wCLYGQ=) — Aurélien Géron (2022)
  Publisher: O'Reilly Media; Pages: 439-482
  涵盖了 `tf.data` API，包括为各种数据类型创建管道和应用转换。
