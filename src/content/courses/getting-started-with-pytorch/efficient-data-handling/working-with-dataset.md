---
course: "getting-started-with-pytorch"
chapter: "efficient-data-handling"
lesson: "working-with-dataset"
sourceId: 2177
sourceUrl: "https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/working-with-dataset"
title: "使用 `torch.utils.data.Dataset`"
description: "通过继承 `torch.utils.data.Dataset` 创建自定义数据集。"
order: 2
plots: []
sourceHash: "2ae9ff7815ce029d4b92bf44f089ad354257ff7b58b9f1b158b385165a419b5a"
sourceCorrections: []
---

高效加载和处理数据对训练深度学习 (deep learning)模型非常重要。PyTorch 提供了一种通过其 `torch.utils.data.Dataset` 抽象类来处理数据集的标准化方式。可以把 `Dataset` 看作一个约定：它定义了访问数据的标准接口，无论数据是存在内存中、磁盘上，还是需要即时生成。

### `Dataset` 抽象类

其核心是，`torch.utils.data.Dataset` 是一个表示数据集的抽象类。你在 PyTorch 中创建的任何自定义数据集都应该继承自这个类。为什么要使用这种结构？它确保了不同的数据集，无论是内置的还是自定义的，都能向其他 PyTorch 组件提供一致的 API，最值得一提的是 `DataLoader`，我们稍后会讲到。这种标准化简化了在相同训练代码中替换数据集或使用不同数据源的过程。

要创建自己的自定义数据集，你需要继承 `torch.utils.data.Dataset` 并重写两个必要的方法：

1. `__len__(self)`: 这个方法应该返回数据集中样本的总数。`DataLoader` 使用它来确定数据集的大小。
2. `__getitem__(self, idx)`: 这个方法负责根据给定索引 `idx` 从数据集中加载并返回一个样本。这是实际数据加载逻辑所在的地方（例如，读取图像文件、从 CSV 获取一行数据、访问列表中的元素）。`DataLoader` 会重复调用此方法来构建批次。

让我们用一个简单的例子来说明这一点。假设你的特征和对应的标签存储在 Python 列表或 NumPy 数组中。

```python
import torch
from torch.utils.data import Dataset
import numpy as np

class SimpleCustomDataset(Dataset):
    """一个带有特征和标签的简单数据集示例。"""

    def __init__(self, features, labels):
        """
        参数:
            features (列表或 np.array): 特征的列表或数组。
            labels (列表或 np.array): 标签的列表或数组。
        """
        # 基本检查：特征和标签必须长度相同
        assert len(features) == len(labels), "特征和标签的长度必须相同。"
        self.features = features
        self.labels = labels

    def __len__(self):
        """返回样本总数。"""
        return len(self.features)

    def __getitem__(self, idx):
        """
        生成一个数据样本。

        参数:
            idx (int): 元素的索引。

        返回:
            tuple: 给定索引对应的 (特征, 标签)。
        """
        # 获取给定索引的特征和标签
        feature = self.features[idx]
        label = self.labels[idx]

        # 通常，你会在Dataloader中将数据转换为 PyTorch 张量
        # 我们假设特征/标签可能还不是张量
        sample = (torch.tensor(feature, dtype=torch.float32),
                  torch.tensor(label, dtype=torch.long)) # 假设是分类标签

        return sample

# --- 示例用法 ---
# 样本数据（请替换为你的实际数据）
num_samples = 100
num_features = 10
features_data = np.random.randn(num_samples, num_features)
labels_data = np.random.randint(0, 5, size=num_samples) # 示例：5 个类别

# 创建自定义数据集实例
my_dataset = SimpleCustomDataset(features_data, labels_data)

# 访问数据集属性和元素
print(f"数据集大小: {len(my_dataset)}")

# 获取第一个样本
first_sample = my_dataset[0]
feature_sample, label_sample = first_sample
print(f"\n第一个样本特征:\n{feature_sample}")
print(f"第一个样本形状: {feature_sample.shape}")
print(f"第一个样本标签: {label_sample}")

# 获取第十个样本
tenth_sample = my_dataset[9]
print(f"\n第十个样本标签: {tenth_sample[1]}")
```

在此示例中：

- `__init__` 方法存储在实例化时传入的特征和标签数据。
- `__len__` 简单地返回特征列表的长度（这与标签列表的长度相同）。
- `__getitem__` 接受一个索引 `idx`，获取对应的特征和标签，将它们转换为 PyTorch 张量，并以元组形式返回。这种转换为张量的操作在 `__getitem__` 中很常见。

### 处理更复杂的场景

自定义 `Dataset` 的真正作用体现在处理那些不能直接在内存中获取的数据时。例如，你的图像文件路径和标签可能存储在一个 CSV 文件中。

```python
import torch
from torch.utils.data import Dataset
from PIL import Image # Python 图像库，用于图像加载
import pandas as pd
import os

class ImageFilelistDataset(Dataset):
    """用于从 CSV 文件加载图像路径和标签的数据集。"""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        参数:
            csv_file (字符串): 包含标注的 CSV 文件路径。
                               假设列有：'image_path', 'label'
            root_dir (字符串): 包含所有图像的目录。
            transform (可调用, 可选): 可选的数据变换，用于对样本进行处理。
                                           应用于样本。
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform # 我们稍后会讨论数据变换

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # 从 CSV 获取相对于 root_dir 的图像路径
        img_rel_path = self.annotations.iloc[idx, 0] # 假设第一列是路径
        img_full_path = os.path.join(self.root_dir, img_rel_path)

        # 使用 PIL 加载图像
        try:
            image = Image.open(img_full_path).convert('RGB') # 确保有 3 个通道
        except FileNotFoundError:
            print(f"错误：未在 {img_full_path} 找到图像")
            # 适当处理错误，例如返回 None 或抛出异常
            # 为简单起见，这里我们将返回 None，并依赖 DataLoader 的 collate_fn
            # 来处理它（或稍后过滤）。一个更好的方法
            # 可能是事先清理 CSV 文件。
            return None, None # 返回 None 值

        # 从 CSV 获取标签
        label = self.annotations.iloc[idx, 1] # 假设第二列是标签
        label = torch.tensor(int(label), dtype=torch.long)

        # 如果有，应用数据变换
        if self.transform:
            image = self.transform(image) # 数据变换通常会将 PIL 图像转换为张量

        # 如果没有提供将图像转换为张量的数据变换，则手动转换
        if not isinstance(image, torch.Tensor):
             # 如果没有应用其他数据变换，进行基本转换
             image = torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, label

# --- 示例用法（需要实际图像和 CSV）---
# 假设你拥有：
# 1. 文件夹 'data/images/' 包含图像文件（例如，cat1.jpg, dog1.png）
# 2. CSV 文件 'data/annotations.csv'，内容如下：
#    image_path,label
#    images/cat1.jpg,0
#    images/dog1.png,1
#    ...

# image_dataset = ImageFilelistDataset(csv_file='data/annotations.csv',
#                                      root_dir='data/')

# 访问方式类似：
# print(f"图像数据集大小: {len(image_dataset)}")
# if len(image_dataset) > 0:
#     img, lbl = image_dataset[0]
#     if img is not None:
#          print(f"第一个图像形状: {img.shape}") # 形状取决于数据变换
#          print(f"第一个图像标签: {lbl}")
```

在这个 `ImageFilelistDataset` 示例中：

- `__init__` 使用 pandas 读取 CSV 文件，并存储文件路径和根目录。它还接受一个可选的 `transform` 参数 (parameter)（我们很快会看到它的用法）。
- `__len__` 返回 CSV 文件中的行数。
- `__getitem__` 构建完整的图像路径，使用 PIL 加载图像，获取标签，应用任何指定的数据变换，确保图像是一个张量，并返回图像张量和标签张量。

请注意，`Dataset` 本身只定义了 *如何* 获取单个项目。它不会一次性将整个数据集加载到内存中（除非你的 `__init__` 明确这样做，但这对于大型数据集通常是避免的）。它也不处理批处理、打乱或并行加载。`DataLoader` 便是为此而生，它直接建立在 `Dataset` 提供的结构之上。通过实现 `__len__` 和 `__getitem__`，你为 `DataLoader` 高效访问数据样本提供了必要的结构。

## 参考资料

- [torch.utils.data](https://pytorch.org/docs/stable/data.html) — PyTorch Developers (2025)
  Publisher: PyTorch Foundation
  PyTorch数据工具的官方文档，提供了`Dataset`及相关类的API参考。
- [Data Loading and Processing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) — Sasank Chilamkurthy (2024)
  Publisher: PyTorch Foundation
  一个教程，通过实际代码示例引导读者构建自定义`Dataset`并使用`DataLoader`。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zachary C. Lipton, Mu Li, and Alex Smola (2024)
  Publisher: Cambridge University Press
  一本开源互动教材，提供了深度学习的概念基础和实际实现，包括数据加载技术。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  本书提供了使用`torch.utils.data.Dataset`和`DataLoader`处理数据的详细说明和示例。
