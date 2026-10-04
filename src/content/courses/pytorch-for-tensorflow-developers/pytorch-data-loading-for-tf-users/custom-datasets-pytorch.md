---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-data-loading-for-tf-users"
lesson: "custom-datasets-pytorch"
sourceId: 5754
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/custom-datasets-pytorch"
title: "在 PyTorch 中实现自定义数据集"
description: "了解如何在 PyTorch 中创建自定义 `Dataset` 类以处理特殊数据源。"
order: 4
plots: []
sourceHash: "2d8a4c948bc4274f12bc04603bea7284a8a1909681380eb97bcc9df1bd92a27a"
sourceCorrections: []
---

PyTorch 提供便捷的预构建数据集，尤其在 `torchvision.datasets` 中，但许多应用需要您处理自定义格式的数据或具有特定加载逻辑的数据。这时，PyTorch 的 `torch.utils.data.Dataset` 类就变得不可或缺。它提供一种灵活、符合 Python 习惯的方式，逐项定义数据如何访问和处理。对于习惯使用 `tf.data.Dataset.from_generator` 或在 `map` 操作中编写自定义解析函数的 TensorFlow 用户，PyTorch 的方法是创建一个继承自 `torch.utils.data.Dataset` 的 Python 类。

### 自定义数据集的构成

其核心是，您在 PyTorch 中创建的任何自定义数据集都将是一个继承 `torch.utils.data.Dataset` 的 Python 类。这个父类是一个抽象类，要创建一个可用数据集，您需要实现两个特殊方法：

1. `__len__(self)`: 此方法必须返回数据集中的样本总数。`DataLoader`（我们之前讨论过用于批处理和迭代）依赖此方法来确定数据集的范围。
2. `__getitem__(self, idx)`: 此方法负责根据给定索引 `idx` 获取单个数据样本。索引范围从 `0` 到 `len(self) - 1`。您通常在此方法中从文件中加载数据，执行必要的预处理，并应用数据转换。`DataLoader` 在构建批次时调用此方法来获取单个样本。

可选地，但几乎总是，您还会实现 `__init__(self, ...)` 方法。此构造函数用于执行任何一次性设置，例如：

- 加载标注或元数据（例如，从 CSV 文件）。
- 构建数据样本的文件路径列表。
- 如果可行，将小型数据集完全预加载到内存中。
- 初始化将应用于数据的任何数据转换。

让我们通过一个常见场景来说明这一点：创建一个数据集，用于处理按类标签命名的子目录结构中存储的图像。例如：

```
data_root/
├── class_A/
│   ├── image001.jpg
│   ├── image002.png
│   └── ...
├── class_B/
│   ├── image101.jpeg
│   ├── image102.gif
│   └── ...
└── class_C/
    ├── image201.jpg
    └── ...
```

### 构建自定义图像数据集

我们将创建一个 `CustomImageDataset` 类来处理这种结构。我们需要 Pillow (PIL) 或 OpenCV 等库来加载图像。在此示例中，我们假设 Pillow 已安装（`pip install Pillow`）。

```python
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 包含所有图像的目录，按类别组织。
            transform (callable, optional): 可选的，应用于样本的数据转换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # 存储 (图像路径, 类别索引) 元组的列表
        self.classes = sorted(os.listdir(root_dir)) # 从文件夹名获取类别名称
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path): # 确保是文件
                    item = (img_path, self.class_to_idx[class_name])
                    self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB') # 确保图像是 RGB 格式
        except IOError:
            print(f"警告：无法加载图像 {img_path}。跳过。")
            # 返回一个虚拟样本或引发错误，或者尝试下一个
            # 为简单起见，如果此操作失败，我们递归地尝试获取下一个样本。
            # 一种解决方案可能涉及在 __init__ 中过滤掉损坏的文件
            # 或者返回一个占位符。
            return self.__getitem__((idx + 1) % len(self.samples))


        if self.transform:
            image = self.transform(image)
        
        # 将标签转换为张量
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor
```

让我们分解这个 `CustomImageDataset`：

- **`__init__(self, root_dir, transform=None)`**:

  - 它接受 `root_dir`（例如，`'data_root/'`）和一个可选的 `transform` 参数 (parameter)。
  - 它扫描 `root_dir` 以查找类别子目录。`self.classes` 存储这些子目录的名称（例如，`['class_A', 'class_B', 'class_C']`）。
  - `self.class_to_idx` 创建从类别名称到整数索引的映射（例如，`{'class_A': 0, 'class_B': 1, 'class_C': 2}`）。这是分类任务的标准做法。
  - 然后它遍历每个类别目录及其中的每个图像文件，用 `(图像路径, 类别索引)` 元组填充 `self.samples`。这个列表实际上成为了我们数据集的索引。
- **`__len__(self)`**:

  - 这很简单。它只是返回 `self.samples` 中收集的图像路径总数。
- **`__getitem__(self, idx)`**:

  - 这个方法是主要处理部分。给定一个索引 `idx`，它从 `self.samples` 中获取对应的 `img_path` 和 `label`。
  - 它使用 `Image.open(img_path).convert('RGB')` 打开图像。转换为 'RGB' 是处理不同通道数图像（如灰度或 RGBA）的良好做法。
  - 包含一个基本的 `try-except` 块来处理 `IOError`，这可能在图像文件损坏时发生。更完善的处理可能涉及在 `__init__` 期间预过滤损坏的文件。
  - 如果在实例化期间提供了 `transform`（例如，一系列 `torchvision.transforms`），它会应用于加载的图像。您可以在此放置调整大小、裁剪、转换为张量和归一化 (normalization)等操作。
  - 最后，它返回（可能经过转换的）图像及其标签作为 `torch.tensor`。标签被转换为 `torch.long` 张量，这通常是像 `torch.nn.CrossEntropyLoss` 这样的损失函数 (loss function)所期望的。

### 使用您的自定义数据集

一旦定义，您就可以实例化您的 `CustomImageDataset` 并用 `DataLoader` 包装它，就像任何内置数据集一样：

```python
# 定义数据转换
# 例如：调整大小、转换为张量和归一化
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 实例化数据集
image_dataset = CustomImageDataset(root_dir='path/to/your/data_root', transform=data_transforms)

# 创建 DataLoader
# 这将处理批处理、混洗和并行数据加载
batch_size = 32
data_loader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 现在您可以在训练循环中遍历 data_loader
# for inputs, labels in data_loader:
#     # 您的训练代码在这里
#     # inputs 将是图像张量的批次
#     # labels 将是标签张量的批次
#     pass
```

在这个设置中，`data_transforms` 是一个常见数据转换流程的示例，来自 `torchvision.transforms`。当 `data_loader` 请求一个批次时，它会多次内部调用 `image_dataset.__getitem__(idx)`，应用这些转换，然后将单个样本（图像张量和标签张量）整理成一个批次。

### 与 TensorFlow 的 `tf.data` 相关

如果您来自 TensorFlow，您可能通过以下方式实现了类似的自定义数据加载：

- 为特定图像文件夹结构使用 `tf.keras.utils.image_dataset_from_directory`。
- 编写一个 Python 生成器函数，然后用 `tf.data.Dataset.from_generator` 包装它。
- 使用 `tf.data.Dataset.list_files` 获取文件路径，然后使用 `.map()` 方法和自定义函数（通常是 TensorFlow 操作或 `tf.py_function`）来加载和预处理每个文件。

PyTorch 的 `Dataset` 类提供一种更直接、面向对象的方法。您将识别、加载和转换单个数据项的所有逻辑封装在一个 Python 类中。这通常感觉与标准 Python 编程习惯更紧密，并且更容易调试，因为您在 `__getitem__` 中处理的是常规 Python 对象和控制流。`DataLoader` 然后有效地在多个工作进程中并行执行 `__getitem__`。

以下图表显示了 `DataLoader` 使用您的自定义 `Dataset` 的一般流程：

> 这个图表说明了 `DataLoader` 如何与您的自定义 `Dataset` 类（特别是其 `__getitem__` 方法）互动，以在批处理前获取和处理单个数据样本。

### 自定义数据集的良好做法

- **按需加载**: 在 `__getitem__` 内部执行 I/O 操作（如从磁盘读取文件），而不是在 `__init__` 中将所有内容加载到内存中，特别是对于大型数据集。这可以降低内存使用量并加快启动时间。`__init__` 理想情况下应侧重于收集元数据，例如文件路径和标签。
- **将数据转换作为参数 (parameter)**: 将数据转换流程（例如 `torchvision.transforms.Compose` 对象）作为参数传递给数据集的 `__init__` 方法，并在 `__getitem__` 中应用它们。这使得您的数据集灵活，并允许用户轻松尝试不同的数据增强和预处理策略。
- **返回张量**: 确保 `__getitem__` 返回 `torch.Tensor` 对象，因为这是 PyTorch 模型和损失函数 (loss function)所期望的。
- **处理损坏的数据**: 在 `__getitem__` 中实现基本的错误处理，以应对文件损坏等情况。您可以选择跳过样本、返回占位符或记录警告。在 `__init__` 期间预过滤有问题的数据也可能有效。
- **`__getitem__` 的效率**: 保持 `__getitem__` 中的操作尽可能高效。由于它为每个样本调用，所以此处的任何瓶颈都会明显减慢您的训练速度，即使 `DataLoader` 中有多个 `num_workers`。

通过熟练创建自定义 `Dataset`，您可以对数据加载流程进行精细控制，使得您能够在 PyTorch 项目中使用几乎任何数据源和结构。这与 `tf.data` 中有时更不透明或特定于框架的操作形成对比，提供了一条 Python 原生数据准备路径。

## 参考资料

- [torch.utils.data.Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) — PyTorch Core Team (2025)
  Publisher: PyTorch
  PyTorch 基础 Dataset 类的官方 API 参考，详细说明了自定义数据加载以及与 DataLoader 交互所需的方法。
- [torch.utils.data.DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) — PyTorch Core Team (2025)
  Publisher: PyTorch Foundation
  DataLoader 的官方文档，它处理批处理、洗牌和多进程数据加载，与自定义 Dataset 协同工作。
- [Learn the Basics - PyTorch Tutorials: Custom Datasets, Transforms and Models](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) — PyTorch Core Team (2024)
  Publisher: PyTorch Foundation
  PyTorch 官方教程，提供一个实用的指南，通过示例说明如何实现自定义数据集、应用转换以及使用 DataLoader。
