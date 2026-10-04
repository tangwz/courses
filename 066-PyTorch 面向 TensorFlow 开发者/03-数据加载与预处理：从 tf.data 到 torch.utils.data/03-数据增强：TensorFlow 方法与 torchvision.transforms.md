# 数据增强：TensorFlow 方法与 torchvision.transforms

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/data-augmentation-pytorch-torchvision)

[返回章节目录](README.md) · [返回课程目录](../README.md)

数据增强是一种普遍采用的技术，用于人工扩充你的训练数据集，并提升模型对未见过数据的泛化能力。像 TensorFlow 这样的框架经常在 `tf.data` 数据管道中利用 `tf.image` 中的函数或 Keras 预处理层来应用随机翻转、旋转和颜色调整等转换。PyTorch 通过其 `torchvision.transforms` 模块，为这些任务提供了类似且功能强大的工具集。

基本理念保持不变：即在训练期间即时对输入数据应用随机（或确定性）修改。这有助于你的模型对输入中的变化更具适应性，从而提高性能并减少过拟合 (overfitting)。

### 增强方法：TensorFlow 和 PyTorch

在 TensorFlow 中，你可能会使用以下常见方法之一应用数据增强：

1. **`tf.image` 函数**：这些函数（例如 `tf.image.random_flip_left_right`、`tf.image.random_brightness`）通常在 `tf.data.Dataset.map()` 调用中应用于单个图像或图像批次。
2. **Keras 预处理层**：诸如 `tf.keras.layers.RandomFlip`、`tf.keras.layers.RandomRotation` 和 `tf.keras.layers.RandomZoom` 等层可以直接集成到你的 `tf.keras.Sequential` 模型或 `tf.data` 管道中。这些层具有它们是模型图的一部分并可能在 GPU 上运行的优势。

PyTorch 将大多数常见的图像转换（包括数据增强）集中在 `torchvision.transforms` 模块中。这些转换通常设计用于操作 PIL (Python Imaging Library) 图像或 PyTorch 张量。

### 使用 `torchvision.transforms`

`torchvision.transforms` 模块提供了一系列可调用类，每个代表一个特定的转换。一种常见的做法是使用 `transforms.Compose` 将多个转换链式组合起来。这会创建一个单一的管道，按顺序应用每个转换。

以下是一些常用转换及其对应的 TensorFlow 功能的简要概述：

- **尺寸调整和裁剪**：

  - `transforms.Resize(size)`: 将输入图像调整为给定尺寸。类似于 `tf.image.resize`。
  - `transforms.CenterCrop(size)`: 裁剪图像中心。类似于 `tf.image.central_crop` 或 `tf.keras.layers.CenterCrop`。
  - `transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')`: 在随机位置裁剪图像。
  - `transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3./4., 4./3.))`: 一种常用增强方法，裁剪图像的随机部分并调整其尺寸。这对于训练图像分类模型非常有效。它与 `tf.image.sample_distorted_bounding_box` 结合尺寸调整的使用方式有些类似。
- **翻转**：

  - `transforms.RandomHorizontalFlip(p=0.5)`: 以给定概率随机水平翻转图像。类似于 `tf.image.random_flip_left_right` 或 `tf.keras.layers.RandomFlip("horizontal")`。
  - `transforms.RandomVerticalFlip(p=0.5)`: 随机垂直翻转图像。类似于 `tf.image.random_flip_up_down` 或 `tf.keras.layers.RandomFlip("vertical")`。
- **旋转**：

  - `transforms.RandomRotation(degrees, interpolation=InterpolationMode.NEAREST, expand=False, center=None, fill=0)`: 以随机角度旋转图像。类似于 `tf.keras.layers.RandomRotation`。
- **颜色和像素调整**：

  - `transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`: 随机改变图像的亮度、对比度、饱和度和色调。此单个转换涵盖的功能类似于 `tf.image.random_brightness`、`tf.image.random_contrast`、`tf.image.random_saturation` 和 `tf.image.random_hue`。
  - `transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))`: 应用随机高斯模糊。
  - `transforms.RandomGrayscale(p=0.1)`: 随机将图像转换为灰度。
- **转换和标准化**：

  - `transforms.ToTensor()`: 这是一个重要的转换。它将范围 [0, 255] 内的 PIL 图像或 NumPy 数组 (H x W x C) 转换为形状为 (C x H x W) 且范围在 [0.0, 1.0] 的 PyTorch FloatTensor。TensorFlow 通常更隐式地处理张量转换和缩放，或通过 `tf.image.convert_image_dtype` 进行。
  - `transforms.Normalize(mean, std, inplace=False)`: 使用均值和标准差对张量图像进行标准化。你通常会使用数据集的均值和标准差。这类似于使用 `tf.keras.layers.Normalization` 或手动执行 $(input - mean) / std$。

### 组合转换

要应用一系列增强，你可以使用 `transforms.Compose`。例如，如果你想随机裁剪并调整图像大小，然后随机水平翻转它，最后将其转换为张量，你可以这样做：

```python
from torchvision import transforms
from PIL import Image

# 示例：定义转换组合
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 标准 ImageNet 归一化
                         std=[0.229, 0.224, 0.225])
])

# 假设 'img' 是从文件加载的 PIL 图像
# transformed_img_tensor = data_transforms(img)
```

在此代码段中，`data_transforms` 现在是一个可调用对象，它将按顺序对输入图像应用每个已定义的转换。

### 将增强集成到数据管道中

你通常会将这些转换集成到你的 `Dataset` 类中。当从你的 `Dataset` 请求一个项目时（即在 `__getitem__` 方法中），你会加载数据样本（例如，一张图像），然后应用组合的转换，最后返回。

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform # 'transform' 参数将是我们的 'transforms.Compose' 对象

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB") # 以 PIL 图像格式加载图像
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image) # 在此处应用转换

        return image, label

# 用法：
# train_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
# train_dataset = CustomImageDataset(image_paths=..., labels=..., transform=train_transforms)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

下图展示了图像在 PyTorch 中通过典型增强管道的流程：

> 图像由 `Dataset` 加载，然后通过 `transforms.Compose` 定义的一系列转换。最终的增强张量再由 `DataLoader` 进行批处理。

`torchvision.datasets` 中的许多内置数据集（如 `ImageFolder`、`CIFAR10` 等）也接受 `transform` 参数 (parameter)，允许你在实例化它们时直接传入你的 `transforms.Compose` 对象。

### TensorFlow 与 PyTorch 增强：快速比较

尽管数据增强的目标相同，但实现细节略有不同。以下是一个比较表：

| 特性/任务 | TensorFlow (`tf.image` / Keras 层) | PyTorch (`torchvision.transforms`) |
| --- | --- | --- |
| **主要模块** | `tf.image`，`tf.keras.layers` (预处理) | `torchvision.transforms` |
| **链式处理** | 在 `Dataset.map()` 中按顺序应用，或在 Keras `Sequential` 模型中。 | `transforms.Compose([...])` |
| **输入类型** | 主要为张量。 | PIL 图像、张量。 |
| **张量转换** | 通常是隐式的，或通过 `tf.image.convert_image_dtype`。 | 通过 `transforms.ToTensor()` 显式转换。 |
| **像素范围** | `tf.image` 函数通常期望 `[0,1]` 或 `[0,255]`。`ToTensor()` 缩放至 `[0,1]`。 | 输入 PIL 图像通常为 `[0,255]`。`ToTensor()` 输出 `[0,1]`。 |
| **形状约定** | 图像为 (H, W, C)。 | `ToTensor()` 后张量为 (C, H, W)。PIL 图像为 (H, W, C)。 |
| **执行** | 可以是 TensorFlow 图的一部分（尤其是 Keras 层），可能经过 JIT 编译并在 GPU 上运行。 | 通常是数据加载期间在 CPU 上执行的 Python 函数。 |
| **集成** | `Dataset.map(augment_fn)`，模型中的 Keras 层。 | 在 `Dataset.__getitem__` 中或传入 `torchvision.datasets`。 |

### 注意事项

- **CPU 与 GPU 增强**：大多数 `torchvision.transforms` 在数据加载过程中作为一部分在 CPU 上执行。TensorFlow 的 Keras 预处理层具有优势，如果它们是模型的一部分，可能在 GPU 上运行增强。对于 PyTorch 中的 GPU 加速增强，你可以尝试 Kornia 等库，尽管这是一个更高级的主题。
- **转换顺序**：`transforms.Compose` 中的顺序很重要。例如，`ToTensor()` 通常应在 `Normalize()` 之前，因为 `Normalize()` 需要一个张量。几何变换通常在颜色变换之前应用。
- **确定性转换与随机转换**：在评估或测试期间，你通常希望使用确定性转换（例如，`Resize` 和 `CenterCrop`，而不是 `RandomResizedCrop`）。你通常会为训练和验证定义单独的转换管道。

  ```python
  # 示例：训练和验证的独立转换
  train_transforms = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])

  val_transforms = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  ```
- **自定义转换**：如果内置转换不够用，PyTorch 允许你通过实现 `__call__` 方法来轻松创建自己的自定义转换类。

了解 `torchvision.transforms` 后，你可以在 PyTorch 项目中有效地应用数据增强策略，就像使用 TensorFlow 的工具一样。`transforms.Compose` 的灵活性以及它与 `Dataset` 对象的轻松集成，使其成为预处理和增强数据的便捷系统。

## 参考资料

- [torchvision.transforms documentation](https://pytorch.org/vision/stable/transforms.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch图像变换模块的官方文档，提供所有可用变换、其参数和用法的详细信息。
- TensorFlow Image module (\`tf.image\`) and Keras preprocessing layers documentation [1](https://www.tensorflow.org/api_docs/python/tf/image) [2](https://www.tensorflow.org/api_docs/python/tf/keras/layers/preprocessing) — TensorFlow Authors (2024)
  TensorFlow图像处理函数（`tf.image`）和Keras预处理层的官方文档，有助于理解TensorFlow的数据增强方法，并可与PyTorch进行比较。
- [Deep Residual Learning for Image Recognition](https://doi.org/10.1109/CVPR.2016.90) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2016)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 770-778; DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  介绍残差网络并详细说明了数据增强策略，包括随机尺寸裁剪和水平翻转，这些策略常用于训练先进的图像分类模型。
- [Data Loading and Preprocessing Tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) — Sasank Chilamkurthy (2024)
  一个官方的PyTorch教程，提供关于自定义数据集、数据加载器以及如何使用`torchvision.transforms`进行PyTorch数据加载和预处理的实用指南。

---

[上一节](02-%E6%89%B9%E5%A4%84%E7%90%86%E4%B8%8E%E8%BF%AD%E4%BB%A3%EF%BC%9ATensorFlow%20DataLoaders%20%E4%B8%8E%20PyTorch%20DataLoaders.md) · [下一节](04-%E5%9C%A8%20PyTorch%20%E4%B8%AD%E5%AE%9E%E7%8E%B0%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AE%E9%9B%86.md)
