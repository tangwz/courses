---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-data-loading-for-tf-users"
lesson: "preprocessing-pytorch-transforms"
sourceId: 5755
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-3-pytorch-data-loading-for-tf-users/preprocessing-pytorch-transforms"
title: "使用 PyTorch 变换进行数据预处理"
description: "了解如何使用 `torchvision.transforms` 进行数据预处理，与 Keras 预处理层类似。"
order: 5
plots: []
sourceHash: "672ae97b3cf94aff56499024844e8bb32b59a8e6adf7fe5eba837f33566cf034"
sourceCorrections: []
---

在将数据送入神经网络 (neural network)之前，对 `Dataset` 对象中的数据进行预处理和增强是常见的操作。PyTorch 主要通过 `torchvision.transforms` 模块来处理这项任务，特别是在处理图像数据时。这些变换在目的上类似于 TensorFlow 的 `tf.image` 函数或 Keras 预处理层（如 `tf.keras.layers.Rescaling`、`tf.keras.layers.RandomFlip` 等），但它们在数据加载流程中的集成方式有所不同。

PyTorch 中的变换本质上是可调用的 Python 对象，它们接收一个数据样本（如 PIL 图像或 PyTorch 张量），并返回一个变换后的版本。这些变换可以链式组合以创建数据预处理流程。

### 常用预处理和数据增强变换

`torchvision.transforms` 模块提供了一系列丰富的预置变换。下面我们来看一些最常用的。

#### 转换与归一化 (normalization)

1. **`transforms.ToTensor()`**: 这是处理图像数据时的一个基础变换。它将 PIL 图像（Python 图像库）或 NumPy 数组（形状为 H x W x C，即高 x 宽 x 通道）转换为形状为 C x H x W 的 `torch.FloatTensor`。重要的是，它还会将像素值从 [0, 255] 范围缩放到 [0.0, 1.0]。这通常是应用于图像数据的首批变换之一。

   在 TensorFlow 中，你可以通过使用 `tf.image.convert_image_dtype` 并指定 `tf.float32` 来达到类似效果（这也会缩放到 [0,1]），或者通过手动类型转换和除法运算来实现。
2. **`transforms.Normalize(mean, std)`**: 此变换使用给定均值和标准差对每个通道的张量图像进行归一化。应用的公式是 `output[channel] = (input[channel] - mean[channel]) / std[channel]`。通过确保输入特征值具有相似的范围，归一化有助于模型更有效地训练。`mean` 和 `std` 是值的序列（如列表或元组），每个通道对应一个值。对于 RGB 图像，你需要为均值和标准差各提供三个值。

   ```python
   # ImageNet 预训练模型的常用均值和标准差
   normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
   ```

   这类似于 TensorFlow 的 `tf.image.per_image_standardization`（如果你对每张图像计算均值和方差），或者更常见的是，使用预计算的统计数据应用 `tf.keras.layers.Normalization` 层，或通过对其数据样本使用 `adapt` 方法。

#### 缩放与裁剪

1. **`transforms.Resize(size)`**: 将输入图像缩放到给定的 `size`。如果 `size` 是一个整数，图像较短的边将被匹配到这个数值，同时保持宽高比。如果 `size` 是一个序列，如 `(h, w)`，它会将图像缩放到这些确切的尺寸。
2. **`transforms.CenterCrop(size)`**: 在中心裁剪给定图像。`size` 可以是一个整数（用于正方形裁剪）或一个序列 `(h, w)`。这通常在初始缩放后，用于验证或测试流程。

#### 数据增强

数据增强是一种通过应用随机变换来人为增加训练数据集多样性的技术。这有助于使模型更有效并减少过拟合 (overfitting)。这些变换通常只应用于训练数据。

1. **`transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3./4., 4./3.))`**: 随机裁剪图像的一部分并将其缩放到给定的 `size`。`scale` 参数 (parameter)定义了原始图像要裁剪区域的范围，而 `ratio` 定义了裁剪区域的宽高比范围。这是图像分类中非常常用的增强方法。
2. **`transforms.RandomHorizontalFlip(p=0.5)`**: 以给定概率 `p`（默认为 0.5）随机水平翻转给定图像。
3. **`transforms.RandomRotation(degrees, interpolation=...)`**: 以随机角度旋转图像。`degrees` 可以是一个数字（例如 `30`，表示在 -30 到 +30 度之间旋转）或一个序列，如 `(-30, 30)`。
4. **`transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`**: 随机改变图像的亮度、对比度、饱和度和色调。你可以为每个参数指定范围。例如，`brightness=0.2` 表示从 `[max(0, 1 - 0.2), 1 + 0.2]` 中随机选择一个亮度因子。

TensorFlow 通过 `tf.image` 函数（例如 `tf.image.random_flip_left_right`、`tf.image.random_brightness`）或 Keras 预处理层（例如 `tf.keras.layers.RandomFlip`、`tf.keras.layers.RandomRotation`）提供类似的数据增强功能。一个主要区别是，Keras 层通常是模型图的一部分，并且可以在加速器（GPU/TPU）上执行，而 PyTorch 变换通常由 `DataLoader` 工作进程在 CPU 上应用。基于 CPU 的预处理，使用 `tf.data.map` 和 `tf.image` 函数，与常见的 PyTorch 方法更直接的对应。

### 使用 `transforms.Compose` 组合变换

为了按顺序应用多个变换，PyTorch 提供了 `transforms.Compose`。你将一个变换对象列表传递给它的构造函数，`Compose` 将按照它们在列表中出现的顺序应用它们。

下面是如何为训练和验证定义独立的变换流程：

```python
from torchvision import transforms

# 训练流程：包含数据增强
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),      # 数据增强：随机裁剪并缩放
    transforms.RandomHorizontalFlip(),      # 数据增强：随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # 数据增强：颜色调整
    transforms.ToTensor(),                  # 转换为张量并缩放到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 归一化
                         std=[0.229, 0.224, 0.225])
])

# 验证/测试流程：无数据增强，仅进行必要的预处理
val_test_transforms = transforms.Compose([
    transforms.Resize(256),                 # 将较短的边缩放到 256
    transforms.CenterCrop(224),             # 中心裁剪 224x224
    transforms.ToTensor(),                  # 转换为张量并缩放到 [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 归一化
                         std=[0.229, 0.224, 0.225])
])
```

然后，你将这些组合变换对象传递给你的 `Dataset` 实例，通常在其初始化期间或在其 `__getitem__` 方法中。`torchvision.datasets` 中的许多内置数据集（如 `ImageFolder`）直接接受 `transform` 参数 (parameter)。

```python
# 使用 torchvision.datasets.ImageFolder 的示例
# from torchvision.datasets import ImageFolder
# train_dataset = ImageFolder(root='path/to/train_data', transform=train_transforms)
# val_dataset = ImageFolder(root='path/to/val_data', transform=val_test_transforms)
```

如果你正在编写自定义 `Dataset`，你通常会在 `__getitem__` 方法中应用变换：

```python
# class MyCustomDataset(Dataset):
#     def __init__(self, data_paths, labels, transform=None):
#         self.data_paths = data_paths
#         self.labels = labels
#         self.transform = transform
#
#     def __getitem__(self, index):
#         img_path = self.data_paths[index]
#         image = Image.open(img_path).convert("RGB") # 以 PIL 格式加载图像
#         label = self.labels[index]
#
#         if self.transform:
#             image = self.transform(image) # 应用变换
#
#         return image, label
#
#     def __len__(self):
#         return len(self.data_paths)
```

### 创建自定义变换

虽然 `torchvision.transforms` 涵盖了许多常见用法，但你可能需要专门的预处理步骤。在 PyTorch 中可以轻松创建自定义变换。变换可以是任何接受样本并返回变换后样本的可调用对象。通常，通过定义一个带有 `__call__` 方法的类来实现这一点。

假设你想为图像添加特定类型的噪声：

```python
import torch
import numpy as np
from PIL import Image # 假设本示例输入为 PIL 图像

class AddSaltAndPepperNoise:
    def __init__(self, amount=0.05):
        self.amount = amount

    def __call__(self, img):
        # 确保 img 是 PIL 图像，转换为 numpy 数组
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        np_img = np.array(img)
        original_shape = np_img.shape

        # 添加椒盐噪声中的“盐”部分
        num_salt = np.ceil(self.amount * np_img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in original_shape]
        if len(original_shape) == 2: # 灰度图
            np_img[coords[0], coords[1]] = 255
        else: # 彩色图
            np_img[coords[0], coords[1], :] = 255

        # 添加椒盐噪声中的“胡椒”部分
        num_pepper = np.ceil(self.amount * np_img.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in original_shape]
        if len(original_shape) == 2: # 灰度图
            np_img[coords[0], coords[1]] = 0
        else: # 彩色图
            np_img[coords[0], coords[1], :] = 0

        return Image.fromarray(np_img) # 返回 PIL 图像

    def __repr__(self):
        return self.__class__.__name__ + f'(amount={self.amount})'

# 在流程中使用自定义变换
# 假设在此之前图像已加载为 PIL 图像
custom_pipeline = transforms.Compose([
    AddSaltAndPepperNoise(amount=0.03),
    transforms.ToTensor(), 
    # ... 其他变换，如 Normalize
])
```

这个自定义变换 `AddSaltAndPepperNoise` 现在可以与内置变换一起集成到 `transforms.Compose` 流程中。请记住，你的自定义变换的输入和输出类型应与流程中相邻的变换兼容（例如，如果它期望 PIL 图像，请确保它放在 `ToTensor()` 之前）。

### 给 TensorFlow 开发者的注意事项

从 TensorFlow 的数据预处理过渡时，请记住以下几点：

1. **执行环境**: 如前所述，Keras 预处理层可以是模型的一部分，并在 GPU/TPU 上执行。`torchvision.transforms` 中的 PyTorch 变换通常由 `DataLoader` 在独立的 worker 进程中在 CPU 上应用。这种差异可能会影响整体训练吞吐量 (throughput)，具体取决于变换的复杂性和 CPU/GPU 工作负载的平衡。使用 `tf.data.map` 和 `tf.image` 函数进行基于 CPU 的预处理，与常见的 PyTorch 方法更直接的对应。
2. **无状态与有状态预处理**: 大多数 PyTorch 变换是无状态的（例如 `RandomHorizontalFlip`），或配置有固定参数 (parameter)（例如带有预定义均值/标准差的 `Normalize`）。如果你需要从数据中计算统计量（如归一化 (normalization)的均值和标准差），你通常会离线计算一次，然后将这些值硬编码到 `Normalize` 变换中。这与 Keras 的 `Normalization` 层形成对比，后者有一个 `adapt()` 方法可以从一批数据中在线计算这些统计量。
3. **模型导出**: 当 Keras 预处理层是模型的一部分时，它们在保存和导出模型时会自然包含在内。对于 PyTorch，由于变换是数据加载流程的一部分，你需要确保在部署模型进行推理 (inference)时应用相同的预处理。这可能涉及在推理代码中复制变换流程，或对于某些部署场景，将一些预处理步骤嵌入 (embedding)到模型本身中（例如，通过在模型的 `forward` 传递中添加基于 `nn.Module` 的变换，尽管这对于标准输入预处理来说不太常见）。

通过理解和使用 `torchvision.transforms`，你可以在 PyTorch 中构建灵活高效的数据预处理流程，调整你在 TensorFlow 生态系统中学习到的技术。请记住，始终只将数据增强应用于训练数据，并确保验证集和测试集预处理的一致性。通常需要通过实验来找到适用于你特定任务和数据集的最佳变换组合。

## 参考资料

- [\`torchvision.transforms\` - PyTorch documentation](https://pytorch.org/vision/stable/transforms.html) — PyTorch Core Team (2024)
  PyTorch图像转换库的官方文档，详细介绍了各种预处理和数据增强转换。
- [Lecture Notes: Training Neural Networks, Part 2: Setting up the Data and the Layers](https://cs231n.github.io/neural-networks-2/) — Stanford University CS231n Course Staff (2023)
  关于数据预处理、数据增强技术和归一化策略的讲义，对有效训练神经网络至关重要。
