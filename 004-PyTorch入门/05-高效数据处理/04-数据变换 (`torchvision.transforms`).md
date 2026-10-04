# 数据变换 (`torchvision.transforms`)

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/data-transformations-torchvision-transforms)

[返回章节目录](README.md) · [返回课程目录](../README.md)

原始数据，如图像或文本，很少直接以完美适合神经网络 (neural network)输入的格式出现。模型通常需要特定大小和分布的数值张量。此外，为了提高模型的泛化能力并防止过拟合 (overfitting)，通常的做法是通过对现有数据应用随机修改来人工扩充训练数据集。这就是数据变换的作用。

PyTorch，特别是通过用于计算机视觉任务的 `torchvision` 库，提供了一个便捷的模块 `torchvision.transforms`，它包含多种常用操作，这些操作可以链式组合以创建数据处理流程。这些变换主要有两个作用：

1. **预处理：** 标准化数据格式、比例和大小。
2. **数据增强：** 对训练数据应用随机改动以增加其多样性。

让我们看一些基础变换。

### 常用预处理变换

这些变换通常应用于所有数据集划分（训练集、验证集和测试集），以确保一致性。

- **`transforms.ToTensor()`**：这通常是对使用PIL（Python图像库）或NumPy等库加载的图像数据最先应用的变换之一。它将PIL图像或NumPy数组（格式为高 x 宽 x 通道）转换为PyTorch `FloatTensor`（格式为通道 x 高 x 宽）。重要的是，它还将像素值从 [0, 255] 范围缩放到 [0.0, 1.0]。这种转换为张量和标准化范围对于模型输入是必需的。
- **`transforms.Resize(size)`**：将输入图像调整到给定 `size`。如果 `size` 是整数，图像的较短边将匹配此数字，同时保持宽高比。如果 `size` 是像 `(h, w)` 这样的序列，它会将图像调整为精确的高度 `h` 和宽度 `w`。这很重要，因为许多神经网络 (neural network)需要固定大小的输入。
- **`transforms.CenterCrop(size)`**：将图像的中心部分裁剪到给定 `size`。这通常在调整大小后使用，以确保最终图像尺寸精确，同时聚焦于中心区域。
- **`transforms.Normalize(mean, std)`**：使用为每个通道提供的均值和标准差对张量图像进行标准化。应用的操作是：
  $输出 = (输入 - 均值) / 标准差$
  标准化有助于稳定训练，并通过确保输入特征具有相似的比例（通常围绕零居中）来促进更快的收敛。`mean` 和 `std` 通常是值序列，每个输入通道对应一个值（例如，RGB图像有3个值）。来自ImageNet等大型数据集的预计算值通常用作默认值：`mean=[0.485, 0.456, 0.406]` 和 `std=[0.229, 0.224, 0.225]`。

### 常用数据增强变换

这些变换引入随机性，并且通常*仅*应用于训练数据集。这有助于模型学习对输入中的微小变化保持不变，从而降低过拟合 (overfitting)倾向。

- **`transforms.RandomHorizontalFlip(p=0.5)`**：以给定概率 `p`（默认为0.5，表示50%的机会）随机水平翻转图像。
- **`transforms.RandomRotation(degrees)`**：通过从 `(-degrees, +degrees)` 中均匀选择的随机角度旋转图像，或者如果 `degrees` 是序列 `(min, max)`，则在特定范围内旋转。
- **`transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)`**：随机改变图像的亮度、对比度、饱和度和色调。你可以指定每个属性的抖动范围。例如，`brightness=0.2` 意味着随机选择一个介于 `[max(0, 1 - 0.2), 1 + 0.2]` 之间的亮度因子。
- **`transforms.RandomResizedCrop(size)`**：裁剪图像的随机部分并将其调整到所需的 `size`。这是一种常用增强技术，特别适用于训练Inception网络等图像分类模型。

### 组合变换

你很少只应用一个变换。PyTorch 通过 `transforms.Compose` 方便地将多个变换链式组合起来。它接收一个变换对象列表并按顺序应用它们。

下面是为训练数据创建处理流程的示例，包括调整大小、增强、转换为张量和标准化：

```python
import torchvision.transforms as transforms

# 训练数据的变换流程示例
train_transform = transforms.Compose([
    transforms.Resize(256),             # 将较短边调整为256
    transforms.RandomCrop(224),         # 随机裁剪224x224的区域
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),              # 将PIL图像转换为张量（0-1范围）
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # 使用ImageNet统计数据进行标准化
                         std=[0.229, 0.224, 0.225])
])

# 验证/测试数据的变换流程示例（无增强）
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),         # 中心裁剪到224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

print("训练变换：")
print(train_transform)

print("\n测试变换：")
print(test_transform)
```

### 将变换与数据集结合

正如上一节关于 `Dataset` 对象所述，这些组合变换通常在实例化 `Dataset` 时作为参数 (parameter)（通常命名为 `transform` 或 `target_transform`）传入。对于 `torchvision.datasets` 中的内置数据集，这直接简单：

```python
# 假设您已安装 torchvision
from torchvision.datasets import ImageFolder
from pathlib import Path

# torchvision's ImageFolder 的使用示例
train_data_path = Path("path/to/your/train_images")
test_data_path = Path("path/to/your/test_images")

train_dataset = ImageFolder(root=train_data_path, transform=train_transform)
test_dataset = ImageFolder(root=test_data_path, transform=test_transform)

# 当您从 train_dataset 访问一个项时，train_transform 会被应用
sample_image, sample_label = train_dataset[0] # sample_image 现在是一个经过变换的张量
```

对于自定义 `Dataset` 类，您通常会在 `__init__` 方法中接受变换对象，并在返回样本之前在 `__getitem__` 方法中应用它。

```python
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB") # 加载图像

        if self.transform:
            image = self.transform(image) # 应用变换

        return image, label

# 使用方法
custom_train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
custom_test_dataset = CustomImageDataset(test_paths, test_labels, transform=test_transform)
```

通过定义适当的变换并将其集成到您的 `Dataset` 中，您可以确保输入到模型的数据格式正确，并且对于训练数据而言，得到充分增强。这为下一步使用 `DataLoader` 有效地按批加载这些已处理的数据做好了准备。

## 参考资料

- [torchvision.transforms documentation](https://pytorch.org/vision/stable/transforms.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  提供了`torchvision`中所有可用数据转换函数的全面细节，包括使用示例和参数。
- [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) — Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton (2012)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 25; Pages: 1097-1105; DOI: [10.1145/3065386](https://doi.org/10.1145/3065386)
  介绍了AlexNet架构，详细阐述了其创新训练方法，其中包含用于图像分类的多种数据增强策略和归一化技术。
- [CS231n: Convolutional Neural Networks for Visual Recognition - Data Augmentation](https://vertexaisearch.cloud.google.com/grounding-api-redirect/AUZIYQHN3jZuwt8r0aaBN_quV3uUzoilb3BxGUXojTXugipb0iQTOvr7CB9FVAmDhADrLc2H9r0UG-hEWpjDIvPKoQUC4lgcpdoSiEef8v0RDgnK2AYjgRAPVmY-pxNLJ5oznO8vTewhVu-BFUIEhWWzIi4=) — Fei-Fei Li, Yunzhu Li, Ruohan Gao (2023)
  Publisher: Stanford University
  提供了卷积神经网络中用于视觉识别任务的标准数据预处理技术和数据增强方法的教育性概述。
- [Deep Learning with PyTorch: A Developer's Guide](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning
  一本使用PyTorch实现深度学习模型的实践指南，其中包含关于各种计算机视觉应用的数据加载、预处理和增强的专门章节。

---

[上一节](03-%E5%86%85%E7%BD%AE%E6%95%B0%E6%8D%AE%E9%9B%86%EF%BC%88%E4%BE%8B%E5%A6%82%EF%BC%9ATorchVision%EF%BC%89.md) · [下一节](05-%E4%BD%BF%E7%94%A8%20%60torch.utils.data.DataLoader%60.md)
