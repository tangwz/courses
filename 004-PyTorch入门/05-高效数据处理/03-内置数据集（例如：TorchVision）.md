# 内置数据集（例如：TorchVision）

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/built-in-datasets-torchvision)

[返回章节目录](README.md) · [返回课程目录](../README.md)

尽管创建自定义 `Dataset` 类为您的特定数据提供了最大的灵活性，但许多深度学习 (deep learning)任务，特别是在研究和基准测试中，使用标准化数据集。手动准备这些数据集涉及下载、解压、组织文件和编写解析逻辑，这可能既耗时又容易出错。

幸运的是，PyTorch 提供了配套库，可以简化常见领域的数据处理过程。对于计算机视觉，`torchvision` 包是一个不可或缺的工具。它不仅包含流行的数据集，还包含预训练 (pre-training)模型和常用的图像转换函数。本节主要介绍如何访问和使用 `torchvision.datasets` 提供的数据集。

### 使用 `torchvision.datasets` 访问数据集

`torchvision.datasets` 模块提供了对许多广泛使用的计算机视觉数据集的便捷访问，例如 MNIST、Fashion MNIST、CIFAR 10/100、ImageNet、COCO 等。使用这些数据集很简单。通常，您会从 `torchvision.datasets` 导入特定的数据集类并实例化它。

让我们看一个使用 CIFAR 10 数据集的例子，它包含 60,000 张 32x32 彩色图像，分为 10 个类别。

```python
import torchvision
import torchvision.transforms as transforms

# 定义一个简单的转换，将图像转换为 PyTorch 张量
transform = transforms.Compose([transforms.ToTensor()])

# 加载训练数据集
# root: 数据将被存储/查找的目录
# train=True: 指定训练集
# download=True: 如果本地未找到数据，则下载
# transform: 将定义的转换应用于每张图像
train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             download=True,
                                             transform=transform)

# 加载测试数据集
test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)

print(f"CIFAR-10 training dataset size: {len(train_dataset)}")
print(f"CIFAR-10 test dataset size: {len(test_dataset)}")

# 访问单个数据点（图像、标签）
img, label = train_dataset[0]
print(f"Image shape: {img.shape}") # 通常输出：torch.Size([3, 32, 32])
print(f"Label: {label}")          # 输出：表示类别的整数
```

当您首次运行此代码时，`torchvision` 会检查指定的 `root` 目录（在本例中为 `./data`）。如果 CIFAR 10 数据不存在，设置 `download=True` 会指示 `torchvision` 自动将数据集下载并解压到该目录中。后续运行将发现数据已存在于本地并跳过下载。

注意 `transform` 参数 (parameter)。您可以在此处指定数据预处理步骤，这些步骤在数据样本加载后但在 `__getitem__` 返回之前应用于每个样本。我们使用了 `transforms.ToTensor()`，它将 PIL 图像格式（`torchvision` 数据集常用）转换为 PyTorch 张量。数据转换将在下一节中进行更详细的介绍。

### 结构与使用

重要的是，`torchvision.datasets` 返回的对象（如上文的 `train_dataset` 和 `test_dataset`）是继承自 `torch.utils.data.Dataset` 的类实例。这意味着它们实现了必需的 `__len__` 和 `__getitem__` 方法，使其与 PyTorch 的 `DataLoader` 完全兼容。

- `len(train_dataset)` 返回数据集中样本的总数。
- `train_dataset[i]` 返回第 $i$ 个样本，通常是一个元组 `(data, target)`，其中 `data` 是预处理后的输入（例如，图像张量），`target` 是对应的标签或标注。

以下是 CIFAR-10 训练集中类别分布的简单可视化：



[交互图表：CIFAR-10 训练集类别分布](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/built-in-datasets-torchvision#plot-ruqdo4)

<details>
<summary>图表数据（JSON）</summary>

```json
{
  "data": [
    {
      "type": "bar",
      "x": [
        "飞机",
        "汽车",
        "鸟",
        "猫",
        "鹿",
        "狗",
        "青蛙",
        "马",
        "船",
        "卡车"
      ],
      "y": [
        5000,
        5000,
        5000,
        5000,
        5000,
        5000,
        5000,
        5000,
        5000,
        5000
      ],
      "marker": {
        "color": [
          "#4dabf7",
          "#f06595",
          "#74c0fc",
          "#e64980",
          "#37b24d",
          "#fd7e14",
          "#51cf66",
          "#ff922b",
          "#1c7ed6",
          "#f76707"
        ]
      }
    }
  ],
  "layout": {
    "title": "CIFAR-10 训练集类别分布",
    "xaxis": {
      "title": "类别"
    },
    "yaxis": {
      "title": "图像数量"
    },
    "margin": {
      "l": 50,
      "r": 20,
      "t": 50,
      "b": 80
    },
    "height": 350
  }
}
```

</details>



> CIFAR-10 数据集是平衡的，每个类别恰好有 5,000 张训练图像。

### 在计算机视觉中

尽管 `torchvision` 最为完善，但其他领域也存在类似的库：

- **`torchaudio`**: 为音频处理任务提供数据集（如 SpeechCommands、LJSpeech 等）、模型和转换功能。
- **`torchtext`**: 为自然语言处理提供数据集（如 IMDb 情感分析、WikiText 语言建模）、分词 (tokenization)器 (tokenizer)和词汇工具。注意：`torchtext` 经历了重大的 API 变更，因此请查阅其文档以了解当前的使用模式。

使用这些库遵循相似的原则：导入所需的数据集类，实例化它（通常带有下载和预处理选项），然后将生成的 `Dataset` 对象与 `DataLoader` 一起使用。

依靠这些内置数据集可以显著加快开发和实验速度，使您能够专注于模型架构和训练，而不是数据获取和准备，尤其是在使用标准基准时。请记住，这些数据集对象直接与本章后面讨论的 `DataLoader` 集成，从而实现高效的批处理和洗牌。

## 参考资料

- [\`torch.utils.data\` - PyTorch Documentation](https://pytorch.org/docs/stable/data.html) — PyTorch Foundation (2025)
  PyTorch数据加载工具的官方文档，包括支持`torchvision`数据集的`Dataset`和`DataLoader`。
- [\`torchvision.datasets\` - PyTorch Documentation](https://pytorch.org/vision/stable/datasets.html) — PyTorch Foundation (2024)
  描述TorchVision中内置数据集、其用法和配置参数的官方文档。
- [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) — Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton (2009)
  介绍CIFAR-10和CIFAR-100数据集的原始技术报告，该数据集在章节中作为主要示例使用。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本涵盖PyTorch的教科书，包括数据加载、`Dataset`、`DataLoader`以及有效使用`torchvision`数据集。

---

[上一节](02-%E4%BD%BF%E7%94%A8%20%60torch.utils.data.Dataset%60.md) · [下一节](04-%E6%95%B0%E6%8D%AE%E5%8F%98%E6%8D%A2%20%28%60torchvision.transforms%60%29.md)
