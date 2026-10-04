# 在PyTorch中构建一个简单的CNN

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/building-simple-cnn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

将卷积神经网络 (neural network) (CNN)的核心概念转化为可运行的PyTorch模型。CNN通常通过堆叠卷积层、激活函数 (activation function)和池化层来构建，之后通常跟随一个或多个全连接层，用于分类或回归。PyTorch的`torch.nn`模块提供了这些核心组件的预构建实现，以便高效构建。

我们的目标是构建一个能够处理图像数据的简单CNN。我们将从定义网络结构开始，将其作为一个Python类，并继承自`torch.nn.Module`。

### CNN的核心层在PyTorch中

1. **卷积层 (`nn.Conv2d`)**：此层对输入应用可学习的滤波器。主要参数 (parameter)有：

   - `in_channels`：输入张量的通道数（例如，灰度图像为1，RGB图像为3）。
   - `out_channels`：滤波器数量（也是输出张量的通道数）。每个滤波器学习检测不同的特征。
   - `kernel_size`：滤波器尺寸（高 x 宽）。单个整数`k`表示`k x k`的滤波器。
   - `stride`：滤波器每次移动的像素数（默认为1）。
   - `padding`：在输入周围添加填充，常用于控制输出的空间尺寸（默认为0）。

   ```python
   import torch
   import torch.nn as nn

   # 示例：一个Conv2d层，接收3个输入通道（例如RGB图像），
   # 使用5x5滤波器生成16个输出通道。
   conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=2)
   ```
2. **池化层 (`nn.MaxPool2d`)**：此层减小特征图的空间尺寸（高和宽），使表示更紧凑，并对特征位置的变化略微更具鲁棒性。

   - `kernel_size`：取最大值的窗口大小。
   - `stride`：窗口移动的距离。对于非重叠池化，通常设为等于`kernel_size`。

   ```python
   # 示例：一个MaxPool2d层，使用2x2窗口和步长为2。
   # 这通常会将输入的高度和宽度减半。
   pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
   ```
3. **激活函数 (activation function) (例如，`nn.ReLU`)**：引入非线性，使网络能够学习复杂的模式。ReLU（修正线性单元）是常用选择。它逐元素应用：$f(x) = max(0, x)$。

   ```python
   # ReLU激活函数
   relu1 = nn.ReLU()
   ```
4. **线性层 (`nn.Linear`)**：一个标准的全连接层。通常用于CNN的末尾，在空间特征被提取和展平之后。

   - `in_features`：输入特征的数量（需要展平卷积/池化层的输出）。
   - `out_features`：输出特征的数量（例如，分类任务中的类别数）。

   ```python
   # 示例：一个线性层，接收一个展平的512个特征向量，
   # 并输出10个值（例如，用于10个类别）。
   fc1 = nn.Linear(in_features=512, out_features=10)
   ```

### 定义CNN结构

我们通过继承`nn.Module`来定义我们的CNN。各层通常在`__init__`方法中定义，而前向传播（数据如何流经各层）则在`forward`方法中定义。

我们来构建一个具有以下结构的CNN：

- 输入：[批大小, 1, 28, 28]（例如，像MNIST那样的灰度图像）
- Conv1：1个输入通道，16个输出通道，5x5核，步长1，填充2
- ReLU1
- MaxPool1：2x2核，步长2
- Conv2：16个输入通道，32个输出通道，5x5核，步长1，填充2
- ReLU2
- MaxPool2：2x2核，步长2
- 展平
- Linear1：（输入特征取决于MaxPool2的输出），128个输出特征
- ReLU3
- Linear2：128个输入特征，10个输出特征（例如，用于10个类别）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F # 通常包含激活函数和其他实用工具

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 层定义
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        # 最大池化层1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # 最大池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        # fc1的输入特征取决于池化后的输出形状
        # 输入：28x28 -> Conv1 (padding=2) -> 28x28 -> Pool1 (stride=2) -> 14x14
        # -> Conv2 (padding=2) -> 14x14 -> Pool2 (stride=2) -> 7x7
        # 因此，展平后的尺寸是 32 个通道 * 7 高度 * 7 宽度 = 1568
        self.fc1 = nn.Linear(in_features=32 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 用于10个类别的输出

    def forward(self, x):
        # 定义数据流经各层的方式
        # 输入x形状：[批大小, 1, 28, 28]

        # 应用Conv1、ReLU、Pool1
        x = self.pool1(F.relu(self.conv1(x)))
        # pool1后的形状：[批大小, 16, 14, 14]

        # 应用Conv2、ReLU、Pool2
        x = self.pool2(F.relu(self.conv2(x)))
        # pool2后的形状：[批大小, 32, 7, 7]

        # 展平张量以用于全连接层
        # -1 保持批大小维度不变
        x = x.view(-1, 32 * 7 * 7)
        # view后的形状：[批大小, 1568]

        # 应用FC1和ReLU
        x = F.relu(self.fc1(x))
        # fc1后的形状：[批大小, 128]

        # 应用FC2（输出层，此处无激活函数，通常与损失函数一起应用）
        x = self.fc2(x)
        # fc2后的形状：[批大小, 10]
        return x
```

我们来可视化架构流程：

> 数据和张量形状流经`SimpleCNN`模型。请注意，通道数增加，而空间维度（高/宽）减小。

### 使用模型

要使用这个模型，首先实例化该类。然后，您可以将输入数据（作为PyTorch张量）传入其中。输入张量必须具有预期的形状，包括批次维度。对于我们的`SimpleCNN`，这具体为`[N, 1, 28, 28]`，其中N是批次中的样本数量。

```python
# 实例化模型
model = SimpleCNN()
print(model)

# 创建一个虚拟输入张量（4张图像的批次，1个通道，28x28）
# 如果您打算训练，需要梯度跟踪
dummy_input = torch.randn(4, 1, 28, 28)

# 将输入传入模型（前向传播）
output = model(dummy_input)

# 检查输出形状
print(f"\nInput shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}") # 预期：[4, 10]
```

运行此代码将打印模型的层结构，并确认输出张量形状符合我们的预期（`[4, 10]`），表示批次中每张图像的10个类别的得分。

这个例子展示了如何在`nn.Module`中组合`nn.Conv2d`、`nn.MaxPool2d`、`nn.ReLU`和`nn.Linear`层来创建一个基本的CNN。设计CNN时的一个重要细节是正确计算每个层之后张量形状的变化，特别是在连接卷积/池化部分和全连接部分时。我们将在下一节更详细地说明这些形状的跟踪。

## 参考资料

- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  一篇开创性的论文，介绍了LeNet-5，这是一种具有里程碑意义的卷积神经网络架构，为CNN的许多核心概念奠定了基础，包括用于图像分类的卷积层、池化层和全连接层。这与本文讨论的简单CNN直接相关。
- [torch.nn](https://pytorch.org/docs/stable/nn.html) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch官方`torch.nn`模块文档，提供了神经网络的构建模块，包括示例中使用的`nn.Conv2d`、`nn.MaxPool2d`、`nn.ReLU`和`nn.Linear`层。
- [Deep Learning](http://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本内容全面的教材，提供了深度学习的理论基础，其中第九章专门介绍了卷积网络，详细解释了其原理、层和操作。
- [Learn the Basics: Building the Neural Network](https://pytorch.org/tutorials/beginner/basics/intro.html) — Suraj Subramanian, Seth Juarez, Cassie Breviu, Dmitry Soshnikov, Ari Bornstein (2021)
  Publisher: PyTorch
  一个PyTorch官方教程，指导用户使用`torch.nn.Module`和本节讨论的层构建神经网络，包括一个简单的CNN，从而加强实际操作。

---

[上一节](01-%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%20%28CNN%29%20%E6%A6%82%E8%BF%B0.md) · [下一节](03-%E7%90%86%E8%A7%A3CNN%E5%B1%82%E7%9A%84%E8%BE%93%E5%85%A5-%E8%BE%93%E5%87%BA%E5%BD%A2%E7%8A%B6.md)
