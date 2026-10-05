# 实践：实现基本CNN和RNN

来源：[原文](https://apxml.com/zh/courses/getting-started-with-pytorch/chapter-7-introduction-common-architectures/practice-implementing-cnn-rnn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

你将使用PyTorch的`nn.Module`以及相关层，实现卷积神经网络 (neural network)（CNN）和循环神经网络（RNN）这两种架构的基本版本。这种动手实践将巩固你对这些模型如何构建以及数据如何在其中流动的理解。

我们将侧重于定义模型结构和理解输入/输出维度，直接基于第4章的`nn.Module`知识和本章前面部分对层的说明进行构建。请记住，这些是简化示例；将它们集成到完整的训练循环中需要添加数据加载（第5章）、损失函数 (loss function)、优化器以及训练逻辑（第6章）。

## 实现一个基本CNN

CNN擅长处理网格状数据，例如图像。让我们构建一个可用于图像分类的简单CNN。我们将定义一个包含卷积层、激活函数 (activation function)、池化层和最终全连接层的网络。

### 定义CNN架构

我们创建一个继承自`nn.Module`的类。在`__init__`中，我们定义所需的层：用于卷积的`nn.Conv2d`，用于激活的`nn.ReLU`，用于池化的`nn.MaxPool2d`，以及用于最终分类的`nn.Linear`层。`forward`方法定义了输入数据如何流经这些层。

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 输入形状: (批次, 1, 28, 28) - 假设是像MNIST那样的灰度图像
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # conv1后的形状: (批次, 16, 28, 28) -> (28 - 3 + 2*1)/1 + 1 = 28
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # pool1后的形状: (批次, 16, 14, 14) -> 28 / 2 = 14

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # conv2后的形状: (批次, 32, 14, 14) -> (14 - 3 + 2*1)/1 + 1 = 14
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # pool2后的形状: (批次, 32, 7, 7) -> 14 / 2 = 7

        # 展平输出以便连接到线性层
        # 展平后的尺寸 = 32 * 7 * 7 = 1568
        self.fc = nn.Linear(32 * 7 * 7, num_classes)

    def forward(self, x):
        # 应用第一个卷积块
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)

        # 应用第二个卷积块
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)

        # 展平卷积层的输出
        # -1 表示推断批次大小
        out = out.view(out.size(0), -1)

        # 应用全连接层
        out = self.fc(out)
        return out
```

在此示例中：

- 我们假设输入是灰度图像（1通道），类似于MNIST数据集中的图像，尺寸为28x28。
- `nn.Conv2d(in_channels=1, out_channels=16, ...)`：接收1个输入通道，应用16个滤波器。`kernel_size=3`、`stride=1`、`padding=1`是常见的选择，它们在卷积后保持空间维度。
- `nn.MaxPool2d(kernel_size=2, stride=2)`：将高度和宽度减半。
- 第二个池化层的输出形状为（批次，32，7，7）。
- `out.view(out.size(0), -1)`：将张量从形状（批次，32，7，7）展平为（批次，32 \* 7 \* 7）=（批次，1568），以便可以将其输入到线性层。
- `nn.Linear(32 * 7 * 7, num_classes)`：最后一层将展平后的特征映射到所需数量的输出类别。

### 使用虚拟数据测试CNN

让我们创建一些匹配预期形状（批次大小，通道，高度，宽度）的虚拟输入数据，并将其传入我们的网络以查看输出形状。

```python
# 实例化模型
cnn_model = SimpleCNN(num_classes=10)

# 创建虚拟输入批次（例如，4张图像，1通道，28x28像素）
# requires_grad=False，因为我们只是进行前向传播演示
dummy_input_cnn = torch.randn(4, 1, 28, 28, requires_grad=False)

# 执行前向传播
output_cnn = cnn_model(dummy_input_cnn)

# 打印输入和输出形状
print(f"Input shape: {dummy_input_cnn.shape}")
print(f"Output shape: {output_cnn.shape}")
```

运行这段代码应该输出：

```
Input shape: torch.Size([4, 1, 28, 28])
Output shape: torch.Size([4, 10])
```

这确认了我们的网络接收一个包含4张图像的批次，并为每张图像输出10个类别的预测。请注意`forward`方法如何决定数据流向，以及我们如何需要根据最终池化层的输出形状来计算线性层的展平尺寸。你可以回顾“理解CNN层的输入/输出形状”一节，练习手动计算这些维度。

## 实现一个基本RNN

RNN旨在处理序列数据。例如，让我们构建一个可以处理字符序列或传感器读数的简单RNN。

### 定义RNN架构

我们将使用`nn.RNN`层。请记住，RNN层期望的输入格式是（序列长度，批次大小，输入特征）。

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN层
        # batch_first=False是默认值，期望输入格式为: (序列长度, 批次, 输入特征)
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)

        # 全连接层，将RNN输出映射到最终输出尺寸
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h0=None):
        # x 形状: (序列长度, 批次, 输入特征)

        # 如果未提供，则初始化隐藏状态
        # 形状: (层数 * 方向数, 批次, 隐藏尺寸)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # 数据通过RNN层
        # out 形状: (序列长度, 批次, 隐藏尺寸) -> 包含每个时间步的输出特征
        # hn 形状: (层数 * 方向数, 批次, 隐藏尺寸) -> 包含最终隐藏状态
        out, hn = self.rnn(x, h0)

        # 我们可以选择使用最后一个时间步的输出
        # out[-1] 形状: (批次, 隐藏尺寸)
        # 或者，如果需要，处理整个序列 'out'
        out_last_step = out[-1, :, :]

        # 将最后一个时间步的输出通过线性层
        final_output = self.fc(out_last_step)
        # final_output 形状: (批次, 输出尺寸)

        return final_output, hn # 返回最终输出和最后一个隐藏状态
```

在此示例中：

- `input_size`：序列中每个步长的特征数量。
- `hidden_size`：隐藏状态中的特征数量。
- `num_layers`：堆叠的RNN层数。
- `nn.RNN(...)`：核心RNN层。`batch_first=False`是默认值，表示序列长度维度在前。
- `forward`方法接受输入序列`x`和一个可选的初始隐藏状态`h0`。如果未提供`h0`，则将其初始化为零。
- `nn.RNN`层返回`out`（每个时间步的输出）和`hn`（最终隐藏状态）。
- 我们经常使用*最后一个*时间步的输出（`out[-1, :, :]`）进行序列分类或预测任务，并将其通过一个最终的线性层。

### 使用虚拟数据测试RNN

让我们创建一个虚拟序列并将其传入我们的RNN。

```python
# 定义参数
input_features = 10 # 例如，字符/单词的嵌入维度
hidden_nodes = 20
output_classes = 5 # 例如，根据序列预测5个类别之一
sequence_length = 15
batch_size = 4

# 实例化模型
rnn_model = SimpleRNN(input_size=input_features, hidden_size=hidden_nodes, output_size=output_classes)

# 创建虚拟输入批次（序列长度，批次大小，输入特征）
# requires_grad=False 用于演示
dummy_input_rnn = torch.randn(sequence_length, batch_size, input_features, requires_grad=False)

# 执行前向传播（不提供h0，它将被初始化）
output_rnn, final_hidden_state = rnn_model(dummy_input_rnn)

# 打印输入和输出形状
print(f"Input sequence shape: {dummy_input_rnn.shape}")
print(f"Output prediction shape: {output_rnn.shape}")
print(f"Final hidden state shape: {final_hidden_state.shape}")
```

运行这段代码应该产生类似如下的输出：

```
Input sequence shape: torch.Size([15, 4, 10])
Output prediction shape: torch.Size([4, 5])
Final hidden state shape: torch.Size([1, 4, 20])
```

这表明模型处理一个包含4个序列的批次，每个序列长15步，每步有10个特征。它为批次中的每个序列输出一个大小为5的最终预测向量 (vector)，以及最终的隐藏状态。隐藏状态的形状反映了（层数，批次大小，隐藏尺寸）。

## 更多实践

现在你已经实现了这些架构的基本版本，请尝试进行试验：

1. **CNN变体：**
   - 更改`nn.Conv2d`层中的`kernel_size`、`stride`或`padding`。在运行代码之前预测输出形状。当步长为1时，`padding='same'`如何影响输出维度？
   - 添加另一个卷积/池化块。请记住重新计算`nn.Linear`层的输入尺寸。
   - 更改卷积层中`out_channels`的数量。
2. **RNN变体：**
   - 增加`SimpleRNN`中的`num_layers`。观察初始隐藏状态`h0`和最终隐藏状态`hn`的形状。
   - 更改`hidden_size`。
   - 将`nn.RNN`替换为`nn.LSTM`或`nn.GRU`。请注意，`nn.LSTM`处理一个*元组*隐藏状态（隐藏状态和单元状态）。你需要相应调整隐藏状态的初始化和处理方式。输入/输出形状大体遵循相同的模式。
   - 修改`forward`方法，以使用*所有*时间步的输出（`out`）而不是仅最后一个，例如通过对每个步应用线性层或使用平均等聚合方法。

这次实践为构建CNN和RNN提供了扎实基础。通过理解如何定义这些层、在`forward`方法中连接它们以及管理它们的输入/输出形状，你将具备良好能力，能够使用PyTorch构建和调整这些强大的架构以完成各种深度学习 (deep learning)任务。

## 参考资料

- [torch.nn - PyTorch 2.2 documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Developers (2024)
  提供 PyTorch 神经网络层和模块的详细文档和使用示例。
- [Gradient-Based Learning Applied to Document Recognition](https://ieeexplore.ieee.org/document/726791) — Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998)
  Journal: Proceedings of the IEEE; Publisher: IEEE; Volume: 86; Pages: 2278-2324; DOI: [10.1109/5.726791](https://doi.org/10.1109/5.726791)
  介绍用于手写数字识别的卷积神经网络 (LeNet-5) 的开创性工作。
- [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) — Sepp Hochreiter and Jürgen Schmidhuber (1997)
  Journal: Neural Computation; Publisher: MIT Press; Volume: 9; Pages: 1735-1780; DOI: [10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)
  介绍了长短期记忆 (LSTM) 循环神经网络架构。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本涵盖深度学习理论和应用的教科书，包括 CNN 和 RNN 架构。

---

[上一节](07-LSTM%20%E5%92%8C%20GRU%20%E7%AE%80%E8%A6%81%E4%BB%8B%E7%BB%8D.md) · [下一节](../08-%E7%9B%91%E6%8E%A7%E4%B8%8E%E8%B0%83%E8%AF%95%E6%A8%A1%E5%9E%8B/01-PyTorch%E5%BC%80%E5%8F%91%E4%B8%AD%E5%B8%B8%E8%A7%81%E9%94%99%E8%AF%AF.md)
