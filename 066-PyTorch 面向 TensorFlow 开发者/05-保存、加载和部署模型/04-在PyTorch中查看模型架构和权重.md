# 在PyTorch中查看模型架构和权重

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/inspecting-pytorch-models)

[返回章节目录](README.md) · [返回课程目录](../README.md)

检查PyTorch模型的内部结构和学习到的参数 (parameter)非常重要，原因有多种。无论是使用预训练 (pre-training)模型，还是已加载到环境中的模型，这一过程都有助于验证模型是否正确加载、在微调 (fine-tuning)前理解其架构、调试意外行为，以及了解特定网络是如何构建的。如果你是来自TensorFlow的开发者，你会发现PyTorch提供了同样强大但不同的模型检查方法。

## 使用 `print()` 获取概览

在PyTorch中获取模型架构概览最直接的方法是直接打印模型对象。此命令会遍历模型构造函数 (`__init__`) 中定义的模块，并打印它们的结构。

我们来定义一个简单的卷积神经网络 (neural network)（CNN）作为示例。假设这个网络用于处理类似MNIST的单通道28x28图像数据集。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # 全连接层
        # 在对 28x28 图像进行两次 5x5 卷积和 2x2 最大池化操作后：
        # conv1 的输出大小：(28-5+1)/1 = 24x24。在 pool1 后：12x12。
        # conv2 的输出大小：(12-5+1)/1 = 8x8。在 pool2 后：4x4。
        # 因此，展平后的大小 = 20 * 4 * 4 = 320
        self.fc1 = nn.Linear(320, 50) 
        self.fc2 = nn.Linear(50, 10) # 10 个类别的输出
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 320) # 展平张量
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

# 实例化模型
model = SimpleNet()
print(model)
```

The output will look something like this:

```
SimpleNet(
  (conv1): Conv2d(1, 10, kernel_size=(5, 5), stride=(1, 1))
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(10, 20, kernel_size=(5, 5), stride=(1, 1))
  (relu2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=320, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
  (log_softmax): LogSoftmax(dim=1)
)
```

此输出列出了 `SimpleNet` 中定义为属性的每个层，包括其类型（例如 `Conv2d`、`Linear`、`ReLU`）以及用于初始化它的参数 (parameter)（例如 `Conv2d` 的 `in_channels`、`out_channels`、`kernel_size`）。

对于TensorFlow开发者来说，这与Keras的 `model.summary()` 方法有些类似。然而，PyTorch中 `print(model)` 的输出是 `nn.Module` 层次结构的直接表示。它通常不像 `model.summary()` 那样包含各层的输出形状或详细的参数计数表，但它提供了模型组件及其配置的清晰视图。

## 遍历模型层

为了以更具编程性的方式访问模型的层，PyTorch提供了几个迭代器。

### 直接子模块：`model.children()` 和 `model.named_children()`

如果你只想遍历模型的直接子模块（即在其 `__init__` 中被分配为属性的那些），你可以使用 `model.children()` 或 `model.named_children()`。后者还会提供你分配给属性的名称。

```python
print("模型的直接子模块：")
for name, module in model.named_children():
    print(f"名称: {name}, 模块: {module}")
```

这将列出 `conv1`、`relu1`、`pool1` 等模块，它们是 `SimpleNet` 的直接属性。

### 所有模块：`model.modules()` 和 `model.named_modules()`

要递归遍历网络中的所有模块（例如，如果你有一个 `nn.Sequential` 块作为子模块，这将包括嵌套模块），请使用 `model.modules()` 或 `model.named_modules()`。

```python
print("\n模型中的所有模块（递归）：")
for name, module in model.named_modules():
    # 顶层模型本身也包含在内，名称为空
    if name: # 在此处过滤掉顶层模型本身，以获得更清晰的输出
      print(f"名称: {name}, 模块类型: {type(module).__name__}")
```

这种递归遍历对于访问网络的每一个部分很有用，包括容器模块内的部分。

下面的图表展示了 `SimpleNet` 模型的架构，显示了数据如何流经其构成层和操作。

> `SimpleNet` 模型内的数据流和层组织。蓝色方框表示可学习层或固定操作模块，绿色 `cds` 形状表示函数操作或张量操作，灰色椭圆表示输入/输出。

## 访问和检查参数 (parameter)（权重 (weight)和偏置 (bias)）

检查模型的实际可学习参数（权重和偏置）。

### `model.named_parameters()`

`model.named_parameters()` 迭代器对此非常有用。它返回形如 (参数 (parameter)名称, 参数张量) 的元组。

```python
print("\n模型参数（名称、大小、是否需要梯度）：")
for name, param in model.named_parameters():
    print(f"参数: {name}, 大小: {param.size()}, 需要梯度: {param.requires_grad}, 数据类型: {param.dtype}")
    # 查看实际值（对大型张量请谨慎）：
    # print(f"  前几个值: {param.data.flatten()[:3].tolist()}")
```

输出将显示诸如 `conv1.weight`、`conv1.bias`、`fc1.weight`、`fc1.bias` 等名称，以及它们的张量形状 (`param.size()`) 和是否跟踪梯度 (`param.requires_grad`)。你可以使用 `param.data` 访问实际的张量数据。

这类似于在Keras中遍历 `model.layers` 并调用 `layer.get_weights()`，后者返回一个NumPy数组列表（通常是权重 (weight)，然后是偏置 (bias)）。在PyTorch中，`param.data` 让你直接访问 `torch.Tensor` 对象本身。

### `model.state_dict()`

如在保存和加载模型部分所述，`model.state_dict()` 返回一个有序字典，其中包含所有参数 (parameter)（权重 (weight)和偏置 (bias)）和持久缓冲区（例如批归一化 (normalization)中的运行均值）。虽然它主要用于持久化，但也是一种方便的方式来查看所有参数名称及其对应的张量。

```python
# state_dictionary = model.state_dict()
# for param_name in state_dictionary:
#     print(f"{param_name}\t{state_dictionary[param_name].size()}")
```

## 定位特定层及其参数 (parameter)

如果你的层被定义为模型类的属性（例如 `self.conv1 = nn.Conv2d(...)`），你可以通过它们的属性名称直接访问它们。一旦你拥有了特定的层对象，你就可以检查其参数，如 `weight` 和 `bias`。

```python
# 访问 conv1 层
conv1_layer = model.conv1
print(f"\nconv1 层: {conv1_layer}")

# 访问 conv1 的权重参数
conv1_weight = model.conv1.weight
print(f"conv1 权重张量形状: {conv1_weight.size()}")
# print(f"conv1 权重值（第一个元素）: {conv1_weight[0,0,0,0].item()}") # 单个值的示例

# 访问 fc1 的偏置参数
if model.fc1.bias is not None:
    fc1_bias_shape = model.fc1.bias.size()
    print(f"fc1 偏置张量形状: {fc1_bias_shape}")
else:
    print("fc1 层没有偏置参数。")
```

这种直接访问对于有针对性地检查甚至修改模型的特定部分非常强大。

## 实际检查：验证加载的模型状态

检查的一个常见用例是验证模型状态是否已从检查点正确加载。让我们模拟这个过程：

```python
import os

# 1. 保存当前模型的 state_dict
torch.save(model.state_dict(), "simplenet_checkpoint.pth")

# 2. 创建模型的新实例
model_reloaded = SimpleNet()

# 可选：加载前检查参数（它将被随机初始化）
# print(f"参数 conv1.weight（第一个值）加载前: {model_reloaded.conv1.weight.data[0,0,0,0].item()}")

# 3. 加载保存的 state_dict
model_reloaded.load_state_dict(torch.load("simplenet_checkpoint.pth"))
model_reloaded.eval() # 如果适用，设置为评估模式

# 4. 检查以验证
# 比较原始模型和重新加载模型中的特定参数
original_conv1_weight_val = model.conv1.weight.data[0,0,0,0].item()
reloaded_conv1_weight_val = model_reloaded.conv1.weight.data[0,0,0,0].item()

print(f"\n原始模型 conv1.weight（第一个值）: {original_conv1_weight_val}")
print(f"重新加载模型 conv1.weight（第一个值）: {reloaded_conv1_weight_val}")

if original_conv1_weight_val == reloaded_conv1_weight_val:
    print("参数验证成功：重新加载模型的权重与原始模型匹配。")
else:
    print("参数验证失败：权重不匹配。")

# 清理创建的文件
os.remove("simplenet_checkpoint.pth")
```

这个流程确保了你的模型持久化机制按预期工作，并且加载的模型准确反映了保存的状态。通过使用这些检查方法，你可以清晰地了解你的PyTorch模型架构及其学习到的参数 (parameter)。这种能力对于有效的模型开发、调试和部署非常重要，它使你能够自信地将TensorFlow的使用经验应用于PyTorch环境。

## 参考资料

- [\`torch.nn\` - PyTorch Documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  PyTorch 神经网络模块的官方文档，它是所有网络模块的基类。其中详细介绍了与架构和参数检查相关的方法和属性。
- [Saving and Loading Models - PyTorch Documentation](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Matthew Inkawhich (2018)
  这份官方教程详细介绍了如何保存和加载 PyTorch 模型，其中包括了 `state_dict` 的使用，这也是访问和检查参数的重要工具。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本实用书籍，提供了 PyTorch 模型定义、训练和检查的深入解释和示例，有助于理解架构和权重方面的知识。

---

[上一节](03-PyTorch%20%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%9A%84%E6%A3%80%E6%9F%A5%E7%82%B9%E7%AD%96%E7%95%A5.md) · [下一节](05-TorchScript%20%E5%9C%A8%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E6%96%B9%E9%9D%A2%E7%9A%84%E7%AE%80%E4%BB%8B.md)
