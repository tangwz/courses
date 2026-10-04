---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-nn-module-for-keras-users"
lesson: "accessing-model-parameters-pytorch"
sourceId: 5749
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/accessing-model-parameters-pytorch"
title: "访问和修改模型参数及层"
description: "了解如何在 PyTorch 模型中查看、访问和修改参数及子模块。"
order: 6
plots: []
sourceHash: "64e1326cd2b152d174d61ac8698649f6d3c5b47ff93c825e92d3d8d9f801be29"
sourceCorrections: []
---

在通过继承 `torch.nn.Module` 类定义了神经网络 (neural network)结构之后，通常需要与其组成部分进行交互。你可能需要检查其结构，查看初始或训练得到的权重 (weight)，修改参数 (parameter)以进行微调 (fine-tuning)，甚至替换整个层。如果你来自 TensorFlow 和 Keras，你可能熟悉 `model.summary()`、访问 `model.layers` 以及使用 `get_weights()` 或 `set_weights()` 等工具。PyTorch 提供了一种更直接、更符合 Python 习惯的方式来完成这些任务。

### 查看模型结构

PyTorch 模型作为标准的 Python 类，可以通过几种直观的方式进行查看。最直接的方法就是简单地打印模型实例。这提供了一个分层的视图，展现了网络中的模块和子模块。

```python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Linear(10 * 12 * 12, 50) # 假设输入为28x28，(28-5+1)/2 = 12
        self.output = nn.Linear(50, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 10 * 12 * 12) # 展平
        x = torch.relu(self.classifier(x))
        x = self.output(x)
        return x

model = SimpleNet()
print(model)
```

这个输出显示了你定义的层、它们的类型以及在实例化时传入的任何参数 (parameter)。在显示架构方面，这类似于 Keras 的 `model.summary()`，尽管 `model.summary()` 通常会包含每层的参数数量和输出形状，这些在 PyTorch 中如果需要的话，你将单独计算。

为了进行更程序化的访问对层（它们本身是 `nn.Module` 实例），PyTorch 提供了几种迭代器：

- `model.children()`: 这会返回一个迭代器，用于遍历模型的直接子模块。对于上面的 `SimpleNet`，`model.children()` 将返回分配给 `self.features` 的 `nn.Sequential` 块、`nn.Linear` 层 `self.classifier` 和 `nn.Linear` 层 `self.output`。
- `model.modules()`: 这会返回一个迭代器，以递归方式遍历网络中的所有模块，包括模型本身作为第一项，然后是其子模块，再然后是子模块的子模块，依此类推。
- `model.named_children()` 和 `model.named_modules()`: 这些通常更有用，因为它们返回 `(name, module)` 元组，其中 `name` 是你在 `__init__` 中指定的属性名称（例如 'features'、'classifier'），或者是 `nn.Sequential` 中层的索引。

```python
print("模型的子模块：")
for name, module in model.named_children():
    print(f"名称: {name}, 类型: {type(module)}")

print("\n模型中的所有模块：")
for name, module in model.named_modules():
    print(f"路径: {name}, 类型: {type(module)}")
```

这种详细的自省对于调试、理解复杂的架构或选择性地对网络中的特定部分应用操作很有价值。

> `SimpleNet` 示例中模块的层级结构。这些注释说明了 `named_parameters()` 和 `named_modules()` 将如何引用此结构内的元素。

### 访问模型参数 (parameter)（权重 (weight)和偏置 (bias)）

在 PyTorch 中，模型的可学习参数（权重和偏置）是 `torch.Tensor` 的实例，它们的 `requires_grad` 属性设置为 `True`。`torch.nn.Module` 提供了便捷的方式来访问这些参数：

- `model.parameters()`: 返回一个迭代器，用于遍历模型的所有参数。
- `model.named_parameters()`: 返回一个迭代器，生成 `(name, parameter)` 元组。名称是字符串，指示参数的路径，例如 `features.0.weight`（`features` 序列块中第一层的权重）或 `classifier.bias`。

让我们查看 `SimpleNet` 的参数：

```python
print("\n模型参数（名称、形状、是否需要梯度）：")
for name, param in model.named_parameters():
    print(f"名称: {name}, 形状: {param.shape}, 需要梯度: {param.requires_grad}")

# 访问特定参数
# 例如，'classifier' 层的权重
classifier_weights = model.classifier.weight
print(f"\n分类器层权重形状: {classifier_weights.shape}")
print(f"分类器层权重数据（第一行的前5个值）：\n {classifier_weights.data[0, :5]}")
```

这与 Keras 不同，在 Keras 中，`layer.get_weights()` 返回一个 NumPy 数组列表。在 PyTorch 中，你直接得到 `torch.Tensor` 对象。参数张量的 `.data` 属性让你能够直接访问底层数据，在需要直接操作时，绕过梯度跟踪系统。

另一个重要方面是 `state_dict`。模块的 `state_dict` 是一个 Python 字典对象，它将每一层映射到其参数张量。对于参数，键是参数名称（例如 `features.0.weight`），值是张量本身。`state_dict` 对于保存和加载模型检查点非常重要，这是一个我们稍后会详细讲解的话题。

```python
# 获取状态字典
state_dict = model.state_dict()
print("\nmodel.state_dict() 中的键：")
for key in state_dict.keys():
    print(key)

# 示例：从 state_dict 访问 'output' 层的偏置
output_bias_from_state_dict = state_dict['output.bias']
print(f"\n从 state_dict 获取的输出层偏置: {output_bias_from_state_dict}")
```

### 修改模型参数 (parameter)

有几种情况你可能想修改模型参数：自定义初始化、将预训练 (pre-training)权重 (weight)加载到模型的一部分（迁移学习 (transfer learning)），或在训练期间冻结层。

#### 直接修改

你可以使用参数的 `.data` 属性就地修改参数。这对于将所有偏置 (bias)设为零或在模型创建后应用自定义初始化方案等操作很有用。

```python
# 示例：将 'classifier' 层中的所有偏置设为零
print(f"\n分类器偏置修改前: {model.classifier.bias.data[:5]}")
with torch.no_grad(): # 对于参数的就地修改很重要
    model.classifier.bias.data.fill_(0)
print(f"分类器偏置修改后: {model.classifier.bias.data[:5]}")
```

`with torch.no_grad():` 上下文 (context)管理器在这里很重要。它暂时禁用梯度计算，当你手动更改 `requires_grad=True` 的参数值时，这是必要的。直接修改 `.data` 是一种较旧的模式；确保没有梯度副作用的更常见方法是在 `torch.no_grad()` 中进行操作。

#### 冻结层（参数 `requires_grad`）

对于迁移学习，一种常见的方法是冻结预训练层的权重，并且只训练模型中新添加的部分。在 PyTorch 中，这是通过将你想要冻结的参数的 `requires_grad` 属性设置为 `False` 来完成的。

```python
# 冻结 'features' 块中的所有参数
print(f"\n冻结前，features.0.weight.requires_grad: {model.features[0].weight.requires_grad}")
for param in model.features.parameters():
    param.requires_grad = False

print(f"冻结后，features.0.weight.requires_grad: {model.features[0].weight.requires_grad}")

# 验证现在只有分类器和输出层的参数需要梯度
print("\n冻结 'features' 后需要梯度的参数：")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
```

这类似于在 Keras 中设置 `layer.trainable = False`。当你将这些参数传递给优化器时，只有那些 `requires_grad=True` 的参数才会在反向传播 (backpropagation)期间计算和更新梯度。

### 访问和修改层（模块）

由于 PyTorch 模型中的层是父模块的属性（或像 `nn.Sequential` 这样的容器中的元素），你可以使用标准的 Python 属性访问或索引来访问和修改它们。

#### 访问层

如前所述，如果在 `__init__` 中将一个层定义为 `self.mylayer = nn.Linear(...)`，你可以通过 `model.mylayer` 访问它。对于 `nn.Sequential` 容器中的层，你可以使用整数索引：

```python
# 访问 'features' 块中的第一个卷积层
first_conv_layer = model.features[0]
print(f"\nfeatures 中的第一个卷积层: {first_conv_layer}")
```

#### 修改层

替换一个层就像将一个新的模块赋给属性一样简单。这尤其强大，展示了 PyTorch 的灵活性。

```python
# 示例：用一个具有不同输出特征的新层替换 'output' 层
print(f"\n原始输出层: {model.output}")
old_output_layer_weights_shape = model.output.weight.shape

# 假设我们想改为20个输出类别
model.output = nn.Linear(model.output.in_features, 20) 
# 新层的权重默认会随机初始化

print(f"新输出层: {model.output}")
print(f"旧输出层权重形状: {old_output_layer_weights_shape}")
print(f"新输出层权重形状: {model.output.weight.shape}")

# 确保新层的参数是可训练的（默认就是）
print(f"新输出层权重是否需要梯度: {model.output.weight.requires_grad}")
```

这种动态修改是 PyTorch 的一个显著优点。如果你使用 Keras 的 Sequential API，替换一个层通常意味着从该点开始重建模型或创建一个新模型。尽管 Keras 的 Functional API 提供了更多的灵活性，但 PyTorch 的方法对于 Python 开发者来说感觉非常自然。

当修改模型架构时，例如替换一个层，请记住，如果优化器已初始化，可能需要更新。具体来说，如果你替换一个层，其参数 (parameter)的 `id` 将改变。优化器通常存储对它们正在优化的参数的引用。如果你创建一个优化器，然后替换一个层，该优化器仍将持有对旧层参数的引用。常见做法是在所有模型修改完成后创建优化器，或者使用新的 `model.parameters()` 集合重新初始化它。

```python
# 如果模型结构改变且优化器已创建，则（重新）初始化优化器
# 例如：
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

能够轻松访问、检查和修改参数以及整个层，为你的 PyTorch 模型提供了细粒度的控制。这对于精准微调 (fine-tuning)、用于研究的模型修改或将预训练 (pre-training)模型适应新任务等高级技术非常宝贵。这与 TensorFlow Keras 不同，在 Keras 中，尽管可能，但此类操作通常感觉不那么直接，并且可能需要对底层图构建有更深入的了解，尤其是在 TF1.x 中。在 PyTorch 中，这只是 Python 对象和属性。

## 参考资料

- [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) — PyTorch Authors (2023)
  PyTorch中所有神经网络模块基类的官方文档，描述了访问、检查和管理子模块、参数及其状态的方法。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本全面的教科书，提供了使用PyTorch构建、检查和定制神经网络的深入解释和实践示例，涵盖了`nn.Module`和参数处理等核心概念。
- [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — Sasank Chilamkurthy (2017)
  一个官方的PyTorch教程，演示了微调预训练模型的实践技术，包括冻结层和替换输出层，这涉及对模型参数和架构的直接操作。
