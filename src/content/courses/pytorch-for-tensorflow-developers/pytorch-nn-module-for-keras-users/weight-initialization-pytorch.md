---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-nn-module-for-keras-users"
lesson: "weight-initialization-pytorch"
sourceId: 5748
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/weight-initialization-pytorch"
title: "PyTorch中的权重初始化方法"
description: "了解 PyTorch 中可用的权重初始化方法，以及它们与 TensorFlow 实践的比较。"
order: 5
plots: []
sourceHash: "75349247bd8f472355028e4fcedb4a309926a5227b30baeea71c946d6d18ccdf"
sourceCorrections: []
---

恰当的权重 (weight)初始化是搭建神经网络 (neural network)以实现有效训练的重要一步。模型权重的初始值会极大影响其学习能力。选择不当的初始权重可能导致梯度消失或爆炸、收敛缓慢，或者网络陷入次优局部极小值。如果您使用过 Keras，可能熟悉为层指定 `glorot_uniform` 或 `he_normal` 等初始化器。PyTorch 提供了类似的灵活性，不过它应用这些初始化方式的方法更明确。

本节将带领您了解 PyTorch 中常见的权重初始化方法，以及如何将它们应用于基于 `torch.nn.Module` 的模型。我们将与 TensorFlow Keras 的实践进行比较，以帮助您迁移现有知识。

### 良好初始化的作用

在网络开始从数据中学习之前，其权重 (weight)需要设置为一些初始值。请思考以下几点:

- **打破对称性**: 如果所有权重都初始化为相同的值（例如，都为零），那么层中的所有神经元在反向传播 (backpropagation)过程中将产生相同的输出并计算相同的梯度。这种对称性会阻止网络学习到不同的特征。随机初始化有助于打破这种对称性。
- **避免梯度问题**: 训练期间，梯度通过网络反向传播。如果权重过小，梯度会呈指数级减小（梯度消失），使早期层难以学习。相反，如果权重过大，梯度会呈指数级增长（梯度爆炸），导致训练不稳定。
- **加快收敛速度**: 周全的初始化方法可以将权重放置在优化更有效的区域，从而加快收敛速度。

### PyTorch 层中的默认初始化

当您在 PyTorch 中创建一个层时，例如 `nn.Linear` 或 `nn.Conv2d`，其参数 (parameter)（权重 (weight)和偏置 (bias)）会自动初始化。例如，`nn.Linear` 和 `nn.Conv2d` 层默认对其权重使用 Kaiming 均匀初始化。偏置（如果存在）通常初始化为零，或者使用源自 Kaiming 初始化范围的均匀分布。

虽然这些默认设置通常是合理的起点，特别是对于常见架构而言，但您通常会想应用针对您的网络架构或激活函数 (activation function)定制的特定初始化方案。

### `torch.nn.init` 模块

PyTorch 将其初始化函数集中在 `torch.nn.init` 模块中。此模块中的大多数函数都是就地操作，这意味着它们直接修改输入张量。这是 PyTorch 的常见约定，通常由函数名中的下划线后缀表示（例如，`xavier_uniform_`）。

我们来看看一些最常用的初始化器。

#### 均匀分布和正态分布

最简单的初始化器从标准分布中抽取值：

- `nn.init.uniform_(tensor, a=0.0, b=1.0)`: 用从均匀分布 $U(a, b)$ 中抽取的值填充输入 `tensor`。
- `nn.init.normal_(tensor, mean=0.0, std=1.0)`: 用从正态分布 $N(\text{均值}, \text{标准差}^2)$ 中抽取的值填充输入 `tensor`。

在 Keras 中，您可以使用 `tf.keras.initializers.RandomUniform` 或 `tf.keras.initializers.RandomNormal`。

#### 常量初始化（零、一、自定义值）

有时，您需要将权重 (weight)或偏置 (bias)初始化为特定的常量值：

- `nn.init.zeros_(tensor)`: 用零填充输入 `tensor`。通常用于初始化偏置。
- `nn.init.ones_(tensor)`: 用一填充输入 `tensor`。
- `nn.init.constant_(tensor, val)`: 用特定值 `val` 填充输入 `tensor`。

Keras 的对应项包括 `tf.keras.initializers.Zeros`、`tf.keras.initializers.Ones` 和 `tf.keras.initializers.Constant`。

#### Xavier (Glorot) 初始化

Xavier 初始化由 Glorot 和 Bengio（2010 年）提出，旨在使激活值和梯度的方差在各层间大致保持不变。这有助于防止信号衰减或爆炸。它特别适合于后面接有 sigmoid 或 tanh 等激活函数 (activation function)的层。

- `nn.init.xavier_uniform_(tensor, gain=1.0)`: 根据均匀分布填充输入 `tensor` 的值，其范围基于 `tensor` 的 `fan_in`（输入单元的数量）和 `fan_out`（输出单元的数量）以及可选的 `gain` 计算得出。范围是 $[-界限, 界限]$，其中
  
  $$
  \text{界限} = \text{增益} \times \sqrt{\frac{6}{fan\_in + fan\_out}}
  $$
  
- `nn.init.xavier_normal_(tensor, gain=1.0)`: 用从正态分布中抽取的值填充输入 `tensor`，其标准差为
  
  $$
  \text{标准差} = \text{增益} \times \sqrt{\frac{2}{fan\_in + fan\_out}}
  $$
  

`gain` 参数 (parameter)允许调整以适应不同的激活函数；`nn.init.calculate_gain(nonlinearity, param=None)` 可用于确定合适的增益（例如，`nn.init.calculate_gain('relu')`）。
在 Keras 中，这些是 `tf.keras.initializers.GlorotUniform` 和 `tf.keras.initializers.GlorotNormal`。

#### Kaiming (He) 初始化

Kaiming 初始化由 He 等人（2015 年）提出，专门为后面接有修正线性单元 (ReLU) 激活函数及其变体（如 Leaky ReLU）的层设计。它考虑了 ReLU 将负输入设置为零这一事实，这会影响输出的方差。

- `nn.init.kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`: 使用均匀分布。`模式` 可以是 `'fan_in'`（在正向传播中保持方差）或 `'fan_out'`（在反向传播 (backpropagation)中保持方差）。`非线性` 参数指定层后使用的激活函数。
  均匀分布的界限是 $[-界限, 界限]$，其中
  
  $$
  \text{界限} = \sqrt{\frac{6}{(1 + a^2) \times fan\_mode}}
  $$
  
  这里，$a$ 是 Leaky ReLU 的负斜率（标准 ReLU 为 0）。
- `nn.init.kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')`: 使用正态分布，其标准差为
  
  $$
  \text{标准差} = \sqrt{\frac{2}{(1 + a^2) \times fan\_mode}}
  $$
  

Keras 提供 `tf.keras.initializers.HeUniform` 和 `tf.keras.initializers.HeNormal`。如前所述，PyTorch 的 `nn.Linear` 和 `nn.Conv2d` 层默认使用 Kaiming 均匀初始化。

#### 正交初始化

- `nn.init.orthogonal_(tensor, gain=1.0)`: 用正交矩阵填充输入二维 `tensor`。这对于初始化循环神经网络 (neural network) (RNN) 中的权重矩阵特别有用，有助于缓解循环连接中的梯度消失/爆炸问题。
  Keras 提供 `tf.keras.initializers.Orthogonal`。

下图说明了 `torch.nn.init` 中的初始化函数如何应用于层的权重和偏置张量。

> 该图表明 `torch.nn.init` 中的函数直接就地修改层实例的权重和偏置张量。

### 在 PyTorch 模型中应用初始化

在 TensorFlow Keras 中，您通常在定义层时将初始化器指定为字符串标识符或初始化器对象，例如 `kernel_initializer='glorot_uniform'` 或 `bias_initializer=tf.keras.initializers.Zeros()`。

PyTorch 的方法更直接：您首先创建层实例，然后将 `torch.nn.init` 中的初始化函数应用于其权重 (weight)和偏置 (bias)张量。

#### 初始化单个层

您可以访问层的参数 (parameter)（例如，`nn.Linear` 和 `nn.Conv2d` 的 `weight` 和 `bias`），并直接应用初始化函数：

```python
import torch
import torch.nn as nn

# 示例：初始化一个线性层
linear_layer = nn.Linear(in_features=128, out_features=64)

# 对权重应用 Xavier 均匀初始化
nn.init.xavier_uniform_(linear_layer.weight)

# 如果偏置存在，将其初始化为零（nn.Linear 默认有偏置）
if linear_layer.bias is not None:
    nn.init.zeros_(linear_layer.bias)

print("已初始化 linear_layer.weight (前5个值):")
print(linear_layer.weight.data[0, :5])
```

这种直接操作对于精细控制或以不同方式初始化特定层很有用。

#### 整个模型的自定义初始化函数

对于更复杂的模型，手动初始化每个层可能很繁琐。常见的做法是编写一个函数，它以模块 `m` 作为输入，检查其类型，并应用所需的初始化。然后，您可以使用 `model.apply(your_init_function)` 方法，该方法会递归地将函数应用于模型中的每个子模块。

```python
import torch
import torch.nn as nn

# 定义一个示例模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * 8 * 8, 10) # 假设输入图片尺寸导致 8x8 特征图

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1) # 展平
        x = self.fc(x)
        return x

# 自定义权重初始化函数
def weights_init_custom(m):
    classname = m.__class__.__name__
    # 对于线性层
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01) # 示例：偏置的小常量值
        print(f"Initialized {classname} with Xavier Uniform for weights and 0.01 for bias.")
    # 对于卷积层
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        print(f"Initialized {classname} with Kaiming Normal for weights and 0 for bias.")
    # 您可以添加更多层类型，如 BatchNorm、RNN 等。

model = SimpleNet()
print("正在应用自定义权重初始化...")
model.apply(weights_init_custom)

# 您可以通过检查一些权重来验证
# print("\n自定义初始化后 conv1 的权重 (第一个过滤器，第一个通道):")
# print(model.conv1.weight.data[0, 0])
```

在 `weights_init_custom` 函数中：

- 我们使用 `isinstance(m, nn.LayerType)` 检查模块 `m` 的类型。
- 根据层类型，我们将所选的初始化应用于 `m.weight` 和 `m.bias`。
- `model.apply(weights_init_custom)` 调用确保此函数对 `model` 本身及其所有子模块（如 `conv1`、`fc` 等）执行。请注意，像 `nn.ReLU` 这样的激活层没有要初始化的参数，因此它们将被传递给函数，但 `isinstance` 检查将防止错误。

#### 在 `__init__` 方法中初始化

另一种简洁的管理初始化方式是直接在自定义 `nn.Module` 的 `__init__` 方法中进行，就在定义每个层之后：

```python
import torch
import torch.nn as nn

class ModelWithInit(nn.Module):
    def __init__(self, in_features, num_classes):
        super(ModelWithInit, self).__init__()
        self.linear1 = nn.Linear(in_features, 128)
        nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
            
        self.relu = nn.ReLU()
        
        self.linear2 = nn.Linear(128, num_classes)
        nn.init.xavier_uniform_(self.linear2.weight) # 输出层示例
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# 创建模型实例
model_custom_init = ModelWithInit(in_features=784, num_classes=10)
# print("__init__ 中初始化后的 linear1 权重:")
# print(model_custom_init.linear1.weight.data[0, :5])
```

这种方式使初始化逻辑与层的定义紧密相关。

### 选择初始化器的考量

- **激活函数 (activation function)**: 激活函数的选择是主要因素。
  - 对于后面接有 ReLU 或其变体的层，使用 Kaiming (He) 初始化。
  - 对于后面接有 sigmoid 或 tanh 的层，使用 Xavier (Glorot) 初始化。
- **网络架构**: 更深的网络可能对初始化更敏感。
- **批量归一化 (normalization)**: 像 `nn.BatchNorm` 这样的层可以使网络对初始权重 (weight)尺度不那么敏感。然而，良好的初始化对于稳定性和速度仍然有利。
- **实验结果**: 理论指导虽然有帮助，但有时针对特定问题或架构的最佳初始化方法是通过实验发现的。

通过了解并应用这些权重初始化技术，您可以显著提升 PyTorch 模型的稳定性和性能，借助您之前在 TensorFlow Keras 相关知识的基础上，同时适应 PyTorch 更明确的风格。

## 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](http://www.jmlr.org/proceedings/papers/v9/glorot10a.html) — Xavier Glorot, Yoshua Bengio (2010)
  Journal: Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (AISTATS); Publisher: JMLR.org; Volume: 9; Pages: 249-256
  提出了Xavier（Glorot）初始化方法，这是稳定神经网络训练，特别是针对sigmoid和tanh激活函数的奠基性工作。
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://doi.org/10.1109/ICCV.2015.122) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (2015)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 1026-1034; DOI: [10.1109/ICCV.2015.122](https://doi.org/10.1109/ICCV.2015.122)
  提出了Kaiming（He）初始化方法，旨在提高使用ReLU激活函数的神经网络的训练稳定性和性能。
- [\`torch.nn.init\` - PyTorch documentation](https://pytorch.org/docs/stable/nn.init.html) — PyTorch Core Team (2022)
  Publisher: PyTorch
  PyTorch官方权重初始化函数的文档，详细说明了其用法和参数。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press; Pages: 289-294
  一本关于深度学习的教科书，提供了参数初始化方法及其理论基础的详细解释。
