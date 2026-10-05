# 模型架构：Keras API 与 PyTorch 的 nn.Module

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-2-pytorch-nn-module-for-keras-users/model-architectures-keras-pytorch)

[返回章节目录](README.md) · [返回课程目录](../README.md)

构建神经网络 (neural network)时，TensorFlow 开发者通常熟悉 `tf.keras.Sequential` 模型，用于简单层堆叠，以及 Keras 函数式 API，用于多输入、多输出或共享层等复杂架构。PyTorch 提供了类似的方法来定义模型架构，主要通过继承 `torch.nn.Module` 来实现，这提供了很大的灵活性，对于较简单的情况则可使用 `torch.nn.Sequential`。这些 PyTorch 方法将被介绍，并与 Keras 中的对应部分进行比较。

### 简单堆叠：Keras `Sequential` 与 PyTorch `nn.Sequential`

对于层按线性顺序排列的直接模型，Keras 提供了 `Sequential` API。

#### Keras `Sequential` API：快速回顾

在 Keras 中，您可以这样定义一个简单模型：

```python
import tensorflow as tf

keras_sequential_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

keras_sequential_model.summary()
```

这会创建一个模型，数据依次流经 `Flatten`、`Dense(128)`，最后是 `Dense(10)`。

#### PyTorch 的 `nn.Sequential` 容器

PyTorch 提供了 `torch.nn.Sequential`，这是一种创建类似线性层堆叠的便捷方式。它是一个容器，会按顺序将数据通过其所包含的模块。

```python
import torch
import torch.nn as nn

pytorch_sequential_model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
    # Softmax 通常作为损失函数的一部分应用（例如 nn.CrossEntropyLoss）
    # 或者在推理时如果需要，明确地在 forward 过程中应用。
)

print(pytorch_sequential_model)
```

请注意以下几点：

1. `nn.Flatten()` 类似于 `tf.keras.layers.Flatten`。
2. `nn.Linear(in_features, out_features)` 等效于 PyTorch 中的 `tf.keras.layers.Dense`。您需要为第一个线性层指定输入特征。
3. 像 `nn.ReLU()` 这样的激活函数 (activation function)通常作为单独的层添加到 `nn.Sequential` 中。
4. 对于分类，如果您需要直接的输出概率，可以将 `nn.LogSoftmax(dim=1)` 或 `nn.Softmax(dim=1)` 添加为最后一层。然而，如果您使用 `nn.CrossEntropyLoss`，则通常会省略最终的 softmax，因为它结合了 `LogSoftmax` 和 `NLLLoss`。

`nn.Sequential` 容器非常适合原型设计，或者当您的模型架构是简单的前馈管道时。然而，对于数据流更复杂、有共享层或多输入/输出路径的模型，您将转向 PyTorch 更通用的方法：继承 `nn.Module`。

### 使用 `torch.nn.Module` 构建更复杂的架构

Keras 使用其函数式 API 来构建像有向无环图 (DAG) 这样的复杂模型，而 PyTorch 的主要方法是创建一个继承自 `torch.nn.Module` 的自定义类。这种面向对象的方法非常灵活且符合 Python 风格，让您可以完全控制模型处理数据的方式。

#### Keras 函数式 API：用于有向无环图

Keras 函数式 API 允许您通过连接层来定义复杂模型。例如，一个带有跳跃连接的模型可能看起来像这样：

```python
# Keras 函数式 API 示例（说明）
input_tensor = tf.keras.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_tensor)
x = tf.keras.layers.MaxPooling2D((2,2))(x)
residual = x # 存储用于跳跃连接
x = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = tf.keras.layers.Conv2D(32, (1,1), activation='relu', padding='same')(x) # 减少通道
x = tf.keras.layers.Add()([x, residual]) # 添加跳跃连接
x = tf.keras.layers.Flatten()(x)
output_tensor = tf.keras.layers.Dense(10, activation='softmax')(x)

keras_functional_model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
# keras_functional_model.summary()
```

这个 API 对于那些不只是简单堆叠的模型来说非常强大。

#### PyTorch 的核心：继承 `torch.nn.Module`

在 PyTorch 中，您通过将模型定义为一个继承自 `nn.Module` 的类来实现这种灵活性。这种方法包含两个主要部分：

1. **`__init__(self)` 方法**：

   - 您必须首先调用 `super().__init__()`。
   - 这里是您定义模型将使用的所有层的地方（例如，卷积层、线性层、如果它们具有可学习参数 (parameter)或状态的激活函数 (activation function)）。这些层作为模型类的属性存储（例如，`self.conv1 = nn.Conv2d(...)`）。
   - 以这种方式定义为属性的层会自动注册到 `nn.Module`，这意味着 PyTorch 可以追踪它们的参数，并在设备（CPU/GPU）之间移动它们等。
2. **`forward(self, input_data, ...)` 方法**：

   - 此方法定义实际的计算过程：输入数据如何流经您在 `__init__` 中定义的层。
   - 您可以调用这些层（例如，`x = self.conv1(input_data)`），并可以使用任何 Python 逻辑来控制数据流。这就是 PyTorch“按运行定义”特性的体现；计算图是随着 `forward` 方法的执行而动态构建的。
   - `forward` 的参数是模型的输入，而它返回的是模型的输出。

让我们将一个具有更复杂路由可能性的网络（尽管为了说明我们将保持其简单）的类似思路转换为 PyTorch `nn.Module`：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F # 常见用于激活函数等无状态操作

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 定义层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # (批量, 1, 28, 28) -> (批量, 16, 28, 28)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # (批量, 16, 28, 28) -> (批量, 16, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # (批量, 16, 14, 14) -> (批量, 32, 14, 14)
        # 再次池化后: (批量, 32, 7, 7)

        # 占位层，可能用于跳跃连接路径
        self.conv_skip_adjust = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1)

        # 展平尺寸: 32 通道 * 7x7 图像
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x 形状: (批量大小, 1, 28, 28)

        # 第一个卷积块
        x1_conv = F.relu(self.conv1(x))
        x1_pool = self.pool(x1_conv) # (批量, 16, 14, 14)

        # 第二个卷积块
        x2_conv = F.relu(self.conv2(x1_pool))
        x2_pool = self.pool(x2_conv) # (批量, 32, 7, 7)

        # 简单“跳跃”或并行路径的示例
        # 对于真正的跳跃连接，维度必须匹配才能进行加法/拼接。
        # 在这里，我们只是处理它并展示如何将其组合。
        # 假设我们想将 x1_pool (16 通道) 与 x2_pool (32 通道) 结合。
        # 我们可能需要调整 x1_pool 的通道。
        skip_path = self.conv_skip_adjust(x1_pool) # (批量, 32, 14, 14)
        skip_path_pooled = self.pool(skip_path) # (批量, 32, 7, 7)

        # 为了演示，假设我们想将 x2_pool 和 skip_path_pooled 相加
        # 这要求它们具有相同的形状。
        # combined = x2_pool + skip_path_pooled # 这里是发生跳跃连接的地方

        # 在此示例中，我们将以 x2_pool 作为主路径
        # 展平输出以用于全连接层
        x_flat = x2_pool.view(-1, 32 * 7 * 7) # .view 类似于 reshape，-1 推断批量大小

        x_fc1 = F.relu(self.fc1(x_flat))
        # 如果使用 nn.CrossEntropyLoss，这里不需要 softmax
        output = self.fc2(x_fc1)
        return output

# 实例化模型
pytorch_custom_model = SimpleCNN(num_classes=10)
print(pytorch_custom_model)

# 假输入张量示例
dummy_input = torch.randn(64, 1, 28, 28) # (批量大小, 通道, 高度, 宽度)
output = pytorch_custom_model(dummy_input)
print("输出形状:", output.shape) # 预期: (64, 10)
```

在这个 `SimpleCNN` 类中：

- 像 `nn.Conv2d`、`nn.MaxPool2d` 和 `nn.Linear` 这样的层是在 `__init__` 中定义的。
- `forward` 方法决定数据流向。我们使用 `torch.nn.functional` 中的 `F.relu` 进行 ReLU 激活，这是无状态操作的常见做法。另外，`nn.ReLU()` 也可以在 `__init__` 中定义并在 `forward` 中调用。
- `.view(-1, 32 * 7 * 7)` 调用在全连接层之前展平张量，类似于 `tf.keras.layers.Flatten()`。`-1` 告诉 PyTorch 推断批量大小。
- `conv_skip_adjust` 和 `skip_path` 行说明了您可以在何处定义更复杂路径（如跳跃连接）的操作。对于真正的加法跳跃连接，`x2_pool` 和 `skip_path_pooled` 需要具有相同的形状。

### 比较 Keras API 和 PyTorch `nn.Module`

| 特性 | Keras `Sequential` | Keras 函数式 API | PyTorch `nn.Sequential` | PyTorch `nn.Module` 继承 |
| --- | --- | --- | --- | --- |
| **主要用途** | 简单线性堆叠 | 带有分支、共享层的模型 | 简单线性堆叠 | 所有模型类型，特别是复杂模型 |
| **灵活性** | 低 | 高 | 低 | 非常高 |
| **模型定义** | 层列表 | 互连层的图 | 模块序列 | 带有 `__init__` 和 `forward` 的 Python 类 |
| **动态行为** | 否（图静态定义） | 否（图静态定义） | 有限（静态序列） | 是（`forward` 中的 Python 逻辑） |
| **调试** | 相对简单 | 对于大型图可能很复杂 | 相对简单 | 使用 Python 调试器更简单 |
| **多输入/输出** | 否 | 是 | 否 | 是（通过 `forward` 签名/返回） |
| **共享层** | 否（非自然） | 是 | 否（非自然） | 是（实例化一次，多次使用） |

### 为什么继承 `nn.Module` 是 PyTorch 的标准做法

尽管 `nn.Sequential` 对于简单情况很有用，但继承 `nn.Module` 是 PyTorch 中构建模型的惯用且最强大的方法，原因在于：

1. **很大的灵活性**：`forward` 方法是纯 Python 代码。您可以使用 `if` 语句、`for` 循环、调用类的其他方法，或集成任何 Python 库来定义计算。这使得模型的结构可以在运行时根据输入数据或其他条件而变化。
2. **清晰与组织性**：对于复杂模型，在 `__init__` 中定义层并在 `forward` 中编写数据流逻辑，提供了清晰且有组织的结构。
3. **易于调试**：由于 `forward` 过程是 Python 代码，您可以使用标准 Python 调试工具（如 `pdb` 或 print 语句）来检查张量并随时了解模型的行为。
4. **自然处理复杂结构**：
   - **多输入/输出**：只需定义您的 `forward` 方法以接受多个参数 (parameter)和/或返回多个张量。

     ```python
     class MultiInputOutputModel(nn.Module):
         def __init__(self):
             super().__init__()
             self.layer1_input1 = nn.Linear(10, 20)
             self.layer1_input2 = nn.Linear(5, 20)
             self.shared_layer = nn.Linear(20, 30)
             self.output_branch1 = nn.Linear(30, 1)
             self.output_branch2 = nn.Linear(30, 1)

         def forward(self, input1, input2):
             x1 = F.relu(self.layer1_input1(input1))
             x2 = F.relu(self.layer1_input2(input2))

             # 示例：拼接或添加处理后的输入
             merged = x1 + x2 # 假设形状匹配以便进行加法

             shared_out = F.relu(self.shared_layer(merged))

             out1 = self.output_branch1(shared_out)
             out2 = self.output_branch2(shared_out)
             return out1, out2
     ```
   - **共享层/模块**：在 `__init__` 中定义一个层一次（例如，`self.shared_conv = nn.Conv2d(...)`），并在 `forward` 中对不同输入多次调用它（例如，`out_a = self.shared_conv(input_a)`，`out_b = self.shared_conv(input_b)`）。

### 模型结构的可视化：Keras 函数式与 PyTorch `nn.Module`

让我们考虑一个带有简单跳跃连接的模型来说明结构定义。

**Keras 函数式 API**
您定义输入，然后定义层，并显式连接它们。跳跃连接涉及获取一个较早的张量，并将其与后面的张量相加或拼接。

**PyTorch `nn.Module` 继承**
`forward` 方法直接实现此流。

> 一张图说明了跳跃连接如何在 Keras 函数式 API 中表达，以及逻辑如何在 PyTorch `nn.Module` 的 `forward` 方法中流经。

在 PyTorch 中，通常没有像 Keras 函数式 API 那样显式的 `Input` 层。输入形状通过您在第一次前向传播期间传递给模型的数据隐式定义，或者在第一层的定义中显式设置（例如，`nn.Conv2d` 的 `in_channels` 或 `nn.Linear` 的 `in_features`）。

从 Keras 的模型构建 API 到 PyTorch 的 `nn.Module` 继承，这是一个转向更编程化和显式定义网络前向传播的方式。这种以 Python 为中心的方法提供了很大的能力和灵活性，特别是对于研究和非传统架构的模型。尽管 `nn.Sequential` 为简单模型提供了类似 Keras 的便利，但掌握 `nn.Module` 继承对于充分运用 PyTorch 的功能非常重要。

## 参考资料

- [Build Keras models](https://www.tensorflow.org/guide/keras/making_new_layers_and_models) — fchollet (2024)
  Publisher: TensorFlow
  Keras模型构建的官方指南，包含Sequential和Functional API的实际示例。
- [torch.nn - PyTorch Documentation](https://pytorch.org/docs/stable/nn.html) — PyTorch Authors (2024)
  PyTorch神经网络模块API的官方文档，说明了`nn.Module`作为基类以及`nn.Sequential`。
- [Dive into Deep Learning](https://d2l.ai/) — Aston Zhang, Zack C. Lipton, Mu Li, and Alex J. Smola (2024)
  Publisher: Cambridge University Press
  一本开放获取的交互式教科书，包含使用PyTorch等框架构建深度网络的章节，适合实际学习。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本实用书籍，提供使用`nn.Module`和相关架构概念构建PyTorch模型的指导。

---

[上一节](01-%E5%AE%9A%E4%B9%89%E7%BD%91%E7%BB%9C%E7%BB%84%E4%BB%B6%EF%BC%9AKeras%20%E5%B1%82%E4%B8%8E%20torch.nn.Module.md) · [下一节](03-%E5%B8%B8%E8%A7%81%E5%B1%82%E7%B1%BB%E5%9E%8B%EF%BC%9A%E5%AF%B9%E6%AF%94%E5%AE%9E%E7%8E%B0.md)
