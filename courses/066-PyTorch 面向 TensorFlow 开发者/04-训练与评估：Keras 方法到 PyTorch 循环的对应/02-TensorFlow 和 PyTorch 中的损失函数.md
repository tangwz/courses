# TensorFlow 和 PyTorch 中的损失函数

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-4-pytorch-training-loops-for-keras-devs/loss-functions-pytorch-tf)

[返回章节目录](README.md) · [返回课程目录](../README.md)

损失函数 (loss function)（一种衡量模型预测与实际目标值之间差距的方法）的指定方式，在 PyTorch 中比 TensorFlow Keras 更为明确。在 Keras 中，损失通常作为 `model.compile()` 步骤的一部分进行定义，常常使用字符串标识符，如 `'binary_crossentropy'`，或 `tf.keras.losses` 中的一个实例。PyTorch 则要求自行实例化损失函数，然后直接在自定义训练循环中调用它。这种方法虽然需要多写几行代码，但能对整个过程有更清晰的认识和更精细的控制。

## `torch.nn` 中的损失函数 (loss function)

PyTorch 在 `torch.nn` 模块中提供了一系列预定义的损失函数。它们以继承自 `torch.nn.Module` 的类形式实现。要使用其中一个，你首先创建所需损失类的实例，然后在训练循环中，用模型的预测值和真实目标值来调用此实例。

例如，要使用均方误差 (MSE) 损失，你可以这样做：

```python
import torch
import torch.nn as nn

# 实例化损失函数
mse_loss_fn = nn.MSELoss()

# 示例模型输出和目标
predictions = torch.randn(10, 1, requires_grad=True) # 10 个样本，1 个输出特征
targets = torch.randn(10, 1)

# 计算损失
loss = mse_loss_fn(predictions, targets)
print(loss) # 输出：tensor(..., grad_fn=<MseLossBackward0>)

# 损失张量随后可用于反向传播
# loss.backward()
```

许多损失函数在 `torch.nn.functional` 中也以函数形式提供（通常导入为 `F`）。例如，`F.mse_loss(predictions, targets)` 可以达到与上述示例相同的效果。虽然函数形式对于无状态操作来说很方便，但对于模型的主要损失函数，通常更推荐使用 `nn.Module` 版本，因为它们可以保持状态（如归约类型）并更有效地融入 `nn.Module` 的方法。

## 与 TensorFlow/Keras 的区别

从 TensorFlow 迁移时，PyTorch 损失函数 (loss function)的运行方式存在一些重要区别，尤其是在输入预期和目标格式方面。

### 输入预期：Logits vs. 概率

这是从 TensorFlow 转向 PyTorch 的开发者最常感到困惑的方面之一。

- **多类别分类 (`nn.CrossEntropyLoss`)**:

  - 在 TensorFlow 中，`tf.keras.losses.CategoricalCrossentropy` 通常期望输入是概率（即 softmax 层的输出），除非指定 `from_logits=True`。
  - PyTorch 的 `nn.CrossEntropyLoss` **期望直接接收模型最终线性层的原始、未归一化 (normalization)的分数 (logits)**。它在一个高效的步骤中结合了 `LogSoftmax` 层和负对数似然损失 (`NLLLoss`)。这意味着在将模型输出传递给 `nn.CrossEntropyLoss` 之前，你*不应*对其应用 `Softmax` 激活。
- **二元分类 (`nn.BCEWithLogitsLoss` vs. `nn.BCELoss`)**:

  - TensorFlow 的 `tf.keras.losses.BinaryCrossentropy` 可以接受概率（sigmoid 的输出，`from_logits=False`）或 logits（`from_logits=True`）作为输入。
  - PyTorch 提供两个主要选项：
    - `nn.BCELoss`：期望概率作为输入（即模型输出经过 `torch.sigmoid` 处理）。
    - `nn.BCEWithLogitsLoss`：期望原始 logits 作为输入。这个版本在数值上**更稳定**，通常比先使用单独的 `torch.sigmoid` 层再使用 `nn.BCELoss` 的方法更受推荐。它将 sigmoid 层和 `BCELoss` 结合在一个操作中。

### 目标形状和格式

- **`nn.CrossEntropyLoss`（多类别）**:

  - TensorFlow 的 `tf.keras.losses.CategoricalCrossentropy` 通常期望目标是独热编码形式。`tf.keras.losses.SparseCategoricalCrossentropy` 则期望整数类别标签。
  - PyTorch 的 `nn.CrossEntropyLoss` 期望目标张量包含**整数类别索引**。对于一个 $C$ 类问题，每个样本的目标应该是一个介于 $[0, C-1]$ 之间的整数。例如，对于一个 N 样本的批次，预测值的形状可能是 `(N, C)`（logits），目标值的形状可能是 `(N,)`（类别索引）。
- **`nn.BCEWithLogitsLoss` / `nn.BCELoss`（二元/多标签）**:

  - 它们期望目标的形状与预测值相同。对于包含 $N$ 个样本的二元分类，预测值（logits 或概率）和目标值通常都具有 `(N, 1)` 或 `(N,)` 的形状。对于多标签分类（其中每个样本可以属于多个类别），预测值和目标值将具有 `(N, C)` 的形状，其中 $C$ 是类别数量，目标值为二元指示符（0 或 1）。

### 归约

TensorFlow 和 PyTorch 的损失函数 (loss function)都允许你指定如何将批次中每个项目的计算损失聚合为一个单一的标量值。

- 在 TensorFlow Keras 中，这通常通过损失构造函数中的 `reduction` 参数 (parameter)进行控制，可使用 `tf.keras.losses.Reduction.SUM`、`tf.keras.losses.Reduction.NONE`，或隐式使用 `'auto'`（通常表示对批次进行平均）。
- 在 PyTorch 中，基于 `nn.Module` 的损失函数在其构造函数中有一个 `reduction` 参数，可以设置为：
  - `'mean'`（默认）：每个样本损失的总和除以元素数量。
  - `'sum'`：每个样本的损失求和。
  - `'none'`：不应用归约；返回每个元素的损失。

```python
# 归约示例
targets = torch.tensor([0., 1., 0.])
predictions_logits = torch.tensor([-1.0, 2.0, -0.5]) # Logits

loss_fn_mean = nn.BCEWithLogitsLoss(reduction='mean')
print(loss_fn_mean(predictions_logits, targets)) # 输出：tensor(0.3711)

loss_fn_sum = nn.BCEWithLogitsLoss(reduction='sum')
print(loss_fn_sum(predictions_logits, targets)) # 输出：tensor(1.1133) # 0.3711 * 3

loss_fn_none = nn.BCEWithLogitsLoss(reduction='none')
print(loss_fn_none(predictions_logits, targets)) # 输出：tensor([0.3133, 0.1269, 0.4741])
```

## 常用损失函数 (loss function)：比较概览

这里快速比较一些常用损失函数及其等效项：

| 用例 | TensorFlow Keras (`tf.keras.losses`) | PyTorch (`torch.nn`) | PyTorch 输入预期 | PyTorch 目标预期（典型批次） |
| --- | --- | --- | --- | --- |
| 回归（均方） | `MeanSquaredError()` | `MSELoss()` | 任意实数值 | 与输入形状相同 |
| 回归（平均绝对） | `MeanAbsoluteError()` | `L1Loss()` | 任意实数值 | 与输入形状相同 |
| 二元分类 | `BinaryCrossentropy(from_logits=True)` | `BCEWithLogitsLoss()`（推荐） | Logits | `(N, 1)` 或 `(N,)`，浮点数 0.0 或 1.0 |
| 二元分类 | `BinaryCrossentropy(from_logits=False)` | `BCELoss()`（需输入 sigmoid 处理） | 概率 | `(N, 1)` 或 `(N,)`，浮点数 0.0 或 1.0 |
| 多类别分类 | `CategoricalCrossentropy(from_logits=True)` 或 `SparseCategoricalCrossentropy()` | `CrossEntropyLoss()`（推荐） | Logits `(N, C)` | 类别索引 `(N,)`，长整型 |
| 多标签分类 | `BinaryCrossentropy(from_logits=True)` | `BCEWithLogitsLoss()` | Logits `(N, C)` | 二元矩阵 `(N, C)`，浮点数 0.0 或 1.0 |

## 激活函数 (activation function)与损失函数 (loss function)的作用关系可视化

最终激活函数（如 Sigmoid 或 Softmax）与损失函数之间的关系非常重要。PyTorch 用于分类的推荐损失函数（`nn.BCEWithLogitsLoss` 和 `nn.CrossEntropyLoss`）通常会包含激活步骤，以提高数值稳定性和效率。

> TensorFlow/Keras 与 PyTorch 在二元和多类别分类中，模型输出、激活函数（如果独立）和损失函数的常见路径。请注意 PyTorch 推荐直接将 logits 与特定损失函数结合使用。

## 在 PyTorch 中选择合适的损失函数 (loss function)：实用建议

- **对于多类别分类**：几乎总是使用 `nn.CrossEntropyLoss`。确保你的模型最终层输出原始 logits（无 Softmax），并且你的目标是整数类别标签。
- **对于二元分类**：优先选择 `nn.BCEWithLogitsLoss`。你的模型最终层应该为每个样本输出一个 logit。目标应该是浮点数（0.0 或 1.0）。
- **对于多标签分类**（其中每个样本可以属于多个类别）：使用 `nn.BCEWithLogitsLoss`。你的模型最终层应该为每个样本输出 C 个 logits（每个类别一个）。目标应该是一个形状为 `(N, C)` 的二元矩阵，包含 0.0 和 1.0。
- **对于回归任务**：`nn.MSELoss`（用于 L2 损失）和 `nn.L1Loss`（用于 L1 损失，即平均绝对误差）是常见的选择。

## 实现自定义损失函数 (loss function)

如果你需要一个 `torch.nn` 中没有的损失函数，你可以轻松地自行实现。
对于不需要可学习参数 (parameter)或维护状态的简单自定义损失，一个接受预测值和目标值并返回标量损失张量的 Python 函数就足够了。

```python
def my_custom_rmse_loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets)**2))

# 用法:
# loss = my_custom_rmse_loss(model_outputs, ground_truth)
```

如果你的自定义损失函数需要存储参数或继承 `nn.Module` 的行为（例如，以便被 `model.children()` 发现），你可以继承 `nn.Module` 类：

```python
class MyParameterizedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        # 示例：L1 损失和 L2 损失的加权和
        l1_loss = torch.abs(predictions - targets).mean()
        mse_loss = ((predictions - targets)**2).mean()
        return self.alpha * l1_loss + (1 - self.alpha) * mse_loss

# 用法:
# criterion = MyParameterizedLoss(alpha=0.3)
# loss = criterion(model_outputs, ground_truth)
```

## 整合到训练循环中

以下是损失函数 (loss function)通常如何融入 PyTorch 训练步骤的示例：

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 假设 model, train_loader 已定义
# model = 你的神经网络()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# 对于多类别分类
criterion = nn.CrossEntropyLoss()

# --- 在你的训练循环内部 ---
# for inputs, labels in train_loader:
#     optimizer.zero_grad()   # 清零梯度
#
#     outputs = model(inputs) # 前向传播（输出为 logits）
#
#     loss = criterion(outputs, labels) # 计算损失
#
#     loss.backward()         # 反向传播（计算梯度）
#     optimizer.step()        # 更新权重
# --- 训练循环代码片段结束 ---
```

在这段代码中，`criterion(outputs, labels)` 直接计算损失。`outputs` 是模型输出的原始 logits，`labels` 是整数类别索引。这个 `loss` 张量随后通过 `loss.backward()` 推动反向传播 (backpropagation)过程。

理解这些细节，特别是关于 logits 以及 `CrossEntropyLoss` 和 `BCEWithLogitsLoss` 等某些 PyTorch 损失函数中的组合操作，将使你从 TensorFlow Keras 的迁移过程更加平滑，并帮助你避免常见错误。通过明确定义和使用损失函数，你可以获得 PyTorch 设计特有的更细致的控制能力。

## 参考资料

- [Loss Functions (torch.nn)](https://pytorch.org/docs/stable/nn.html#loss-functions) — PyTorch Developers (2024)
  Publisher: PyTorch Foundation
  PyTorch `torch.nn` 模块的官方文档，详细介绍了可用的损失函数、它们的构造函数、输入要求和归约选项。它是理解 PyTorch 损失方法的主要资源。
- [tf.keras.losses](https://www.tensorflow.org/api_docs/python/tf/keras/losses) — TensorFlow Developers (2024)
  TensorFlow Keras 损失函数的官方 API 文档，描述了不同的损失类别、它们的参数（例如 `from_logits`）以及在 Keras 框架中的典型用法。有助于与 PyTorch 实现进行比较。
- [Deep Learning](https://www.deeplearningbook.org/) — Ian Goodfellow, Yoshua Bengio, and Aaron Courville (2016)
  Publisher: MIT Press
  一本基础教科书，解释了各种损失函数的理论基础，包括交叉熵和均方误差，以及处理 logits 时数值稳定性的重要性。它提供了坚实的理论基础。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  本书提供了在 PyTorch 中实现深度学习模型的实用视角，包括关于如何正确使用不同损失函数并将其集成到自定义训练循环中的详细讨论和示例。

---

[上一节](01-%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%BC%8F%EF%BC%9ATensorFlow%20%E7%9A%84%20fit%20%E6%96%B9%E6%B3%95%E4%B8%8E%20PyTorch%20%E8%AE%AD%E7%BB%83%E5%BE%AA%E7%8E%AF.md) · [下一节](03-%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95%EF%BC%9ATensorFlow%20%E5%92%8C%20PyTorch%20%E4%BC%98%E5%8C%96%E5%99%A8.md)
