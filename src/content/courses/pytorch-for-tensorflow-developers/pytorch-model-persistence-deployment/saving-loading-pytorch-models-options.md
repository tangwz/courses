---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-model-persistence-deployment"
lesson: "saving-loading-pytorch-models-options"
sourceId: 5767
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/saving-loading-pytorch-models-options"
title: "保存和加载完整模型与仅保存参数"
description: "区分 PyTorch 中保存整个模型与仅保存学习到的参数（`state_dict`）的方法。"
order: 2
plots: []
sourceHash: "9d1183b49ae2783df078e00f21e2794016e2ac302c9ed62d817c0b9a757a1e0b"
sourceCorrections: []
---

保存 PyTorch 模型时，您通常有两种主要选择：保存整个模型对象，或仅保存其学习到的参数 (parameter)（`state_dict`）。`state_dict` 是 PyTorch 模型持久化的主要机制，它常与其他框架（例如 TensorFlow）的格式进行比较。了解 PyTorch 中这两种保存方法的实际区别和影响对于有效管理模型很重要，特别是在协作或将模型转移到不同环境时。

### 仅保存和加载模型参数 (parameter)（state\_dict）

PyTorch 推荐且最常见的做法是仅保存和加载模型的 `state_dict`。简单回顾一下，`state_dict` 是一个 Python 字典对象，它将每个层映射到其可学习参数（权重 (weight)和偏置 (bias)）。

**保存 `state_dict`**

要保存 `state_dict`，您可以通过 `model.state_dict()` 访问它，然后使用 `torch.save()`：

```python
import torch
import torch.nn as nn
import os

# 定义一个简单模型用于演示
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=5, output_size=2):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleNet()

# --- 保存 state_dict ---
STATE_DICT_PATH = "simple_net_state_dict.pth"
torch.save(model.state_dict(), STATE_DICT_PATH)
print(f"模型 state_dict 已保存到 {STATE_DICT_PATH}")
```

这会创建一个文件（例如 `simple_net_state_dict.pth`），仅包含您的 `SimpleNet` 模型的权重和偏置。

**加载 `state_dict`**

要将参数加载回模型中，首先需要实例化模型类的一个对象。然后，使用 `torch.load()` 从文件中加载 `state_dict`，并使用 `model.load_state_dict()` 将这些参数填充到模型实例中：

```python
# --- 加载 state_dict ---
# 首先，实例化模型结构
loaded_model_from_state_dict = SimpleNet() # 确保类定义可用

# 加载 state_dict
state_dict = torch.load(STATE_DICT_PATH)
loaded_model_from_state_dict.load_state_dict(state_dict)

# 如果您将模型用于推断，请记住调用 model.eval()
# 以将 dropout 和批归一化层设置为评估模式。
loaded_model_from_state_dict.eval()

print("模型已成功从 state_dict 加载。")
# 您现在可以将 loaded_model_from_state_dict 用于推断
# 例如：
# dummy_input = torch.randn(1, 10)
# output = loaded_model_from_state_dict(dummy_input)
# print("从加载模型得到的输出：", output)
```

**使用 `state_dict` 的优点：**

- **灵活性**：这是最灵活的方法。保存的 `state_dict` 不依赖于定义模型的精确 Python 代码，只要新模型的架构具有匹配的名称和参数形状的层。您可以重构模型类，将其移动到不同的文件，甚至将参数加载到略有不同的架构中（如果您仔细管理 `state_dict` 键）。
- **稳定性**：当 PyTorch 版本变化或发生细微的代码修改时，它更不容易损坏，因为它只序列化原始张量数据及其层关联。
- **透明性**：它明确表示您只恢复学习到的参数，而不是任意 Python 代码。这通常更安全。
- **标准做法**：这是 PyTorch 团队为大多数用例推荐的方法。

**使用 `state_dict` 的缺点：**

- 在加载 `state_dict` 之前，您必须拥有模型的类定义才能实例化模型。这意味着您需要访问定义模型架构的 Python 代码。

### 保存和加载整个模型对象

PyTorch 也允许您直接使用 `torch.save()` 保存整个模型对象。这种方法在后台使用 Python 的 `pickle` 模块来序列化模型对象本身。

**保存整个模型**

```python
# --- 保存整个模型对象 ---
ENTIRE_MODEL_PATH = "simple_net_entire_model.pth"
torch.save(model, ENTIRE_MODEL_PATH)
print(f"整个模型已保存到 {ENTIRE_MODEL_PATH}")
```

**加载整个模型**

加载很简单；`torch.load()` 直接返回模型对象：

```python
# --- 加载整个模型对象 ---
loaded_entire_model = torch.load(ENTIRE_MODEL_PATH)

# 请记住为推断调用 model.eval()
loaded_entire_model.eval()

print("整个模型已成功加载。")
# 您现在可以将 loaded_entire_model 用于推断
# 例如：
# output_entire = loaded_entire_model(dummy_input)
# print("从完整加载的模型得到的输出：", output_entire)
```

**保存整个模型的优点：**

- **语法简洁性**：保存和加载的代码显得更简洁，因为加载前无需单独实例化模型类。

**保存整个模型的缺点：**

- **脆弱性**：这种方法不那么稳固。序列化数据与保存模型时使用的特定 Python 类和目录结构紧密关联。如果您重构代码（例如，重命名类、移动定义类的文件）、更改 PyTorch 版本，或者尝试在没有完全相同代码结构的不同项目中加载模型，加载很容易失败。
- **安全风险**：由于它使用 `pickle`，从不可信来源加载以这种方式保存的模型可能存在安全风险，因为 `pickle` 可以执行任意代码。
- **可移植性问题**：这些文件在不同 Python 环境或 PyTorch 版本之间通常可移植性较差。

### 应该选择哪种方法？

对于大多数情况，特别是在共享模型、将其部署到生产环境或计划长期使用时，\*\*保存和加载 `state_dict` 是强烈推荐的方法。\*\*它提供了更大的灵活性、稳定性及安全性。

保存整个模型可能适用于：

- 在个人实验期间的快速、临时保存，且您确定环境和代码不会改变。
- 模型定义始终可用且不变的简单脚本。

但是，请注意其局限性和可能导致的问题。

### 与 TensorFlow 实践相关联

如果您是 TensorFlow 用户，这些 PyTorch 方法有一些对应的相似之处：

1. **保存/加载 `state_dict` (PyTorch)** 类似于 **保存/加载 TensorFlow 中的权重 (weight)（例如，`model.save_weights('my_weights.h5')` 和 `model.load_weights('my_weights.h5')`）**。在两种情况下，您仅保存学习到的参数 (parameter)，并且需要代码中定义的模型架构来恢复模型。要加载 TensorFlow 中的权重，您首先构建模型（例如 `model = create_my_model()`），然后调用 `model.load_weights()`。这与实例化您的 `nn.Module` 类然后调用 `model.load_state_dict()` 类似。
2. **保存/加载整个模型（PyTorch 的 `torch.save(model, PATH)`)** 可能最初看起来像 TensorFlow 的 `model.save('my_model.h5')` 或 `tf.saved_model.save(model, 'my_saved_model_dir')`。两者都旨在保存不仅仅是权重。然而，模型架构的持久化方式存在明显区别。

   - PyTorch 的 `torch.save(model, PATH)` 使用 Python 的 `pickle` 来序列化模型*对象*，包括其代码。这使得它在加载时依赖于精确的 Python 类定义和文件结构必须可用且相同。
   - 另一方面，TensorFlow 的 `SavedModel` 格式以一种更不依赖特定语言的方式将模型架构保存为计算图。这使得 `SavedModel` 通常更可靠，更适合部署和共享，因为它与原始 Python 代码结构的联系较少。旧的 Keras HDF5 格式（`model.save('my_model.h5')`）也保存架构，但 `SavedModel` 是现代 TensorFlow 中更推荐的格式。

PyTorch 针对类似 TensorFlow `SavedModel` 的基于图的序列化格式的解决方案是 **TorchScript**，您将在本章后面学习到它。TorchScript 允许您将 PyTorch 模型转换为可以独立于 Python 运行的中间表示，为部署提供更好的可移植性和性能。

通过理解 PyTorch 中保存和加载模型的这两种主要方法，以及它们各自的权衡，您能更好地管理训练好的模型。选择 `state_dict` 方法通常会带来更易于维护和共享的代码，符合 PyTorch 社区的最佳实践。

## 参考资料

- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Matthew Inkawhich (2024)
  Publisher: PyTorch Foundation
  提供了关于保存和加载 PyTorch 模型的官方指南，详细介绍了 `state_dict` 和整个模型的序列化，并推荐了最佳实践。
- [Save and load Keras models](https://www.tensorflow.org/guide/keras/save_and_serialize) — Neel Kovelamudi, Francois Chollet (2024)
  Publisher: TensorFlow
  解释了如何在 TensorFlow 中保存和加载 Keras 模型，包括仅保存权重和 `SavedModel` 格式，这对于与 PyTorch 进行比较很有意义。
- [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) — PyTorch Contributors (2023)
  介绍了 TorchScript 作为在 PyTorch 中创建可序列化和可优化模型的方法，类似于 TensorFlow 的 `SavedModel`，用于模型部署。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, and Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本全面介绍 PyTorch 实用方面的书籍，涵盖了模型的保存、加载以及模型持久化的最佳实践。
