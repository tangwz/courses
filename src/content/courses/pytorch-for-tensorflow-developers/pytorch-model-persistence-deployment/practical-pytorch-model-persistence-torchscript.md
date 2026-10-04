---
course: "pytorch-for-tensorflow-developers"
chapter: "pytorch-model-persistence-deployment"
lesson: "practical-pytorch-model-persistence-torchscript"
sourceId: 5773
sourceUrl: "https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/practical-pytorch-model-persistence-torchscript"
title: "动手实践：模型持久化与TorchScript基础"
description: "实践以不同方式保存和加载PyTorch模型，并尝试TorchScript转换。"
order: 8
plots: []
sourceHash: "203bb4c66d219b8292211307a57828381e70e11d1a73a074c20d544daf116f03"
sourceCorrections: []
---

通过编程练习，演示如何保存和加载模型状态、管理检查点，以及将模型转换为TorchScript以实现更灵活的部署。一个简单的神经网络 (neural network)将用作测试实例，以便能侧重于模型持久化和序列化的运作原理。

首先，让我们设置环境并定义将在这些示例中使用的模型。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import os

# 定义一个目录来保存模型文件
MODEL_SAVE_DIR = "saved_models_practice"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# 定义一个简单的神经网络
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

# 辅助函数，用于打印模型参数（以供验证）
def print_model_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data.numpy().sum():.4f}") # 打印权重的总和，以求简洁
```

这个`SimpleNet`是一个基本的两层全连接网络。我们将用它来演示各种保存和加载技术。

### 保存和加载模型参数 (parameter) (state\_dict)

持久化PyTorch模型的推荐方法是保存模型的`state_dict`。这个字典对象将每一层映射到其可学习参数（权重 (weight)和偏置 (bias)）。与保存整个模型对象相比，它更轻量，对代码更改也更可靠。

**保存`state_dict`**

让我们实例化模型并保存其参数。

```python
# 实例化模型
model_state_dict_example = SimpleNet()
print("Initial parameters (sum):")
print_model_parameters(model_state_dict_example)

# 定义保存state_dict的路径
STATE_DICT_PATH = os.path.join(MODEL_SAVE_DIR, "simplenet_state_dict.pth")

# 保存模型的state_dict
torch.save(model_state_dict_example.state_dict(), STATE_DICT_PATH)
print(f"\nModel state_dict saved to {STATE_DICT_PATH}")
```

`state_dict()` 方法返回一个包含所有权重和偏置的Python字典。然后，`torch.save()` 将这个字典序列化到磁盘上。`.pth` 扩展名是PyTorch模型文件的常见约定。

**加载`state_dict`**

要加载参数，首先需要模型结构的一个实例。然后，加载保存的`state_dict`并将其应用到这个模型实例上。

```python
# 创建一个新的模型实例
loaded_model_state_dict = SimpleNet()
print("\nParameters of new model instance (before loading, sum):")
print_model_parameters(loaded_model_state_dict)

# 加载已保存的state_dict
state_dict = torch.load(STATE_DICT_PATH)

# 将加载的state_dict应用到模型
loaded_model_state_dict.load_state_dict(state_dict)
print("\nParameters of model after loading state_dict (sum):")
print_model_parameters(loaded_model_state_dict)

# 如果您将模型用于推理，请记住调用 model.eval()
loaded_model_state_dict.eval()
```

请注意，加载后 `loaded_model_state_dict` 的参数与原始 `model_state_dict_example` 的参数一致。重要的是，您的脚本中定义的模型架构必须与保存 `state_dict` 时使用的架构相符。

### 保存和加载整个模型

PyTorch也允许您使用Python的`pickle`模块保存整个模型对象。虽然这很方便，但它的移植性可能较差，因为它将保存的文件与保存期间使用的特定类结构和目录路径绑定。如果您重构代码或移动文件，加载模型时可能会遇到问题。

**保存整个模型**

```python
# 实例化另一个模型
model_full_save_example = SimpleNet(input_size=10, hidden_size=8, output_size=3) # 不同的架构，以便清晰演示
model_full_save_example.fc1.weight.data.fill_(0.5) # 修改权重以便区分

print("\nParameters of model for full save (sum):")
print_model_parameters(model_full_save_example)

# 定义保存完整模型的路径
FULL_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "simplenet_full_model.pth")

# 保存整个模型
torch.save(model_full_save_example, FULL_MODEL_PATH)
print(f"\nEntire model saved to {FULL_MODEL_PATH}")
```

**加载整个模型**

加载完整保存的模型很简单，因为 `torch.load()` 直接返回模型对象。

```python
# 加载整个模型
loaded_full_model = torch.load(FULL_MODEL_PATH)
print("\nParameters of fully loaded model (sum):")
print_model_parameters(loaded_full_model)

# 设置为评估模式
loaded_full_model.eval()

# 您可以直接将其用于推理
dummy_input = torch.randn(1, 10) # 此模型的原始 input_size 为 10
# If loaded_full_model was SimpleNet(input_size=10, hidden_size=8, output_size=3)
# dummy_input should have 10 features
# output = loaded_full_model(dummy_input)
# print(f"\nOutput from fully loaded model: {output}")
```

这种方法更简单，但如前所述，对于长期存储或在不同项目或Python环境之间共享而言，其灵活性较差。`SimpleNet`的类定义必须在您加载模型的环境中可用且可访问。

### 训练期间的检查点

检查点（Checkpointing）是指在漫长的训练过程中，在不同时间点保存模型的状态（以及可能包括优化器状态和 epoch 数等其他训练信息）。这使得您可以在训练中断时恢复训练，或者将模型恢复到表现最佳的状态。

让我们模拟一个基本的检查点机制。

```python
# 模拟训练循环的设置
model_for_checkpointing = SimpleNet()
optimizer = optim.Adam(model_for_checkpointing.parameters(), lr=0.001)
num_epochs_mock = 5
current_epoch = 0

CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "simplenet_checkpoint.pth")

print("\nSimulating training and checkpointing...")
for epoch in range(num_epochs_mock):
    current_epoch = epoch
    # 模拟一些训练（例如，在此示例中手动更新权重）
    for param in model_for_checkpointing.parameters():
        if param.requires_grad:
            param.data += 0.01 * (epoch + 1) # 简单修改

    mock_loss = 1.0 / (epoch + 1)
    print(f"Epoch {epoch+1}, Mock Loss: {mock_loss:.4f}")

    # 每2个epoch保存一个检查点
    if (epoch + 1) % 2 == 0:
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model_for_checkpointing.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mock_loss,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch + 1} to {CHECKPOINT_PATH}")

print("\nSimulated training finished.")
print("Parameters of model after simulated training (sum):")
print_model_parameters(model_for_checkpointing)
```

现在，让我们看看如何从这个检查点恢复。

```python
# 要恢复训练，请创建模型和优化器的新实例
resumed_model = SimpleNet()
resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=0.001) # 确保优化器参数一致

print("\nParameters of new model before loading checkpoint (sum):")
print_model_parameters(resumed_model)

# 加载检查点
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)

    resumed_model.load_state_dict(checkpoint['model_state_dict'])
    resumed_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    last_loss = checkpoint['loss']

    print(f"\nCheckpoint loaded. Resuming from epoch {start_epoch}.")
    print("Parameters of model after loading checkpoint (sum):")
    print_model_parameters(resumed_model)

    # 如果您打算继续训练，请将模型设置为训练模式
    resumed_model.train() 
    # 或者如果用于推理，则设置为评估模式
    # resumed_model.eval()
else:
    print("\nNo checkpoint found to resume from.")
```

保存优化器的`state_dict`很重要，因为它包含在训练期间更新的缓冲区和参数 (parameter)（如学习率或动量值）。

### 模型部署的TorchScript简介

TorchScript提供了一种方法来创建PyTorch模型的可序列化和可优化的表示，这些表示可以独立于Python运行，例如在C++环境中或在Python开销不理想的场景中。将PyTorch模型转换为TorchScript有两种主要方法：追踪（tracing）和脚本化（scripting）。

- **追踪（Tracing）（`torch.jit.trace`)**: 您提供一个示例输入，TorchScript会记录该输入通过模型时执行的操作。这非常适用于具有直接、数据无关控制流的模型。
- **脚本化（Scripting）（`torch.jit.script`)**: TorchScript直接分析您模型的Python源代码（包括`if`语句和循环等控制流），并将其转换为TorchScript中间表示。这更适合具有复杂控制流的模型。

对于本次实践，我们将侧重于追踪，因为它通常更容易上手。

**追踪模型**

```python
# 实例化一个用于追踪的模型
model_to_trace = SimpleNet()
model_to_trace.eval() # 重要：将模型设置为评估模式以便追踪

# 创建一个形状正确的示例输入张量
# 我们的SimpleNet期望 input_size 为 10
example_input = torch.randn(1, 10) # 批大小为 1，10 个特征

# 追踪模型
try:
    traced_model = torch.jit.trace(model_to_trace, example_input)
    print("\nModel successfully traced.")

    # 您可以检查追踪模型的图（可选）
    # print(traced_model.graph) 
    # 及其代码（可选）
    # print(traced_model.code)

except Exception as e:
    print(f"\nError during tracing: {e}")
    traced_model = None
```

`traced_model`现在是一个`torch.jit.ScriptModule`对象。它已经捕获了`model_to_trace`在给定`example_input`时执行的操作序列。

**保存追踪模型**

追踪模型有自己的保存方法。

```python
TRACED_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "simplenet_traced.pt") # .pt 是 TorchScript 的常见扩展名

if traced_model:
    traced_model.save(TRACED_MODEL_PATH)
    print(f"Traced model saved to {TRACED_MODEL_PATH}")
```

**加载和使用追踪模型**

您可以使用`torch.jit.load()`加载TorchScript模型。一个重要优势是，您不需要原始的Python模型类定义（在本例中是`SimpleNet`）即可加载和运行追踪模型。

```python
if os.path.exists(TRACED_MODEL_PATH):
    loaded_traced_model = torch.jit.load(TRACED_MODEL_PATH)
    print("\nTraced model loaded successfully.")

    # 您现在可以使用加载的追踪模型进行推理
    # 确保输入张量具有正确的形状和类型
    test_input = torch.randn(1, 10) 
    with torch.no_grad(): # 对推理而言，这始终是一个良好的实践
        output = loaded_traced_model(test_input)

    print(f"Output from loaded traced model: {output.numpy()}")

    # 与原始模型验证输出（可选，如果可用）
    # original_output = model_to_trace(test_input)
    # print(f"Output from original Python model: {original_output.detach().numpy()}")

else:
    print("\nTraced model file not found.")
```

这种无需原始Python代码即可运行的能力使得TorchScript模型具有高度可移植性，并适用于各种部署场景。

这些练习涵盖了模型持久化的基本技术以及TorchScript的介绍。当您处理更复杂的模型和部署需求时，您将在这些方面进一步学习。例如，您可能研究TorchScript脚本化用于具有动态控制流的模型，或关注ONNX以实现与其他机器学习 (machine learning)框架的互操作性。在您自己的模型上尝试这些方法，以巩固您的理解。

## 参考资料

- [Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — Matthew Inkawhich (2024)
  PyTorch官方教程，详细介绍了保存和加载模型参数、完整模型以及实现检查点的方法。
- [Introduction to TorchScript](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html) — PyTorch Contributors (2023)
  PyTorch官方教程，解释了TorchScript的基础知识，包括用于模型部署的追踪和脚本化。
- [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch) — Eli Stevens, Luca Antiga, Thomas Viehmann (2020)
  Publisher: Manning Publications
  一本全面介绍PyTorch的书籍，涵盖了模型持久化、序列化和部署概念。
