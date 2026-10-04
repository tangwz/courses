# TorchScript 在模型序列化方面的简介

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/torchscript-serialization-pytorch)

[返回章节目录](README.md) · [返回课程目录](../README.md)

当您将模型从 PyTorch 灵活的 Python 开发训练环境转为生产部署时，通常需要一种方法将模型序列化为独立于 Python、可优化且可移植的格式。这就是 TorchScript 的作用。可以将其看作一座桥梁，将您的动态 PyTorch 模型转换为更静态、图表般的表示形式，以便在各种环境中运行，包括 C++ 运行时、移动设备或 Python 可能不够理想或性能不足的服务器。

对于熟悉 TensorFlow 的人来说，TorchScript 的作用类似于 `tf.function` 将 Python 代码转换为 TensorFlow 图，以及 SavedModel 如何将此图与权重 (weight)一起打包以供部署。尽管 PyTorch 的即时执行（定义即运行）因其即时性和 Python 式的风格而非常适合研究和实验，但生产环境通常能从静态图表示所提供的优化和可移植性中获益。

### 什么是 TorchScript？

TorchScript 是 PyTorch 模型的一种中间表示 (IR)。它允许您创建可序列化和可优化的模型版本，这些版本不依赖于 Python 运行时。这意味着您可以在 Python 中定义模型，然后将其转换为 TorchScript 以便：

1. **保存和加载：** 将模型的架构和学习到的参数 (parameter)存储在一个单一、可移植的文件中。
2. **优化：** TorchScript 表示可以进行各种基于图的优化（如运算符融合），从而加快执行速度。
3. **部署：** 在没有 Python 解释器的环境中运行模型，例如使用 LibTorch 的 C++ 应用程序或在移动平台上。

TorchScript 本质上捕获了模型的计算图，使其更适用于这些训练后操作。

### 创建 TorchScript 模型：追踪与脚本化

有两种主要方法可以将您的 PyTorch `nn.Module` 转换为 TorchScript 模块：追踪和脚本化。

> 从 Python 定义的 `nn.Module` 到可部署的 TorchScript 模型，可通过追踪或脚本化完成。

#### 1. 使用 `torch.jit.trace` 进行追踪

追踪的工作原理是使用一些示例输入执行您的模型一次。PyTorch 会记录所有在这些输入流经模型时执行的操作，从而有效地“追踪”一条路径。这条被记录的操作序列随后形成了 TorchScript 图。

```python
import torch
import torch.nn as nn

# 一个简单的演示模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# 实例化模型
model = SimpleNet()
model.eval() # 设置为评估模式

# 提供一个示例输入张量
example_input = torch.randn(1, 10) # 批大小 1，10 个特征

# 追踪模型
try:
    traced_model = torch.jit.trace(model, example_input)
    print("模型追踪成功！")
    # 您可以检查“代码”（图表的 Python 类似表示）
    # print(traced_model.code)
except Exception as e:
    print(f"追踪过程中出错：{e}")

# 保存追踪的模型
traced_model_path = "traced_simple_net.pt"
traced_model.save(traced_model_path)
print(f"追踪的模型已保存到 {traced_model_path}")

# 加载追踪的模型
loaded_traced_model = torch.jit.load(traced_model_path)
print("追踪的模型加载成功。")

# 您现在可以使用加载的追踪模型进行推理
output = loaded_traced_model(example_input)
print("从加载的追踪模型输出：", output.shape)
```

**追踪的优点：**

- 使用简单：只需提供一个示例输入即可。
- 适用于控制流直接、静态的模型（即操作不根据输入值改变）。

**追踪的局限性：**

- 不捕获依赖数据的控制流：如果您的模型包含根据实际张量值行为不同的 `if` 语句或循环，追踪只会记录您提供的特定 `example_input` 所采用的路径。其他路径可能会被遗漏。例如，如果 `if` 条件取决于 `x.sum() > 0` 且您的示例输入使其为真，那么 `else` 分支将不会是追踪图的一部分。

#### 2. 使用 `torch.jit.script` 进行脚本化

另一方面，脚本化涉及 TorchScript 编译器直接分析您的 Python 源代码（特别是 `forward` 方法及其调用的任何函数或模块）。它将此 Python 代码转换为 TorchScript 中间表示。这种方法对于具有动态控制流的模型更有效。

您可以将 `@torch.jit.script` 作为装饰器应用于函数或整个 `nn.Module`。对于 `nn.Module`，它通常会编译 `forward` 方法以及您显式装饰或从 `forward` 中调用的任何其他方法。

```python
import torch
import torch.nn as nn

class ScriptableNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(ScriptableNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x: torch.Tensor, use_relu: bool) -> torch.Tensor:
        h_relu = torch.relu(self.linear1(x))
        # 控制流示例
        if use_relu:
            y_pred = self.linear2(h_relu)
        else:
            y_pred = self.linear2(self.linear1(x)) # 第一层输出不应用 ReLU
        return y_pred

# 实例化模型
script_model_instance = ScriptableNet(10, 20, 5)
script_model_instance.eval()

# 脚本化模型实例
try:
    scripted_model = torch.jit.script(script_model_instance)
    print("模型脚本化成功！")
    # 您可以检查生成的“代码”
    # print(scripted_model.code)
except Exception as e:
    print(f"脚本化过程中出错：{e}")


# 保存脚本化的模型
scripted_model_path = "scripted_net.pt"
scripted_model.save(scripted_model_path)
print(f"脚本化的模型已保存到 {scripted_model_path}")

# 加载脚本化的模型
loaded_scripted_model = torch.jit.load(scripted_model_path)
print("脚本化的模型加载成功。")

# 使用不同的控制流路径进行测试
example_input = torch.randn(1, 10)
output_with_relu = loaded_scripted_model(example_input, True)
output_without_relu = loaded_scripted_model(example_input, False)

print("带 ReLU 路径的输出：", output_with_relu.shape)
print("不带 ReLU 路径的输出：", output_without_relu.shape)
```

**脚本化的优点：**

- 捕获动态控制流：依赖张量值的 `if` 语句、循环和其他 Python 结构通常会得到保留。
- 对复杂模型更有效。

**脚本化的注意事项：**

- TorchScript 兼容性：您的 Python 代码必须符合 TorchScript 语言子集。并非所有 Python 功能都受支持。常见问题源于高度动态的 Python 功能或某些全局变量的使用。
- 类型注解：虽然并非总是严格要求，但提供类型提示（例如 `x: torch.Tensor`）可以显著帮助 TorchScript 编译器理解您的代码，这通常是一个好的做法。

您也可以脚本化单独的函数：

```python
@torch.jit.script
def custom_activation(input_tensor: torch.Tensor) -> torch.Tensor:
    if input_tensor.mean() > 0:
        return torch.relu(input_tensor)
    else:
        return torch.sigmoid(input_tensor)

example_tensor = torch.randn(5)
print(custom_activation(example_tensor))
example_tensor_neg_mean = torch.tensor([-1.0, -2.0, -0.5])
print(custom_activation(example_tensor_neg_mean))
```

### 使用 TorchScript 的好处

一旦您的模型采用 TorchScript 格式，您将获得多项优势：

- **序列化和可移植性：** `traced_model.save()` 或 `scripted_model.save()` 保存的 `.pt` 文件包含模型的架构（以图的形式）及其参数 (parameter)（权重 (weight)和偏置 (bias)）。这个单一文件可以在其他 Python 环境中加载，或者更重要地，在使用 LibTorch 的 C++ 等非 Python 环境中加载。这对于部署不希望或无法提供完整 Python 堆栈的模型来说非常重要。
- **性能优化：** TorchScript 允许 PyTorch 执行图级别优化。例如，它可以将多个操作融合到一个更高效的内核中，从而可能加快推理 (inference)速度。它还减少了模型核心逻辑执行期间 Python 解释器的开销。
- **框架独立性（在某种程度上）：** 尽管它仍然是 PyTorch 格式，但序列化的模型不依赖于定义它的原始 Python 源代码。

### 何时选择追踪与脚本化

- **从追踪开始：** 如果您的模型是标准的 `nn.Sequential` 或具有非常线性、静态的操作流，`torch.jit.trace` 通常是获取 TorchScript 模型最快捷、最简单的方法。
- **复杂模型转向脚本化：** 如果您的模型使用依赖于输入数据的控制流（例如 `if x.sum() > 0: ... else: ...`），或者如果追踪未能捕获完整行为，`torch.jit.script` 是更可靠的方法。
- **混合方法：** 您也可以混合使用追踪和脚本化的模块。例如，一个复杂的子模块可以被脚本化，然后使用该子模块的更大模型可能会被追踪。

TorchScript 是一种功能强大的机制，可将您的 PyTorch 模型从研发阶段带入生产环境。通过将模型转换为这种可序列化和优化的格式，您可以使它们适用于更广泛的部署情况，从而摆脱 Python 解释器的限制，并实现与各种应用环境的集成。这是重要的一步，类似于 TensorFlow 开发者如何使用 SavedModels 来打包其训练好的模型以用于推理 (inference)引擎和服务平台。在接下来的部分中，我们将简要介绍 ONNX 以实现更广泛的互操作性，以及 TorchServe 用于部署这些模型。

## 参考资料

- [TorchScript Reference](https://pytorch.org/docs/stable/jit.html) — PyTorch Contributors (2018)
  Publisher: PyTorch Foundation
  提供了TorchScript设计、追踪、脚本化和部署能力的详细解释，是其使用的主要资源。
- [Save and load Keras models](https://www.tensorflow.org/guide/saved_model) — TensorFlow Team (2024)
  通过描述SavedModel在TensorFlow生态系统中如何实现模型序列化和部署，为TensorFlow用户提供了对比视角。
- [PyTorch C++ Frontend (LibTorch) Documentation](https://pytorch.org/cppdocs/frontend.html) — PyTorch Contributors (2024)
  详细介绍了使用LibTorch在C++应用程序中加载和执行TorchScript模型的过程，这是一个重要的部署目标。

---

[上一节](04-%E5%9C%A8PyTorch%E4%B8%AD%E6%9F%A5%E7%9C%8B%E6%A8%A1%E5%9E%8B%E6%9E%B6%E6%9E%84%E5%92%8C%E6%9D%83%E9%87%8D.md) · [下一节](06-%E4%BD%BF%E7%94%A8%20ONNX%20%E5%AE%9E%E7%8E%B0%E6%A1%86%E6%9E%B6%E4%BA%92%E6%93%8D%E4%BD%9C%E6%80%A7.md)
