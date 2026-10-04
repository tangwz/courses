# 使用 ONNX 实现框架互操作性

来源：[原文](https://apxml.com/zh/courses/pytorch-for-tensorflow-developers/chapter-5-pytorch-model-persistence-deployment/onnx-pytorch-tensorflow-interoperability)

[返回章节目录](README.md) · [返回课程目录](../README.md)

以下是根据英文更新内容同步修改后的完整中文文本：

尽管 PyTorch 的 `state_dict` 和 TorchScript 为在 PyTorch 生态系统内保存和部署模型提供了很好的机制，但有时你需要模型能在不同的机器学习 (machine learning)框架中运行。也许你的团队使用 TensorFlow 进行部署，或者你希望使用对通用模型格式有优化支持的特定硬件加速器。在这种情况下，开放神经网络 (neural network)交换 (ONNX) 格式就变得非常有用了。

ONNX 是一个用于表示机器学习模型的开放标准。它定义了一组通用的运算符（模型的基本组成部分，如卷积或矩阵乘法）和一种通用文件格式（`.onnx`）。目标是实现互操作性：你可以在一个框架中训练模型（如 PyTorch），将其导出为 ONNX 格式，然后在另一个框架（如 TensorFlow、Caffe2、MXNet）或专用的 ONNX 运行时中加载并运行它。

### 为什么需要 ONNX？

对于熟悉 TensorFlow 生态系统的开发者来说，该生态系统包括用于部署的 SavedModel 和用于移动设备的 TensorFlow Lite，ONNX 可能看起来是多余的一步。然而，它提供了多个优点：

1. **框架灵活性**：你不会被锁定在单一框架中。你可以借助 PyTorch 的动态性和对 Python 的友好性进行研究和开发，然后通过 ONNX 导出模型，以便在可能针对其他工具进行优化或标准化的环境中部署。
2. **硬件加速**：许多硬件供应商和芯片制造商为 ONNX 模型提供优化的运行时或编译器。导出到 ONNX 可以提升在特定 CPU、GPU 或专用 AI 加速器上的性能。
3. **模型共享与协作**：如果你与使用不同机器学习 (machine learning)框架的团队或个人合作，ONNX 提供了一种中立的中间表示形式，用于共享模型。
4. **使用更广泛的生态系统**：ONNX 生态系统包含多种用于模型可视化、优化和转换的工具。

可以把 ONNX 看作你的神经网络 (neural network)模型的通用翻译器。

> ONNX 工作流程：一个 PyTorch 模型被导出到 ONNX 格式，然后可以被各种运行时和工具使用，包括 TensorFlow 生态系统中的工具。

### 将 PyTorch 模型导出到 ONNX

PyTorch 使用 `torch.onnx.export()` 函数内置支持将模型导出到 ONNX 格式。此函数会跟踪你的模型，将其操作转换为 ONNX 图。

让我们看一下 `torch.onnx.export()` 的常见参数 (parameter)：

- `model`：你的 PyTorch 模型（`torch.nn.Module` 的实例）。
- `args`：模型期望的示例输入元组。此输入用于跟踪模型的执行路径。此示例输入的形状和数据类型很重要。
- `f`：ONNX 模型将要保存的路径（例如，`"my_model.onnx"`）。
- `input_names`：（可选）为 ONNX 图中的输入节点指定的名称列表。
- `output_names`：（可选）为 ONNX 图中的输出节点指定的名称列表。
- `dynamic_axes`：（可选）一个字典，指定输入/输出的哪些轴是动态的（例如，批大小、序列长度）。例如，`{'input_tensor': {0: 'batch_size'}, 'output_tensor': {0: 'batch_size'}}` 表示输入和输出的第一个维度都是动态的。
- `opset_version`：要使用的 ONNX 运算符集版本。ONNX 在发展，新版本增加了对更多运算符的支持。通常最好使用目标部署环境支持的、相对较新的版本。

这是一个导出基本 PyTorch 模型的简单例子：

```python
import torch
import torch.nn as nn
import torch.onnx

# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5) # 输入特征数：10，输出特征数：5

    def forward(self, x):
        return self.linear(x)

# 实例化模型
model = SimpleModel()
model.eval() # 将模型设置为评估模式

# 创建与模型预期输入形状匹配的虚拟输入
# 批大小为 1，10 个输入特征
dummy_input = torch.randn(1, 10)

# 为 ONNX 模型中的输入和输出定义名称，以提高清晰度
input_names = ["input_tensor"]
output_names = ["output_tensor"]

# 导出模型
torch.onnx.export(model,
                  dummy_input,
                  "simple_model.onnx",
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=14,
                  dynamic_axes={'input_tensor': {0: 'batch_size'},
                                'output_tensor': {0: 'batch_size'}})

print("模型已导出到 simple_model.onnx")
```

当你运行此代码时，PyTorch 会使用 `dummy_input` 执行你的 `SimpleModel`，记录操作，并将其转换为 ONNX 格式，将结果保存为 `simple_model.onnx`。`dynamic_axes` 参数特别重要。没有它，导出的 ONNX 模型将期望固定批大小（本例中为 1）的输入。通过指定 `dynamic_axes`，我们告诉 ONNX 导出器批处理维度可以变化。

### 使用 ONNX 模型

一旦你有了 `.onnx` 文件，就需要一个运行时来执行它。最常见的是 **ONNX Runtime**，它是一个用于 ONNX 模型的开源、高性能推理 (inference)引擎。它是跨平台的，并支持硬件加速。

你可以通过 pip 安装 ONNX Runtime：
`pip install onnxruntime`

下面是如何加载并运行我们刚刚创建的 `simple_model.onnx`：

```python
import onnxruntime
import numpy as np

# 创建一个 ONNX Runtime 推理会话
ort_session = onnxruntime.InferenceSession("simple_model.onnx")

# 准备一个示例输入（必须与模型的预期输入匹配，包括批大小）
# 在此次推理运行中使用批大小为 3
input_data = np.random.randn(3, 10).astype(np.float32)

# 从模型获取输入名称（或使用我们在导出时定义的名称）
input_name = ort_session.get_inputs()[0].name

# 运行推理
ort_inputs = {input_name: input_data}
ort_outs = ort_session.run(None, ort_inputs)

# 输出是一个 numpy 数组列表
output_data = ort_outs[0]
print("输入形状:", input_data.shape)
print("输出形状:", output_data.shape)
print("输出数据（第一行）:", output_data[0])
```

此代码片段加载 ONNX 模型，准备一个 NumPy 输入数组，并使用 ONNX Runtime 执行推理。你会注意到我们可以使用与导出时 `dummy_input` （1）不同的批大小（3），这得益于 `dynamic_axes`。

### 与 TensorFlow 的互操作性

作为一名 TensorFlow 开发者，你可能会想 ONNX 如何适应你现有的工作流程。如果你收到一个 ONNX 模型（可能从 PyTorch 导出），你可以将其转换为 TensorFlow 格式（如 SavedModel 或 TFLite），以便将其集成到基于 TensorFlow 的部署流程中（例如 TensorFlow Serving）。

目前推荐且积极维护的工具是 `onnx2tf`：

```bash
pip install onnx2tf
```

你可以直接通过 Python 或其命令行界面进行转换：

```python
import onnx2tf

# 将 ONNX 模型转换为 TensorFlow SavedModel 格式
onnx2tf.convert(
    input_onnx_file_path="simple_model.onnx",
    output_folder_path="simple_model_tf_savedmodel",
    non_verbose=True
)

print("ONNX 模型已转换为 TensorFlow SavedModel 格式，位于 'simple_model_tf_savedmodel'")
```

或者，你也可以在命令行中进行转换：

```bash
onnx2tf -in simple_model.onnx -co simple_model_tf_savedmodel
```

这会创建一个标准的 TensorFlow SavedModel 目录，然后可以使用 `tf.saved_model.load()` 加载或通过 TensorFlow Serving 部署。

反之，如果你有 TensorFlow 模型，可以使用 `tf2onnx` 等工具将其转换为 ONNX（`pip install tf2onnx`）。这使得你可以将 TensorFlow 模型引入 ONNX 生态系统，可能供 PyTorch 或其他 ONNX 兼容工具使用，尽管本课程侧重于从 PyTorch 到 TensorFlow 的方向。

### 注意事项与推荐做法

尽管 ONNX 很强大，但请记住以下几点：

1. **运算符支持**：大多数常见的神经网络 (neural network)操作都有良好支持。然而，高度自定义或非常新的 PyTorch 操作在所选 ONNX `opset_version` 中可能没有直接的等效项。你可能需要简化模型或实现自定义 ONNX 运算符（这是一个高级话题）。请务必查看 ONNX 文档以了解支持的运算符。
2. **版本兼容性**：确保 PyTorch、ONNX、你选择的 ONNX `opset_version`、ONNX Runtime 版本以及任何转换工具（如 `onnx2tf`）之间的兼容性。不匹配可能导致错误或意想不到的行为。
3. **全面测试**：将模型导出到 ONNX 并可能转换为另一种格式后，务必将其预测结果与原始 PyTorch 模型进行严格对比测试。检查是否存在数值差异和性能问题。
4. **模型追踪**：`torch.onnx.export` 函数通过使用样本输入追踪模型来工作。如果你的模型具有基于输入数据变化的控制流（这对于旨在导出到 ONNX 的模型来说不太常见，但在 PyTorch 中可能出现），追踪可能无法捕捉到所有执行路径。前面讨论的 TorchScript，有时对于具有复杂控制流的模型，在导出到 ONNX 之前会更稳定。
5. **输入/输出命名**：在导出期间使用 `input_names` 和 `output_names` 会使 ONNX 模型在后续使用时更方便，尤其是在使用 ONNX Runtime 或转换工具时。

ONNX 在不同机器学习 (machine learning)框架之间搭建了一个重要桥梁。对于学习 PyTorch 的 TensorFlow 开发者来说，理解如何将 PyTorch 模型导出到 ONNX，为将这些模型集成到现有的以 TensorFlow 为中心的部署流程中，或运用更广泛的 ONNX 生态系统进行优化和执行提供了机会。

## 参考资料

- [Open Neural Network Exchange (ONNX)](https://onnx.ai/) — ONNX Community (2024)
  Publisher: Linux Foundation AI & Data Foundation
  ONNX（开放神经网络交换）格式的官方网站和文档，详细介绍了其规范、运算符和生态系统。
- [torch.onnx - PyTorch Documentation](https://pytorch.org/docs/stable/onnx.html) — PyTorch (2025)
  Publisher: PyTorch Foundation
  PyTorch 官方关于将模型导出为 ONNX 格式的文档，涵盖了 `torch.onnx.export` 函数及其参数。
- [ONNX Runtime](https://onnxruntime.ai/) — Microsoft and ONNX Runtime Community (2024)
  ONNX Runtime 的官方网站和文档，这是一个用于在各种硬件和平台上执行 ONNX 模型的高性能推理引擎。

---

[上一节](05-TorchScript%20%E5%9C%A8%E6%A8%A1%E5%9E%8B%E5%BA%8F%E5%88%97%E5%8C%96%E6%96%B9%E9%9D%A2%E7%9A%84%E7%AE%80%E4%BB%8B.md) · [下一节](07-TorchServe%20PyTorch%20%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1%E6%A6%82%E8%A7%88.md)
