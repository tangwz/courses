# 模型导出为 ONNX 格式

来源：[原文](https://apxml.com/zh/courses/advanced-pytorch/chapter-4-deployment-performance-optimization/exporting-onnx)

[返回章节目录](README.md) · [返回课程目录](../README.md)

虽然 TorchScript 在 PyTorch 生态系统中提供了一种序列化 PyTorch 模型的方法，但为了实现更广泛的互操作性，通常需要一种标准化格式。开放神经网络 (neural network)交换 (ONNX) 格式满足了这一要求，它定义了一个开放标准来表示机器学习 (machine learning)模型。将您的 PyTorch 模型导出为 ONNX 可以使它们在多种平台和推理 (inference)引擎上运行，例如 ONNX Runtime、TensorRT、OpenVINO 以及各种移动/边缘设备，并且通常可以从这些运行时提供的硬件特定优化中获益。将 PyTorch 模型转换为 ONNX 格式的过程进行了详细说明。

### ONNX 在部署中的作用

ONNX 充当中间表示形式。您可以使用 PyTorch 灵活的环境训练模型，然后将训练好的模型图及其学习到的参数 (parameter)导出到 `.onnx` 文件。此文件随后可以由任何 ONNX 兼容的运行时加载和执行。这种解耦显著简化了部署过程，因为您不需要在每个目标部署系统上都安装 PyTorch。它还通过允许专用运行时应用图优化并更有效地使用加速器，从而提升了性能，这可能比通用框架更有效。

> 工作流程示意 PyTorch 模型导出到 ONNX 以及随后在各种推理 (inference)运行时中的部署。

### 使用 `torch.onnx.export` 导出模型

PyTorch 在 `torch.onnx` 模块中提供了 `torch.onnx.export()` 函数作为此转换的主要工具。此函数的核心功能通常是利用追踪来记录当样本输入通过模型时执行的操作，并将这些操作转换为其 ONNX 等效项。

该函数签名有几个重要参数 (parameter)：

```python
torch.onnx.export(
    model,               # 要导出的模型 (torch.nn.Module)
    args,                # 用于追踪的模型输入元组
    f,                   # 输出路径（字符串）或类文件对象
    export_params=True,  # 在文件中存储训练过的参数
    opset_version=None,  # ONNX 算子集版本
    do_constant_folding=True, # 执行常量折叠优化
    input_names=None,    # ONNX 图中输入节点名称列表
    output_names=None,   # ONNX 图中输出节点名称列表
    dynamic_axes=None    # 指定动态维度的字典
    # ... 其他参数
)
```

参数

- `model`：您的 `torch.nn.Module` 实例。如果模型在训练和推理 (inference)之间的行为（例如 dropout 或批归一化 (normalization)）有所不同，请确保其处于评估模式 (`model.eval()`)。
- `args`：一个元组，包含具有正确数据类型和形状的示例输入，这些是您的模型 `forward` 方法所期望的。此输入用于追踪执行路径。重要地，`args` 中的形状定义了导出的 ONNX 图中的输入形状，除非使用了 `dynamic_axes`。
- `f`：`.onnx` 模型将要保存的文件路径。
- `export_params`：如果为 `True`（默认），模型的训练权重 (weight)将直接嵌入 (embedding)到 ONNX 文件中，使其成为自包含文件。
- `opset_version`：指定要使用的 ONNX 算子集版本。不同的版本支持不同的算子集和功能。选择正确的 opset 对于与目标推理运行时的兼容性很重要。请查阅目标运行时的文档以获取支持的 opset。常见选择介于 11 到 17 之间，但新版本会定期发布。
- `input_names` / `output_names`：可选的字符串列表，为 ONNX 图中的输入和输出节点提供有意义的名称。这提高了可读性，并使得在运行时使用 ONNX 模型时更容易提供数据和获取结果。
- `dynamic_axes`：这是处理可变输入/输出形状的一个非常重要的参数。

### 处理动态形状

追踪本质上会捕获所提供 `args` 的特定形状。如果您的模型需要处理不同维度的输入（例如，NLP 模型中的可变批大小或序列长度），您必须使用 `dynamic_axes` 参数 (parameter)来指定这一点。

`dynamic_axes` 是一个字典，其中键是前面定义的 `input_names` 或 `output_names`，值是另一个字典，将轴索引映射到描述性名称。例如，要指定名为 'input\_ids' 的输入的批大小（轴 0）和序列长度（轴 1）可以变化，您可以使用：

```python
dynamic_axes = {
    'input_ids': {0: 'batch_size', 1: 'sequence_length'}, # 输入轴定义
    'output_logits': {0: 'batch_size'}                  # 输出轴定义
}
```

这会告诉导出器不要将这些维度硬编码到图中，从而允许 ONNX 运行时处理沿这些指定轴具有不同大小的输入并生成输出。

### 导出实践示例

我们来导出一个简单的卷积模型。

```python
import torch
import torch.nn as nn
import torch.onnx

# 定义一个简单的 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 10) # 假设输入图像为 32x32

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = torch.flatten(x, 1) # 展平除批次维度外的所有维度
        x = self.fc1(x)
        return x

# 实例化模型并设置为评估模式
model = SimpleCNN()
model.eval()

# 创建与预期维度匹配的虚拟输入（批大小、通道、高度、宽度）
# 注意：这里批大小设置为 1，但我们会使其变为动态
dummy_input = torch.randn(1, 3, 32, 32, requires_grad=False)

# 定义输入和输出名称
input_names = ["input_image"]
output_names = ["output_logits"]

# 定义动态轴（使批大小动态化）
dynamic_axes_config = {
    'input_image': {0: 'batch_size'}, # 输入的可变批大小
    'output_logits': {0: 'batch_size'} # 输出的可变批大小
}

# 指定输出文件路径
onnx_model_path = "simple_cnn.onnx"

# 导出模型
torch.onnx.export(
    model,
    dummy_input,
    onnx_model_path,
    export_params=True,
    opset_version=12, # 选择合适的 opset 版本
    do_constant_folding=True,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes_config
)

print(f"模型已导出到 {onnx_model_path}")
```

### 常见的导出挑战

虽然 `torch.onnx.export` 对许多模型都适用，但您可能会遇到一些问题：

1. **不支持的 PyTorch 算子：** 并非每个 PyTorch 函数或模块在目标 ONNX `opset_version` 中都有直接的等效项。如果追踪器遇到不支持的操作，导出将会失败。解决方案包括：

   - 重写 PyTorch 代码以使用 ONNX 兼容的操作。
   - 选择一个可能支持该算子的不同（通常是更新的）`opset_version`。
   - 实现自定义 ONNX 算子（一个涉及 C++ 的高级主题）。
   - 如果操作仅在训练期间发生，请确保模型处于 `eval()` 模式。
2. **动态控制流：** 追踪难以处理依赖于数据的控制流（例如，条件或迭代次数取决于张量值的 `if` 语句或循环）。虽然 `torch.jit.script` 有时可以捕获此类逻辑，但将脚本化模型导出到 ONNX 也可能具有挑战性。通常需要简化控制流或使其与数据无关。
3. **Opset 兼容性：** 导出的 ONNX 模型必须使用目标推理 (inference)引擎（例如 ONNX Runtime）支持的 opset 版本。请务必查看运行时的文档以了解兼容的 opset。

### 验证导出模型

导出后，验证 ONNX 模型的正确性很重要。一种常见的方法是使用 `onnxruntime` 库：

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
ort_session = ort.InferenceSession(onnx_model_path)

# 准备输入数据（需要是 NumPy 数组）
# 创建一个不同批大小的输入以测试动态轴
test_input_np = np.random.randn(4, 3, 32, 32).astype(np.float32) # 批大小 = 4

# 运行推理
ort_inputs = {ort_session.get_inputs()[0].name: test_input_np}
ort_outputs = ort_session.run(None, ort_inputs)
onnx_result = ort_outputs[0]

# 与 PyTorch 输出进行比较（可选，但建议）
# 如果 ONNX Runtime 使用 CPU，请确保模型在 CPU 上以便直接比较
model.cpu()
dummy_input = torch.from_numpy(test_input_np)
with torch.no_grad():
    pytorch_result = model(dummy_input).numpy()

# 检查输出是否接近（允许存在潜在的微小数值差异）
if np.allclose(pytorch_result, onnx_result, rtol=1e-03, atol=1e-05):
    print("验证成功：ONNX Runtime 输出与 PyTorch 输出一致。")
else:
    print("验证失败：输出不一致。")
    # 可能需要进一步调试
```

此验证步骤有助于确保转换过程没有引入错误，并且模型在目标运行时环境中表现符合预期，至少在数值上是这样。

导出到 ONNX 是一种有用的技术，可以使您的先进 PyTorch 模型可移植，并为在各种硬件和软件平台上的高效部署做好准备。掌握这一过程，包括处理动态形状和解决常见问题，是使您的深度学习 (deep learning)应用程序投入生产的重要一步。

## 参考资料

- [\`torch.onnx\` - PyTorch Documentation](https://pytorch.org/docs/stable/onnx.html) — PyTorch Contributors (2024)
  Publisher: PyTorch Foundation
  关于将 PyTorch 模型导出到 ONNX 格式的指南，包含 `torch.onnx.export` 参数和示例。
- [Open Neural Network Exchange (ONNX)](https://onnx.ai/) — ONNX Community (2024)
  Publisher: LF AI & Data Foundation
  ONNX 格式的官方资源，详细介绍了其标准、优势和生态系统。
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/) — Microsoft (2024)
  Publisher: Microsoft
  ONNX Runtime 用于模型推理的文档，包括会话创建和输入/输出处理。

---

[上一节](05-%E9%80%9A%E8%BF%87%E5%A4%96%E9%83%A8%E5%BA%93%E4%BC%98%E5%8C%96%E7%AE%97%E5%AD%90.md) · [下一节](07-%E4%BD%BF%E7%94%A8%20TorchServe%20%E6%8F%90%E4%BE%9B%E6%A8%A1%E5%9E%8B%E6%9C%8D%E5%8A%A1.md)
