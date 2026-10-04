---
course: "advanced-tensorflow"
chapter: "model-deployment-optimization"
lesson: "converting-models-tflite"
sourceId: 2933
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/converting-models-tflite"
title: "为 TF Lite 转换模型"
description: "了解将 TensorFlow SavedModels 或 Keras 模型转换为 TF Lite 格式的过程。"
order: 6
plots: []
sourceHash: "bddbd83d5446b045315ad50d34d0f3949813d3598c5c40c9f18949639439826c"
sourceCorrections: []
---

在边缘设备或移动应用上部署已训练和验证的TensorFlow模型，通常需要将其转换为TensorFlow Lite（`.tflite`）格式。这种格式是一种专门且优化的表示形式，适用于资源受限硬件上的低延迟推理 (inference)。转换过程将标准TensorFlow模型（通常为SavedModel格式）转换为包含模型图结构和权重 (weight)的FlatBuffer文件，可供TF Lite解释器使用。

### TFLiteConverter 的作用

TensorFlow 提供 `tf.lite.TFLiteConverter` 类作为此转换的主要工具。这个 Python API 充当训练好的 TensorFlow 模型与 TF Lite 运行时环境之间的桥梁。它处理分析模型图、应用所需优化并将结果序列化为 `.tflite` 格式的复杂过程。

转换器支持多种输入模型格式：

1. **SavedModel：** 这是推荐的格式，因为它包含完整的TensorFlow程序，包括图定义和权重 (weight)。它为转换提供了最直接的起点。
2. **Keras 模型 (H5)：** 您也可以直接从 `tf.keras.Model` 对象或保存的 Keras H5 文件进行转换。
3. **具体函数：** 为了更精细的控制，您可以转换表示模型计算部分或全部的特定 `tf.function` 装饰的 Python 函数（具体函数）。

使用 SavedModel 格式通常更受推荐，因为它能捕获模型的结构，包括任何 `tf.function` 装饰器和签名，从而确保更可靠的转换过程，特别是对于具有复杂控制流或自定义组件的模型。

### 基本转换流程

最简单的转换是将浮点 SavedModel 转换为浮点 `.tflite` 模型，不进行任何额外优化。

假设您已将训练好的模型保存为 SavedModel 格式在 `saved_model_dir` 目录中。使用 Python API 的转换过程如下所示：

```python
import tensorflow as tf

# 定义 SavedModel 目录的路径
saved_model_dir = 'path/to/your/saved_model'

# 从 SavedModel 创建 TFLiteConverter 实例
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# 执行转换
tflite_model = converter.convert()

# 将转换后的模型保存为 .tflite 文件
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

print("TensorFlow Lite 模型已保存为 model.tflite")
```

这段代码片段使用 `from_saved_model` 初始化转换器，调用 `convert()` 方法，然后将生成的字节流写入名为 `model.tflite` 的文件。现在这个文件包含了您的模型，可供 TensorFlow Lite 解释器部署。

> 转换过程接收 SavedModel 作为输入，使用 TFLiteConverter，并输出一个 `.tflite` 文件。

### 转换期间应用优化

虽然基本转换可行，但 TensorFlow Lite 的主要优势通常来自能减少模型大小和加速推理 (inference)的优化，特别是量化 (quantization)。`TFLiteConverter` 允许您直接指定优化策略。

转换器的 `optimizations` 属性用于启用这些功能。它接受一个优化标志列表。最常用的标志是 `tf.lite.Optimize.DEFAULT`，它启用训练后量化（具体来说，默认是动态范围量化）。

```python
import tensorflow as tf

saved_model_dir = 'path/to/your/saved_model'

# 创建转换器
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# 启用默认优化（包括动态范围量化）
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# 执行转换
tflite_quant_model = converter.convert()

# 保存量化模型
with open('model_quant.tflite', 'wb') as f:
  f.write(tflite_quant_model)

print("量化后的 TensorFlow Lite 模型已保存为 model_quant.tflite")
```

这会生成一个 `.tflite` 模型，其中权重 (weight)被量化为 8 位整数，激活在运行时动态量化。这通常会导致模型大小大幅减小（大约 4 倍），同时对许多模型的精度影响很小，还可能加速兼容硬件上的推理。

#### 高级量化技术

除 `Optimize.DEFAULT` 之外，您可以应用更具体的量化策略：

1. **Float16 量化：** 这将权重和/或激活量化为 16 位浮点数。它将模型大小减半，并能在支持 float16 计算的硬件（如许多 GPU）上提供加速。

   ```python
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.target_spec.supported_types = [tf.float16]
   ```
2. **全整数化：** 这种方法的目标是将整个模型（权重和激活）转换为仅使用整数运算（通常是 int8）运行。这通常能在 CPU 和 Edge TPU、DSP 等专用硬件上提供最佳性能，并大幅减小尺寸。但是，它需要一个*代表性数据集*进行校准。校准过程涉及对典型输入数据的小样本运行推理，以确定激活的动态范围。

   ```python
   import numpy as np

   # 假设 representative_dataset() 是一个生成输入样本的生成器
   def representative_dataset():
     # 替换为您的实际数据加载逻辑
     for _ in range(100):
       data = np.random.rand(1, 224, 224, 3) # 示例输入形状
       yield [tf.constant(data, dtype=tf.float32)]

   converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   # 提供代表性数据集生成器
   converter.representative_dataset = representative_dataset
   # 确保对不支持的操作使用仅整数回退（可选但常见）
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   # 将输入和输出类型设置为整数（可选，取决于模型）
   # converter.inference_input_type = tf.int8
   # converter.inference_output_type = tf.int8

   tflite_int8_model = converter.convert()

   with open('model_int8.tflite', 'wb') as f:
     f.write(tflite_int8_model)
   ```

   提供一个好的代表性数据集对于保持全整数量化的精度很重要。这个数据集应该反映模型在生产中将遇到的输入的分布和范围。
3. **带浮点回退的整数量化：** 这类似于全整数量化，但如果某些操作没有高效的整数实现，它允许转换器将其保留为浮点形式。如果全整数量化导致精度下降或因不支持的操作而失败，这通常是一个良好的平衡选择。从全整数示例中删除 `converter.target_spec.supported_ops = [...]` 行即可启用浮点回退。

优化策略的选择取决于目标硬件、性能要求以及可接受的精度权衡。它通常需要通过实验来找到最佳平衡。

### 处理自定义操作

如果您的 TensorFlow 模型使用了未包含在标准 TensorFlow Lite 内置操作集中的自定义操作（ops），转换过程将默认失败。您有两个主要选择：

1. **在 TF Lite 中实现自定义操作：** 提供与 TF Lite 运行时兼容的操作的自定义实现。这涉及编写 C++ 代码并将操作符注册到目标设备上的解释器。这是性能最高但也是最复杂的方法。
2. **使用选择性 TensorFlow 操作：** 配置转换器以在 TF Lite 模型中包含必要的 TensorFlow ops 实现（`tf.lite.OpsSet.SELECT_TF_OPS`）。这会增加解释器和模型文件的二进制大小，但允许直接使用原始 TensorFlow ops，从而简化转换，但会牺牲性能和可移植性。

```python
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

# 允许 TF 操作用于 TF Lite 不原生支持的操作
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # 启用默认 TF Lite 操作。
  tf.lite.OpsSet.SELECT_TF_OPS   # 启用包含选择性 TF 操作。
]

tflite_model_with_tf_ops = converter.convert()

with open('model_with_tf_ops.tflite', 'wb') as f:
  f.write(tflite_model_with_tf_ops)
```

为自定义操作选择正确的方法取决于具体操作、性能需求和可用的开发资源。

### 验证转换

转换后，建议使用 `tf.lite.Interpreter` API 加载 `.tflite` 模型，并在一些测试样本上运行推理 (inference)，以验证转换是否成功且输出符合预期，特别是在应用了量化 (quantization)等优化之后。

使用 `TFLiteConverter` 进行转换是准备 TensorFlow 模型在各种边缘平台上高效执行的一个根本步骤。通过了解不同的输入格式、优化策略以及处理自定义操作的选项，您可以创建高度优化的 `.tflite` 模型，以满足您的特定部署要求。下一节将介绍如何进一步优化这些模型，特别是针对设备上推理场景。

## 参考资料

- [Convert TensorFlow models to TensorFlow Lite](https://www.tensorflow.org/lite/convert) — TensorFlow Authors (2024)
  Publisher: Google
  此官方指南解释了tf.lite.TFLiteConverter API、支持的输入格式（SavedModel、Keras、Concrete Functions）以及基本转换流程。
- [Optimize your TensorFlow Lite model](https://www.tensorflow.org/lite/performance/model_optimization) — TensorFlow Authors (2024)
  Publisher: Google
  此官方指南详细介绍了TF Lite模型的各种优化技术，特别是训练后量化策略，如动态范围、float16和全整数量化，包括代表性数据集的使用。
