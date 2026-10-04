---
course: "practical-llm-quantization"
chapter: "quantization-aware-training-qat"
lesson: "implementing-qat-frameworks"
sourceId: 4643
sourceUrl: "https://apxml.com/zh/courses/practical-llm-quantization/chapter-4-quantization-aware-training-qat/implementing-qat-frameworks"
title: "使用深度学习框架实现量化感知训练"
description: "讨论如何使用标准的深度学习框架（如 PyTorch、TensorFlow）来执行 QAT。"
order: 4
plots: []
sourceHash: "f2693959a9b0de59164865454c6cad0ea4d573bc12a26dbe25588d5e98ca485d"
sourceCorrections: []
---

使用深度学习 (deep learning)框架实现量化 (quantization)感知训练 (QAT) 涉及在训练期间模拟量化并利用直通式估计器 (STE)。主流深度学习框架为此过程提供了工具。例如，PyTorch 和 TensorFlow（通过 TensorFlow 模型优化工具包）提供了 API，可以自动化大部分实现过程。这些 API 使开发者能够专注于训练本身，而不是手动实现伪量化和梯度处理。

### 在 PyTorch 中实现 QAT

PyTorch 提供了一个专门的 `torch.quantization` 模块来方便 QAT。一般流程包含为 QAT 准备模型、进行微调 (fine-tuning)，然后将其转换为真正的量化 (quantization)模型。

1. **模型准备：**

   - 您需要定义一个 `QConfig`，它指定量化设置（例如，用于激活统计的观测器、用于权重 (weight)和激活的伪量化模块、目标数据类型如 `torch.qint8`）。
   - 在您想要量化的模型部分的开头和结尾插入 `QuantStub` 和 `DeQuantStub` 层。这些层作为标记 (token)，告知框架量化操作的起点和终点。
   - 使用 `torch.quantization.prepare_qat` 根据提供的 `QConfig` 自动将伪量化模块和观测器插入到您的模型中。此函数会就地修改模型，或返回一个为 QAT 准备好的新模型实例。

   ```python
   import torch
   import torch.nn as nn
   import torch.quantization

   # 示例：一个简单模型
   class MyModel(nn.Module):
       def __init__(self):
           super().__init__()
           self.quant = torch.quantization.QuantStub() # 输入量化标记
           self.linear = nn.Linear(10, 20)
           self.relu = nn.ReLU()
           self.dequant = torch.quantization.DeQuantStub() # 输出去量化标记

       def forward(self, x):
           x = self.quant(x) # 对输入应用伪量化
           x = self.linear(x)
           x = self.relu(x)
           x = self.dequant(x) # 返回浮点数前对输出去量化
           return x

   # 1. 实例化浮点模型
   float_model = MyModel()
   float_model.train() # 将模型设置为训练模式以进行 QAT

   # 2. 定义 QConfig（支持 INT8 对称逐张量后端示例）
   #    根据目标硬件/后端需求进行调整。
   qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') # 或 'qnnpack' 等

   # 3. 为 QAT 准备模型
   prepared_model = torch.quantization.prepare_qat(float_model, {'': qconfig})

   print(prepared_model) # 观察插入的 FakeQuantize 模块
   ```
2. **微调：**

   - 像训练常规浮点模型一样训练 `prepared_model`。使用您的标准训练循环、损失函数 (loss function)和优化器。
   - 在前向传播过程中，伪量化模块根据观测器收集的统计数据模拟量化效果（钳制、舍入）。
   - 在反向传播 (backpropagation)过程中，STE 允许梯度通过模拟量化步骤回传，使模型权重能够适应量化过程。
   - 通常做法是从预训练 (pre-training)的浮点模型检查点开始 QAT，并以较小的学习率微调几个周期。

   ```python
   # 假设 'train_loader'、'criterion'、'optimizer' 已定义
   ```

# 训练循环代码示例

````
num_epochs_qat = 3 # 通常比初始训练的周期数少

for epoch in range(num_epochs_qat):
    prepared_model.train() # 确保模型处于训练模式
    for data, target in train_loader:
        optimizer.zero_grad()
        output = prepared_model(data)
        loss = criterion(output, target)
        loss.backward() # 梯度通过 STE 流经伪量化节点
        optimizer.step()
    # 如有需要，添加验证循环
```
````

3. **转换为量化模型：**
\* 微调后，将模型切换到评估模式（`prepared_model.eval()`）。
\* 使用 `torch.quantization.convert` 将经过 QAT 训练的模型转换为真正的量化模型。这会使用学到的参数 (parameter)，将伪量化模块和观测到的浮点模块（如 `nn.Linear`）替换为它们的基于整数的对应模块（如 `nn.quantized.Linear`）。

````
```python
# 转换前确保模型处于评估模式
prepared_model.eval()

# 将 QAT 模型转换为可部署的量化模型
quantized_model = torch.quantization.convert(prepared_model.cpu()) # 通常先转换为 CPU 模型

print(quantized_model) # 观察量化模块（例如，QuantizedLinear）

# 现在 'quantized_model' 可以保存并用于推理
# torch.save(quantized_model.state_dict(), "quantized_model.pth")
```
````

### 在 TensorFlow/Keras 中实现 QAT

TensorFlow 使用 TensorFlow 模型优化 (TF MOT) 工具包进行 QAT。该过程与 Keras API 紧密结合。

1. **模型准备：**

   - 从预训练 (pre-training)的 Keras 模型开始。
   - 使用 `tfmot.quantization.keras.quantize_model` 函数自动用量化 (quantization)模拟逻辑包装您的 Keras 模型层。此函数会在您打算量化的 Keras 层周围插入 `QuantizeWrapperV2` 层。

   ```python
   import tensorflow as tf
   import tensorflow_model_optimization as tfmot

   # 假设 'float_model' 是一个预训练的 tf.keras.Model

   # 应用 QAT 包装器
   quantize_model = tfmot.quantization.keras.quantize_model
   qat_model = quantize_model(float_model)

   # 编译 QAT 模型（训练前必需）
   # 使用您的标准优化器、损失、指标
   qat_model.compile(optimizer='adam',
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                     metrics=['accuracy'])

   qat_model.summary() # 观察 quantize_wrapper 层
   ```
2. **微调 (fine-tuning)：**

   - 使用标准 `model.fit()` 方法训练 `qat_model`，就像训练常规 Keras 模型一样。
   - `QuantizeWrapperV2` 层在前向和反向传播 (backpropagation)过程中处理量化模拟（隐式使用 STE）。
   - 同样，与初始训练相比，微调通常涉及较少的周期和较低的学习率。

   ```python
   # 假设 'train_dataset'、'validation_dataset' 是已准备好的 tf.data.Dataset 对象
   num_epochs_qat = 3 # 示例周期计数

   history = qat_model.fit(
       train_dataset,
       epochs=num_epochs_qat,
       validation_data=validation_dataset
   )
   ```
3. **转换为量化模型：**

   - 微调后，QAT 模型包含为量化调整的权重 (weight)，但它内部仍使用浮点运算和模拟的量化步骤。
   - 要获得适合部署（例如，在 TensorFlow Lite 支持的设备上）的真正量化模型，您需要使用 TensorFlow Lite 转换器转换 QAT 模型。该转换器识别 QAT 包装器并生成一个仅包含整数运算的模型。

   ```python
   # 将 QAT Keras 模型转换为 TensorFlow Lite 模型
   converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
   converter.optimizations = [tf.lite.Optimize.DEFAULT] # 启用包括量化在内的默认优化

   quantized_tflite_model = converter.convert()

   # 将量化模型保存到 .tflite 文件
   # with open('quantized_model.tflite', 'wb') as f:
   #     f.write(quantized_tflite_model)

   # 此 .tflite 模型对量化层使用整数算术
   ```

### 框架共同点和注意事项

- **抽象：** 两个框架都抽象掉了插入伪量化 (quantization)节点和实现 STE 的复杂性。您与更高级别的 API 交互。
- **配置：** 您通常需要指定量化参数 (parameter)，例如目标位宽（标准 QAT 通常为 8 位，尽管有时也可能更低）、对称量化与非对称量化，以及粒度（逐张量或逐通道）。这些通常捆绑到配置对象或方案中（PyTorch 中的 `QConfig`，TF MOT 中由 `quantize_model` 默认值或自定义方案隐式处理）。
- **微调 (fine-tuning)过程：** 核心训练循环基本保持不变，但它应用于包含量化模拟逻辑的修改后模型。从训练良好的浮点模型开始是标准做法。
- **最终转换：** QAT 微调后始终需要一个独立的步骤，将模拟模型转换为使用实际低精度整数算术的最终可部署模型。

以下图表说明了使用框架进行 QAT 的一般流程：

> 该流程显示了使用框架进行 QAT 时所涉的不同阶段：从浮点模型开始，为 QAT 准备它，通过模拟量化进行微调，最后将其转换为可部署的整数模型。

通过使用这些框架工具，您可以有效应用 QAT 来恢复 PTQ 期间损失的精度，尤其是在旨在大幅度模型压缩时。接下来的部分将更直接地比较 QAT 和 PTQ，并讨论成功实现 QAT 的实际注意事项。

## 参考资料

- [Quantization-Aware Training (QAT) with PyTorch](https://pytorch.org/docs/stable/quantization.html#quantization-aware-training) — PyTorch Documentation (2019)
  Publisher: PyTorch
  PyTorch官方文档提供了使用`torch.quantization`模块实现QAT的详细指导和示例，涵盖了模型准备、微调和转换的步骤。
- [Quantization-aware training with TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization/guide/quantization/quantization_aware_training) — TensorFlow (2024)
  Publisher: TensorFlow
  TensorFlow官方文档解释了如何使用TensorFlow模型优化工具包对Keras模型进行QAT，包括转换为TensorFlow Lite的流程。
- [Model Compression and Acceleration for Deep Neural Networks](https://link.springer.com/book/10.1007/978-981-15-5858-6) — Yingting Hu, Xiang Liu (2020)
  Publisher: Springer Nature Singapore; DOI: [10.1007/978-981-15-5858-6](https://doi.org/10.1007/978-981-15-5858-6)
  本书系统地介绍了多种技术，包括量化感知训练，旨在提高深度神经网络部署的效率。
