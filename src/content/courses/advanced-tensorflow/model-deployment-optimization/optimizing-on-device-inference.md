---
course: "advanced-tensorflow"
chapter: "model-deployment-optimization"
lesson: "optimizing-on-device-inference"
sourceId: 2937
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/optimizing-on-device-inference"
title: "优化设备端推理"
description: "在资源受限设备上进一步优化TF Lite模型性能和大小的方法。"
order: 7
plots: ["plots/2937-0.json"]
sourceHash: "ac63481f6286b0d4ef646a361d91d78d876122054f339d5c45273a6103c145ab"
sourceCorrections: []
---

将TensorFlow模型转换为TensorFlow Lite（`.tflite`）格式是实现设备端部署的一个必要步骤。然而，虽然`.tflite`模型通常比原始SavedModel小，但转换后的模型对于移动、嵌入 (embedding)式或物联网硬件的严格限制而言，可能仍然过大或过慢。这些设备通常具有有限的算力 (compute)（CPU/DSP/NPU）、内存受限（RAM）、存储容量较小，并且常常依赖电池供电，使其对计算效率有极高要求。针对这些资源受限环境的`.tflite`模型优化方法，侧重于减小模型大小和提升推理 (inference)速度。

TF Lite生态系统中，设备端优化的主要工具是**量化 (quantization)**。

### 模型量化 (quantization)

量化是降低用于表示模型参数 (parameter)（权重 (weight)）以及在推理 (inference)时可选地表示其激活值的数字精度的方法。通常，模型使用32位浮点数（float32）进行训练。量化将这些数字转换为低精度类型，最常见的是8位整数（int8）或16位浮点数（float16）。

**为何进行量化？**

1. **模型大小减小：** 低精度类型需要更少存储空间。从float32转换为float16可使模型大小减半，而转换为int8通常可减小四倍。这对于存储受限的设备和减小下载大小很重要。
2. **推理速度加快：** 许多处理器，特别是智能手机或Edge TPU中发现的NPU（神经网络 (neural network)处理单元）等专用硬件，执行整数运算比浮点运算快得多。量化为int8可以带来显著的延迟改善（2倍-4倍或更多）。Float16在原生支持的硬件（如许多GPU）上也能提供加速。
3. **功耗降低：** 整数运算通常比浮点运算消耗更少电量，这对于电池供电设备很重要。

TensorFlow Lite提供几种量化策略，大致分为训练后量化和量化感知训练。

#### 训练后量化 (PTQ)

这是最常见且通常最简单的方法，因为它在模型已训练*后*对其进行优化。您只需要已训练的float32模型（通常是SavedModel或Keras H5文件）。

- **动态范围量化：**

  - **作用：** 仅将权重从float32量化为int8。激活值在推理时动态量化为int8，并在下一个操作前反量化回float32。
  - **优点：** 最简单的PTQ方法；不需要代表性数据集。在大小减小（权重约小4倍）和易用性之间取得了良好平衡。由于权重加载更小和潜在的int8计算，提供了一些性能提升。
  - **缺点：** 激活值的量化/反量化会增加开销。与全整数量化相比，延迟改善可能不那么显著。
  - **方法：** 转换时设置`converter.optimizations = [tf.lite.Optimize.DEFAULT]`。
- **Float16量化：**

  - **作用：** 将权重（以及可选的激活值）量化为float16。
  - **优点：** 模型大小减小50%。在原生支持float16的硬件（例如GPU）上可以提供加速。与int8量化相比，对模型精度影响最小。易于应用。
  - **缺点：** 速度优势完全取决于硬件支持。在侧重整数的硬件上，其大小减小或速度提升不如int8显著。
  - **方法：** 设置`converter.optimizations = [tf.lite.Optimize.DEFAULT]`和`converter.target_spec.supported_types = [tf.float16]`。
- **全整数量化：**

  - **作用：** 将权重和激活值都量化为int8。这使得整个模型推理可以潜在地仅使用整数运算来运行。
  - **优点：** 模型大小最大程度减小（约4倍）。延迟减小的潜力最大，尤其是在仅支持整数的硬件加速器（如Edge TPU或DSP）上。功耗降低。
  - **缺点：** 需要一个*代表性数据集*来校准激活值的量化范围。此数据集应反映模型在生产环境中将看到的典型输入。如果模型对精度变化敏感，与其他方法相比，可能导致更大的精度损失。
  - **方法：** 需要`converter.optimizations = [tf.lite.Optimize.DEFAULT]`，设置`converter.representative_dataset`（一个生成器函数，提供示例输入），并且通常需要设置`converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]`来强制执行仅整数操作。您可能还需要将`converter.inference_input_type`和`converter.inference_output_type`设置为`tf.int8`或`tf.uint8`。

  ```python
  import tensorflow as tf
  import numpy as np

  # 假设 'model' 是您训练好的Keras模型
  # 假设 'representative_dataset_generator' 产出代表性输入数据的批次

  # 定义代表性数据集生成器
  def representative_data_gen():
    # 示例：提供100个典型输入数据样本
    # 确保形状和类型与模型的输入签名匹配
    num_calibration_steps = 100
    for i, input_value in enumerate(representative_dataset_generator()):
      if i >= num_calibration_steps:
          break
      # 模型为单输入，如有多个输入请调整。必须是列表。
      yield [input_value.astype(np.float32)] # 确保校准输入为float32

  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_data_gen
  # 强制仅使用整数操作
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  # 设置输入/输出类型为整数
  converter.inference_input_type = tf.int8  # 或 tf.uint8，取决于模型/校准
  converter.inference_output_type = tf.int8 # 或 tf.uint8

  tflite_quant_model = converter.convert()

  # 保存量化模型
  with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)
  ```

  代表性数据集在这里很重要。它不需要标签，仅用于在真实数据流经模型时观察模型内部中间张量（激活值）的动态范围（最小值/最大值）。这使得转换器能够确定量化这些激活值的合适缩放因子。

> 使用在校准期间得出的比例因子和零点值，将浮点激活范围映射到8位整数范围。

#### 量化感知训练 (QAT)

有时，PTQ，特别是全整数量化，可能导致模型精度出现不可接受的下降。这是因为模型最初训练时未考虑低精度的限制。QAT通过在训练（或微调 (fine-tuning)）过程中模拟量化效果来解决此问题。

- **作用：** 使用TensorFlow模型优化工具包（`tfmot`）等工具修改您的Keras模型定义。它在图中插入“伪”量化节点。在训练期间，这些节点模拟int8在前向和反向传播 (backpropagation)中的精度损失。模型学习到的权重对量化效果更具弹性。
- **优点：** 与PTQ相比，量化模型通常能达到更高精度，有时可接近匹配原始float32的精度。
- **缺点：** 需要修改模型架构并重新训练或微调模型，这比PTQ计算成本更高且更复杂。
- **方法：** 在编译和训练/微调之前，使用`tfmot.quantization.keras.quantize_model`封装您现有的Keras模型。训练完成后，使用标准转换器将QAT模型转换为TF Lite；量化信息已嵌入 (embedding)模型中。

```python
import tensorflow_model_optimization as tfmot

# 假设 'model' 是您训练好的float32 Keras模型
quantize_model = tfmot.quantization.keras.quantize_model

# 应用QAT封装
q_aware_model = quantize_model(model)

# 编译并微调（或从头训练）
# 使用标准的compile/fit方法
q_aware_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

# q_aware_model.fit(...) # 使用训练数据进行微调

# 转换QAT模型（此处不需要代表性数据集）
converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # 转换器识别QAT模型

tflite_qaware_model = converter.convert()

# 保存模型
with open('model_qaware_int8.tflite', 'wb') as f:
    f.write(tflite_qaware_model)
```

### 选择合适的优化策略

1. **从简开始：** 从训练后动态范围或float16量化 (quantization)开始。这些方法易于应用，能立即带来大小优势并可能提供一些加速。评估精度下降情况。
2. **优先性能：** 如果延迟很重要且您的硬件支持高效的整数运算，请尝试训练后全整数量化。准备一个好的代表性数据集。仔细评估精度。
3. **恢复精度：** 如果全整数PTQ显著降低精度，请考虑量化感知训练。它需要更多精力（重新训练），但通常能为int8模型提供性能和精度的最佳平衡。
4. **考虑硬件：** 务必了解目标设备的性能。它是否有加速int8的NPU/DSP？它是否原生支持float16？针对特定硬件特性进行优化会带来最佳结果。



![TF Lite量化的影响](plots/2937-0.json)



> 比较模型在不同TF Lite量化方案下的模型大小和相对推理 (inference)延迟。INT8通常能在大小和延迟上提供最大幅度的减少，前提是硬件兼容。

### 其他优化考虑

- **权重 (weight)剪枝：** 尽管权重剪枝（将权重设置为零）主要是使用TensorFlow模型优化工具包（`tfmot.sparsity.keras`）在TF Lite转换*之前*应用的方法，但它能创建更稀疏的模型。这直接减小了需要量化 (quantization)和存储的权重大小。虽然TF Lite本身对基于非结构化稀疏性的自动推理 (inference)加速支持有限，但高度稀疏的模型压缩效果更好，有时可以通过专用硬件或自定义内核进行加速。
- **算子选择：** 确保您的模型主要使用TF Lite内置算子（`tf.lite.OpsSet.TFLITE_BUILTINS`或`TFLITE_BUILTINS_INT8`）。这些算子针对各种平台进行了高度优化。避免过度依赖TensorFlow Select算子（`tf.lite.OpsSet.SELECT_TF_OPS`），因为它们需要引入部分更大的TensorFlow运行时，这会增加二进制文件大小，并可能降低与原生TF Lite算子相比的性能。检查转换器日志中关于算子被转换为Flex算子的消息。

### 测量设备上性能

“理论上的好处是一回事；性能是另一回事。在实际目标硬件或非常接近的等效硬件上对优化后的`.tflite`模型进行**基准测试**绝对重要。”

- **使用TensorFlow Lite基准测试工具：** 这个命令行工具允许您在Android、Linux和其他平台上运行您的`.tflite`模型，提供初始化时间、推理 (inference)延迟（平均值、标准差）和内存使用（如果平台支持）的详细测量结果。
- **测量精度：** 使用量化 (quantization)后的`.tflite`模型和代表性测试数据集重新评估模型的精度。确保精度下降在您的应用程序可接受的范围内。比较float32模型和量化模型在样本数据上的输出，以了解任何差异的性质。

通过系统地应用量化方法并仔细测量目标硬件上的结果，您可以显著减小TensorFlow Lite模型的占用空间并提高速度，从而使复杂的机器学习 (machine learning)即使在最小的设备上也能实现。

## 参考资料

- [TensorFlow Lite Post-training quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) — TensorFlow Authors (2024)
  官方指南，解释TensorFlow Lite模型的训练后量化技术。
- [Quantization aware training overview](https://www.tensorflow.org/model_optimization/guide/quantization/training) — TensorFlow Authors (2024)
  使用TensorFlow模型优化工具包进行量化感知训练的官方指南。
- [Quantization and Training of Neural Networks for Efficient On-Device Inference](https://arxiv.org/abs/1712.05877) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Xu, Matthew Sandler, Andrew Howard, Andrew G. Howard, Hartwig Adam (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 1056-1065; DOI: [10.1109/CVPR.2018.00116](https://doi.org/10.1109/CVPR.2018.00116)
  介绍用于高效设备端推理的神经网络量化关键概念和方法的论文，与TensorFlow Lite相关。
- [Benchmark TensorFlow Lite models](https://www.tensorflow.org/lite/performance/measurement) — TensorFlow Authors (2024)
  用于使用TensorFlow Lite基准测试工具测量目标设备上模型性能的官方文档。
