---
course: "advanced-tensorflow"
chapter: "model-deployment-optimization"
lesson: "model-optimization-techniques"
sourceId: 2929
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-6-model-deployment-optimization/model-optimization-techniques"
title: "模型优化方法"
description: "了解量化和剪枝等方法，以减小模型大小并提升推理速度。"
order: 4
plots: ["plots/2929-0.json"]
sourceHash: "232c2ec648c542aaeb4a4c834edb378cd4e4277e5c09a548c2f13eba8dc9ad38"
sourceCorrections: []
---

模型训练完成后，为高效部署进行准备是很重要的一步。原始训练好的模型通常使用32位浮点数（`float32`），可能体积庞大且推理 (inference)时计算要求高。优化模型可以解决这些问题，使其更小、更快，并更适合各种部署目标，从强大的服务器到资源有限的边缘设备。TensorFlow生态系统中提供了多种常用方法来实现这些目的。

## 量化 (quantization)：降低数值精度

量化是一个减少表示模型权重 (weight)以及（可选地）激活所需的比特数的处理过程。量化通常将数字从`float32`转换为较低精度格式，例如16位浮点数（`float16`）或8位整数（`int8`）。

**量化的好处：**

- **模型大小减小：** 精度降低意味着每个参数 (parameter)的比特数减少，显著减小了模型的存储占用。一个`int8`模型大约是其`float32`模型的四分之一大小。
- **推理 (inference)更快：** 许多现代处理器（CPU、GPU、TPU、DSP）拥有针对低精度算术的专用硬件指令，从而带来显著的速度提升。随着较小数据类型的获取，内存带宽需求也随之降低。
- **功耗降低：** 更快的计算和更少的内存访问通常意味着更低的能耗，这对于移动和边缘设备尤为重要。

**量化的类型：**

TensorFlow提供了几种量化方法，主要在使用`tf.lite.TFLiteConverter`面向TensorFlow Lite时访问，但这些思想应用范围更广。

### 训练后量化 (quantization)（PTQ）

PTQ将量化应用于已经训练好的`float32`模型。它通常更容易实现，因为它不需要重新训练。

1. **动态范围量化：** 这是最简单的形式。权重 (weight)离线量化为`int8`或`float16`。激活在推理 (inference)过程中根据其观测到的范围动态量化为`int8`。这提供了不错的压缩（`int8`权重大约4倍）和中等程度的加速，尤其是在CPU上。由于其易用性，它是一个不错的起点。

# 示例：使用TFLiteConverter进行动态范围PTQ

````
import tensorflow as tf

# 假设'model'是您训练好的tf.keras.Model或SavedModel路径
converter = tf.lite.TFLiteConverter.from_keras_model(model) # 或 from_saved_model()
converter.optimizations = [tf.lite.Optimize.DEFAULT] # 启用默认优化，包括动态范围量化

tflite_quant_model = converter.convert()

# 保存量化模型
# with open('model_dynamic_quant.tflite', 'wb') as f:
#     f.write(tflite_quant_model)
```
````

2. **Float16量化：** 将所有`float32`权重和激活转换为`float16`。这使模型大小减半，并且可以在支持`float16`的硬件（如许多GPU和TPU）上显著加速推理。与`int8`量化相比，对准确性的影响通常很小。

````
```python
````

# 示例：使用TFLiteConverter进行Float16 PTQ

````
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16] # 指定float16目标

tflite_fp16_model = converter.convert()

# 保存量化模型
# with open('model_fp16_quant.tflite', 'wb') as f:
#     f.write(tflite_fp16_model)
```
````

3. **全整数量化（INT8）：** 这种方法将权重和激活都量化为`int8`。为了准确实现这一点，它需要使用一个代表性数据集进行“校准”步骤。在校准期间，转换器会在样本数据上运行推理，以确定网络中不同点的激活动态范围（最小值/最大值）。然后，这些范围用于将浮点值映射到`int8`整数。这种方法通常能带来最大的性能提升和大小减小，但与`float16`或动态范围量化相比，对准确性的影响可能略大。它通常需要TensorFlow Lite运行时或特定硬件（如Edge TPU）才能实现最佳执行。

````
```python
````

# 示例：使用TFLiteConverter进行全整型PTQ

````
import tensorflow as tf
import numpy as np

# 代表性数据集生成器（生成输入样本）
def representative_dataset_gen():
    # 提供少量（例如100-500）代表性样本
    for _ in range(100):
        # 替换为您实际的数据加载和预处理
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)] # 示例输入

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
# 确保只有整型操作
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# 可选：如果需要，指定输入/输出类型（通常自动推断）
# converter.inference_input_type = tf.int8 or tf.uint8
# converter.inference_output_type = tf.int8 or tf.uint8

tflite_int8_model = converter.convert()

# 保存量化模型
# with open('model_int8_quant.tflite', 'wb') as f:
#     f.write(tflite_int8_model)
```
````

### 量化 (quantization)感知训练（QAT）

QAT在训练或微调 (fine-tuning)过程中模拟`int8`量化的效果。它在模型图中插入“假量化”节点，这些节点在前向传播期间模拟实际量化的舍入和钳制行为，同时允许梯度在反向传播 (backpropagation)期间流动。

与PTQ相比，QAT通常能为最终的量化模型带来更高的准确性，特别是对于INT8，因为模型在训练期间会学习适应精度损失。然而，它需要修改模型定义和进行再训练/微调阶段。TensorFlow模型优化工具包（TF MOT）提供了QAT的API。

```python
# 示例：使用TF MOT应用QAT
import tensorflow_model_optimization as tfmot

# 假设'model'是您预训练的float32 tf.keras.Model
quantize_model = tfmot.quantization.keras.quantize_model

# 应用QAT封装器
qat_model = quantize_model(model)

# 用必要的callback编译模型
qat_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 使用QAT节点微调模型
# qat_model.fit(train_dataset, epochs=..., validation_data=...)

# 训练后，将QAT模型转换为完全整型的TFLite模型
# 转换器识别QAT结构
converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # QAT转换所需
tflite_qat_model = converter.convert()

# 保存量化模型
# with open('model_qat_int8.tflite', 'wb') as f:
#     f.write(tflite_qat_model)
```

**量化的权衡：**

优化程度（大小减小、加速）与潜在准确性下降之间通常存在权衡。对于INT8，QAT通常比PTQ能更好地保持准确性，但需要更多精力。



![量化权衡](plots/2929-0.json)



> 比较不同的量化技术。向左移动会增加实现复杂性，但可能会提高性能；较低的点表示优化程度较轻。

## 剪枝：降低模型复杂度

剪枝旨在通过移除（置为零）那些被认为不重要的权重 (weight)来减少模型中的参数 (parameter)数量，从而有效地创建稀疏模型。

**剪枝的好处：**

- **模型大小减小：** 置零的权重可以使用标准压缩技术（如gzip）有效压缩，从而减小存储的模型文件。
- **潜在加速：** 如果目标硬件和推理 (inference)引擎支持稀疏张量计算，剪枝后的模型可以执行得更快。然而，如果没有专门支持，加速可能会受到限制，因为底层计算通常仍涉及包含零的密集矩阵。
- **可以组合使用：** 剪枝通常可以与量化 (quantization)结合使用，以进行进一步的优化。

**剪枝的类型：**

TensorFlow模型优化工具包（TF MOT）提供了剪枝的工具。

1. **基于幅度的剪枝：** 这种常用方法移除绝对值（幅度）最低的权重。假设是，接近零的权重对模型的输出贡献较小。通常，剪枝在训练或微调 (fine-tuning)期间根据稀疏度计划逐步应用。

   - **非结构化剪枝：** 单个权重被置为零，导致稀疏权重矩阵中零可能散布其间。这实现了高压缩，但通常需要专门的库或硬件才能显著加速推理。
   - **结构化剪枝：** 移除整个权重块（例如，卷积层中的滤波器、通道，或全连接层中的行/列）。这会产生更小、更密集的矩阵，这些矩阵通常可以在标准硬件上更容易加速，尽管可实现的最大稀疏度可能低于非结构化剪枝。

**TF MOT中的实现：**

TF MOT主要侧重于基于幅度的剪枝，通常在训练或微调期间应用。

```python
# 示例：使用TF MOT在微调期间应用剪枝
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# 定义剪枝计划（例如，从第0步开始保持50%的恒定稀疏度）
pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity=0.5,
                                                               begin_step=0)
}

# 假设'model'是您预训练的float32 tf.keras.Model
model_for_pruning = prune_low_magnitude(model, **pruning_params)

# 用必要的StepPruningSchedule callback编译模型
model_for_pruning.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

# 在训练/微调期间需要包含更新callback
log_dir = '/tmp/pruning_logs' # 示例日志目录
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir) # TensorBoard可选
]

# 微调模型，使权重在剪枝后进行调整
# model_for_pruning.fit(train_dataset, epochs=..., validation_data=..., callbacks=callbacks)

# 为了获得可部署的模型，剥离剪枝封装器
model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

# 现在'model_for_export'可以保存或转换（例如，转换为TFLite）
# 注意：为了获得大小优势，导出的模型需要压缩（例如，zip/gzip）。
# 为了获得速度优势，目标推理引擎需要稀疏张量支持。
```

**剪枝的权衡：**

剪枝需要仔细微调以恢复权重移除后损失的准确性。实现显著加速通常在很大程度上取决于部署平台高效处理稀疏性的能力。结构化剪枝通常比非结构化剪枝对硬件更友好。

> 非结构化剪枝（单个权重置零）和结构化剪枝（整列置零）的区别。红色单元格表示被剪枝（置零）的权重。

## 权重 (weight)聚类

权重聚类，同样受TF MOT支持，将每层中的权重分组为少量簇（例如8、16或32个）。所有属于同一簇的权重共享一个质心值。这减少了需要存储的独特权重值的数量。

**好处：**

- **模型大小减小：** 通过仅存储簇质心值和指示每个原始权重属于哪个簇的索引映射，可以实现显著压缩（例如5倍-10倍），特别是与标准压缩技术结合时。
- **与量化 (quantization)互补：** 聚类可以在量化之前或之后应用，以获得潜在的叠加好处。

**实现：**

类似于剪枝和QAT，聚类通常通过TF MOT封装器应用，通常需要微调 (fine-tuning)。

```python
# 示例：使用TF MOT应用权重聚类
import tensorflow as tf
import tensorflow_model_optimization as tfmot

cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

# 定义聚类参数（例如，将全连接层中的权重聚类为16个簇）
clustering_params = {
  'number_of_clusters': 16,
  'cluster_centroids_init': CentroidInitialization.LINEAR # 或 KMEANS_PLUS_PLUS
}

# 将聚类封装器应用于模型或特定层
# 可以选择性地应用于全连接层、卷积层等
clustered_model = cluster_weights(model, **clustering_params)

# 编译模型（损失通常隐式包含聚类项）
clustered_model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

# 微调模型以调整质心和分配
# clustered_model.fit(train_dataset, epochs=..., validation_data=...)

# 剥离聚类封装器以进行部署
model_for_export = tfmot.clustering.keras.strip_clustering(clustered_model)

# 保存或转换聚类模型。大小优势需要压缩。
```

## 选择合适的优化方法

最佳优化策略取决于您的具体目标和限制：

- **易用性：** 训练后动态范围或float16量化 (quantization)是最简单的起点。
- **最大速度/大小减小：** 全整型（INT8）量化（特别是QAT）通常带来最佳效果，特别是在兼容硬件（移动CPU、Edge TPU）上。剪枝和聚类可以进一步减小大小，但速度优势取决于稀疏计算支持。
- **准确性保持：** Float16量化通常影响最小。QAT通常比PTQ在INT8量化方面能更好地保持准确性。剪枝和聚类需要仔细微调 (fine-tuning)以保持准确性。
- **目标平台：**
  - **服务器（GPU/TPU）：** Float16量化通常有效。INT8（PTQ或QAT）可以在较新的硬件上提供进一步好处。
  - **移动/边缘（CPU/DSP/NPU）：** INT8量化（PTQ或QAT）通常是最有效的方法，因为有专用硬件支持和内存限制。剪枝/聚类增加了减小大小的好处。
  - **微控制器：** 全整数量化是重要的。

通常，结合多种方法能带来最佳效果。例如，应用剪枝后进行量化感知训练，可以生成高度紧凑和高效的模型。在目标部署平台上进行实验和仔细评估是必要的，以找到性能、大小和准确性之间最适合您特定应用的平衡。

## 参考资料

- [TensorFlow Model Optimization Toolkit | Documentation](https://www.tensorflow.org/model_optimization) — Google (2024)
  Publisher: Google
  官方文档，提供在TensorFlow生态系统中实现量化、剪枝和聚类等模型优化技术的指南和API。
- [Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149) — Song Han, Huizi Mao, William J. Dally (2016)
  Journal: ICLR 2016; DOI: [10.48550/arXiv.1510.00149](https://doi.org/10.48550/arXiv.1510.00149)
  这篇开创性论文提出了一个通过剪枝、训练量化和霍夫曼编码实现神经网络压缩的三阶段框架，有效减小了模型大小。
- [Automatic Mixed Precision for Deep Learning](https://developer.nvidia.com/automatic-mixed-precision) — NVIDIA Developer (2024)
  该资源解释了在深度学习训练和推理中使用float16等低精度数据类型的好处和方法，以加速兼容硬件上的计算。
