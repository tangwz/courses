---
course: "advanced-tensorflow"
chapter: "high-performance-tensorflow"
lesson: "mixed-precision-training"
sourceId: 2825
sourceUrl: "https://apxml.com/zh/courses/advanced-tensorflow/chapter-2-high-performance-tensorflow/mixed-precision-training"
title: "混合精度训练技术"
description: "实现混合精度训练以加速模型训练并减少现代GPU上的内存使用。"
order: 3
plots: ["plots/2825-0.json"]
sourceHash: "6a06ff4a3fd31f849f98d475085ef001f92ec765bb376d321aee0b281c740e68"
sourceCorrections: []
---

提升模型复杂性通常会遇到计算瓶颈。训练大型模型需要大量时间与内存资源。缓解这些限制的一种有效方法，特别是在NVIDIA GPU（Volta架构及更高版本）和Google TPU等现代硬件加速器上，就是**混合精度训练**。

其主要思想是：对计算的某些部分策略性地使用较低精度的浮点数，具体来说是16位浮点数（$float16$ 或半精度），同时将重要部分保持在标准的32位单精度（$float32$）。这种精度的“混合”旨在实现平衡：获得$float16$的速度和内存优势，同时保持通常与$float32$相关的数值稳定性与精度。

### 理解精度取舍

标准的深度学习 (deep learning)模型主要使用$float32$来存储权重 (weight)、激活值和计算梯度。每个$float32$数字占用32位内存。相比之下，$float16$仅使用16位。

- **内存节省：** 从$float32$切换到$float16$大约将模型权重和激活值所需的内存减半。这种减少可以很显著，使您能够在相同的硬件内存限制内训练更大的模型或使用更大的批次大小。
- **计算速度：** 现代GPU（NVIDIA Volta、Turing、Ampere和更新的架构）配备了称为Tensor Core的专用硬件单元，设计用于在$float16$数据上进行操作时显著加速矩阵乘法和卷积。类似地，TPU针对低精度计算进行了优化。使用$float16$允许TensorFlow运用这些硬件加速，从而大幅提升训练速度，在兼容的GPU上通常是2倍或更多。



![每个参数的近似内存使用量](plots/2825-0.json)



> 单精度（$float32$）与半精度（$float16$）浮点数的内存占用对比。

然而，这种效率是以牺牲相较于$float32$更小的数值范围和精度为代价的。$float16$的较小范围使其在训练期间更容易出现两个主要的数值问题：

1. **下溢：** 小的梯度值，尤其是在深度网络中或梯度裁剪后常见的梯度值，当用$float16$表示时可能会变为零。这实际上停止了这些参数 (parameter)的学习。
2. **上溢：** 大的梯度值可能超过$float16$所能表示的最大值，导致无穷大（`Inf`）或非数字（`NaN`）值，破坏训练过程的稳定性。

### 稳定的混合精度训练方法

为应对这些数值挑战并使训练成为可能，混合精度通常采用两种主要方法：

1. **维护$float32$主权重 (weight)：** 虽然层内计算（如矩阵乘法）通常使用$float16$输入和输出来提高速度，但模型的权重主副本仍保持在$float32$中。梯度更新，尽管可能使用$float16$激活值和梯度计算，但会累积到这些$float32$主权重中。这可以防止由于小的梯度更新被反复直接应用于$float16$权重而可能发生的精度损失。用于计算的$float16$权重是在前向传播之前通过对$float32$主权重进行类型转换生成的。
2. **损失缩放：** 为防止梯度在$float16$范围内下溢（变为零），在反向传播 (backpropagation)开始*之前*，损失值会乘以一个大的缩放因子。这会按比例放大所有中间梯度。在优化器将这些梯度应用于$float32$主权重之前，它们会被取消缩放（除以相同的缩放因子），恢复到原始大小。

   - **静态损失缩放：** 使用固定的、手动选择的缩放因子。这需要实验以找到一个足够大的因子来防止下溢，但又足够小以避免模型中典型梯度的上溢。
   - **动态损失缩放：** 在训练期间自适应地调整缩放因子。它从一个大因子开始，如果检测到上溢（`Inf`或`NaN`梯度），则减小该因子。这通常更受欢迎，因为它会自动找到接近最优的缩放比例，无需手动调整。

> 混合精度训练的流程，显示了类型转换、计算、损失缩放和权重更新。

### 在TensorFlow中启用混合精度

TensorFlow提供了一个简单直接的API，通过`tf.keras.mixed_precision`来启用混合精度训练。最简单的方式是设置全局策略。

```python
import tensorflow as tf

# 检查GPU是否支持Tensor Core
# (NVIDIA GPU计算能力7.0或更高版本)
# TPU本身也支持混合精度。

# 设置全局策略为 'mixed_float16'
# 这会自动为兼容的Keras层启用混合精度
tf.keras.mixed_precision.set_global_policy('mixed_float16')

print(f"Compute dtype: {tf.keras.mixed_precision.global_policy().compute_dtype}")
print(f"Variable dtype: {tf.keras.mixed_precision.global_policy().variable_dtype}")

# 像往常一样构建您的Keras模型
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28), name='input'),
    # Flatten层没有计算密集型操作，数据类型策略对其影响不大
    tf.keras.layers.Flatten(),
    # 全连接层计算将使用float16，权重保留在float32中
    tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
    # 输出层可能保持float32以保证数值稳定性，具体取决于设置
    # Keras策略会自动处理标准层（如带有softmax的全连接层）的此问题
    tf.keras.layers.Dense(10, activation='softmax', name='output')
])

# 检查层的Dtype
dense_layer = model.get_layer('dense_1')
print(f"Dense layer compute dtype: {dense_layer.compute_dtype}")
print(f"Dense layer variable dtype: {dense_layer.variable_dtype}")
# 输出层通常默认为float32计算以保证稳定性，特别是softmax
output_layer = model.get_layer('output')
print(f"Output layer compute dtype: {output_layer.compute_dtype}")

# 编译模型 - 默认优化器会自动处理损失缩放
# 当使用model.fit()与混合精度策略时
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 现在，当您调用model.fit()时，Keras将自动：
# 1. 将输入转换为float16，用于兼容的层。
# 2. 以float16执行计算（例如，矩阵乘法）。
# 3. 将主权重保留在float32中。
# 4. 在梯度计算期间应用动态损失缩放。
# 5. 在将梯度应用于float32权重之前取消其缩放。
```

当您将全局策略设置为`'mixed_float16'`时，Keras层会自动调整：

- 大多数计算密集型层（如`Dense`、`Conv2D`、循环层）将使用$float16$执行其计算，并期望$float16$输入。它们的内部变量数据类型（用于权重 (weight)）仍保持为$float32$。
- 某些层，通常是那些涉及大量约简或`BatchNormalization`或最终`Softmax`激活等敏感操作的层，即使在混合精度策略下，也可能默认以$float32$进行计算以确保数值稳定性。这种行为通常是自动且有益的。
- 标准的`model.fit()`训练循环会自动将优化器封装在`tf.keras.mixed_precision.LossScaleOptimizer`中，该优化器处理动态损失缩放。

如果您正在编写自定义训练循环，则需要手动管理损失缩放。这需要使用`tf.keras.mixed_precision.LossScaleOptimizer`，它会封装您的常规优化器。您可以使用它在计算梯度之前缩放损失，并在应用梯度之前取消其缩放。

```python
# 自定义混合精度训练循环示例代码

# 假设 'optimizer' 是您的基本优化器（例如 tf.keras.optimizers.Adam）
# 假设 'model' 和 'loss_fn' 已定义
# 策略 'mixed_float16' 必须全局设置

# 封装优化器以进行损失缩放
scaled_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

@tf.function
def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True) # 前向传播使用混合精度
        # 如有需要，确保损失计算在float32中完成
        loss = loss_fn(targets, predictions)
        # 缩放损失
        scaled_loss = scaled_optimizer.get_scaled_loss(loss)

    # 使用缩放后的损失计算梯度
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    # 在应用前取消梯度缩放
    gradients = scaled_optimizer.get_unscaled_gradients(scaled_gradients)
    # 使用LossScaleOptimizer应用梯度（更新float32权重）
    scaled_optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 在您的训练循环中：
# for batch_data in dataset:
#   inputs, targets = batch_data
#   loss_value = train_step(inputs, targets)
#   print(f"Step loss: {loss_value.numpy()}")
```

### 何时使用混合精度

混合精度训练在以下情况下最有益：

- 您正在使用支持Tensor Core的NVIDIA GPU（Volta、Turing、Ampere、Hopper或更新版本）或Google TPU进行训练。在较旧的GPU或CPU上，性能提升通常微乎其微或不存在。
- 您的模型足够大，以至于内存容量成为限制因素（阻碍使用更大的批次大小或更深的网络结构）。
- 训练时间是您开发周期中的一个显著瓶颈。

务必验证启用混合精度不会对您特定任务的模型最终精度产生负面影响，尽管在大多数情况下，由于正则化 (regularization)效应，影响微乎其微甚至略有积极作用。请监测训练中损失的`NaN`值，这可能表示损失缩放或特定操作中存在数值稳定性问题。

总之，混合精度是TensorFlow中一种强大优化手段，它通过运用专用硬件能力，可以大幅减少训练时间与内存使用。它与Keras API的集成使其在许多标准模型架构中都相对容易实现。

## 参考资料

- [Mixed-Precision Training of Deep Neural Networks](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了混合精度训练的基础技术，包括损失缩放和使用float32主权重。
- [Mixed precision](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Developers (2024)
  TensorFlow和Keras中实现混合精度的官方指南，涵盖了API使用和最佳实践。
- [Accelerating AI Training with NVIDIA Tensor Cores and Automatic Mixed Precision](https://developer.nvidia.com/blog/accelerating-ai-training-with-nvidia-tensor-cores-and-automatic-mixed-precision/) — Mark Harris (2020)
  Journal: NVIDIA Developer Blog; Publisher: NVIDIA
  提供了关于NVIDIA Tensor Core如何加速混合精度训练的见解，并讨论了NVIDIA硬件上的优势。
