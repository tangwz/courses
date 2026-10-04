---
course: "planning-optimizing-ai-infrastructure"
chapter: "strategies-for-performance-optimization"
lesson: "using-mixed-precision-training"
sourceId: 7025
sourceUrl: "https://apxml.com/zh/courses/planning-optimizing-ai-infrastructure/chapter-5-strategies-for-performance-optimization/using-mixed-precision-training"
title: "混合精度训练的应用"
description: "采用 FP16 等低精度格式实现混合精度训练，以加速现代 GPU 上的训练并减少内存使用。"
order: 3
plots: ["plots/7025-0.json"]
sourceHash: "f5cb09ac2ecdf8c1adcb196561fea579965d8e9f2c031579484446c2ea739309"
sourceCorrections: []
---

混合精度训练是一种旨在提升单张 GPU 计算效率的强大优化技术。大多数深度学习 (deep learning)模型都使用标准的32位浮点数 ($FP32$) 进行训练，这也被称为单精度。这种格式提供宽动态范围和高精度，使其成为一个安全的默认选择。然而，对于许多深度学习操作而言，这种精度水平并非完全必要。在这种情况下，混合精度训练成为一种有用的方法。

混合精度训练结合使用16位浮点数 ($FP16$)（即半精度）与传统的 $FP32$ 数值。主要原因是显著提升性能并减少内存占用。

- **更快的计算速度：** 现代 GPU，特别是 NVIDIA GPU 自 Volta 架构以来，都包含称为 Tensor 核心的专用硬件。这些核心设计用于以比标准 $FP32$ 计算更高的速率对 $FP16$ 输入执行矩阵运算。
- **内存占用减少：** 一个 $FP16$ 值占用的内存是 $FP32$ 值的一半。这立即将模型权重 (weight)、激活和梯度所需的内存减半。这让您能够训练更大的模型，使用更大的批量大小以提高 GPU 利用率，或降低现有模型的内存压力。



![NVIDIA A100 GPU上的理论吞吐量](plots/7025-0.json)



> NVIDIA A100 GPU 上标准 FP32 核心与用于 FP16 操作的专用 Tensor 核心之间的性能对比。性能提升显著。

### 数值稳定性的挑战

简单地将整个模型切换到 $FP16$ 可能会导致问题。与 $FP32$ 相比，$FP16$ 格式的可表示范围要小得多。在反向传播 (backpropagation)过程中，梯度值会变得非常小。在 $FP16$ 中，这些小值可能会四舍五入为零，这种现象称为**下溢**。当梯度变为零时，模型这些部分的权重 (weight)更新就会停止，网络会停止有效学习。

解决方案并非放弃 $FP16$，而是选择性地使用它，结合一种称为**动态损失缩放**的技术。这种方法构成了现代混合精度训练的核心。

### 混合精度训练的工作原理

现代框架无需手动决定哪些操作以何种精度运行，而是自动化一个三步流程：

1. **在 FP32 中维护主权重 (weight)：** 模型权重的主副本始终以 $FP32$ 存储。这确保了微小的梯度更新可以在数千次训练步骤中精确累积，而不会损失精度。
2. **以 FP16 执行计算：** 对于每个训练步骤，$FP32$ 主权重会转换为 $FP16$。然后使用这些 $FP16$ 权重和激活执行前向和反向传播 (backpropagation)。Tensor 核心带来的加速就是在这里实现的。
3. **使用动态损失缩放：** 为防止梯度下溢，在反向传播开始*之前*，将计算出的损失乘以一个大的缩放因子。这会放大所有产生的梯度，将其值移至 $FP16$ 可以安全表示的范围。梯度计算完成后，在更新 $FP32$ 主权重*之前*，将其按比例缩小回原始范围。缩放因子在训练过程中动态调整；如果梯度溢出（变为 `inf` 或 `NaN`），则因子会减小。如果训练稳定一段时间，则因子会增加，以充分使用 $FP16$ 的范围。

> 带有动态损失缩放的混合精度训练单个步骤中的数据流。

### 实际操作中的实现

幸运的是，您很少需要从头实现此逻辑。主要深度学习 (deep learning)框架提供高层 API，可以为您处理所有细节。

#### 在 PyTorch 中

PyTorch 通过其 `torch.cuda.amp` 模块提供自动混合精度功能。实现只需要向标准训练循环添加几行代码。

您需要两个组件：`autocast` 和 `GradScaler`。

- `torch.cuda.amp.autocast`：一个上下文 (context)管理器，它自动选择在其作用域内的每个操作的适当精度（$FP16$ 或 $FP32$）。
- `torch.cuda.amp.GradScaler`：管理动态损失缩放过程，以防止梯度下溢。

以下是标准 PyTorch 训练循环与修改为自动混合精度 (AMP) 的训练循环的对比。

**标准 PyTorch 训练循环：**

```python
# 标准训练循环
optimizer.zero_grad()
outputs = model(inputs)
loss = loss_fn(outputs, targets)
loss.backward()
optimizer.step()
```

**使用 PyTorch AMP 的训练循环：**

```python
# 建议：在训练循环外部创建 scaler 一次
scaler = torch.cuda.amp.GradScaler()

# --- 在训练循环内部 ---
optimizer.zero_grad()

# 在适当情况下将操作转换为 FP16
with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)

# 放大损失。在放大的损失上调用 backward() 以创建放大的梯度。
scaler.scale(loss).backward()

# scaler.step() 首先还原优化器参数的梯度。
# 如果没有发现 infs/NaNs，则调用 optimizer.step()。
# 否则，optimizer.step() 将被跳过。
scaler.step(optimizer)

# 更新下次迭代的缩放因子。
scaler.update()
```

#### 在 TensorFlow 中

TensorFlow 的 Keras API 使启用混合精度更加简单。您设置一个全局策略，Keras 会自动处理其余部分，包括 `model.fit()` 方法中的损失缩放。

要启用混合精度，您需要在脚本开头添加一行代码。

```python
import tensorflow as tf

# 设置全局策略以使用混合精度
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ... 像往常一样定义模型
inputs = tf.keras.Input(shape=(...))
# 层会自动使用混合精度
outputs = tf.keras.layers.Dense(10, activation='softmax')(inputs) 
model = tf.keras.Model(inputs=inputs, outputs=outputs)

optimizer = tf.keras.optimizers.Adam()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

# model.fit 会自动处理损失缩放
# model.fit(x_train, y_train, ...)
```

当您设置此策略时，Keras 会自动封装您的优化器以处理损失缩放，并确保像 `Dense` 和 `Conv2D` 这样的层使用 $FP16$ 计算，同时保持 $FP32$ 权重 (weight)。此更改是透明的，只需要最少的代码修改。

通过采用混合精度训练，您通常可以在支持的硬件上通过几行代码实现 1.5 倍到 3 倍的加速，使其成为对于 GPU 瓶颈的工作负载最有效的优化方法之一。

## 参考资料

- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory F. Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍混合精度训练和动态损失缩放原理及优势的原始研究论文。
- [Automatic Mixed Precision (AMP) for Deep Learning Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) — NVIDIA Developer Documentation (2023)
  Publisher: NVIDIA
  NVIDIA 提供的综合指南，解释混合精度训练的优点、实现细节和最佳实践。
- [Automatic Mixed Precision (AMP)](https://pytorch.org/docs/stable/amp.html) — PyTorch Authors (2025)
  Publisher: PyTorch Documentation
  PyTorch 官方文档，提供 torch.cuda.amp、autocast 和 GradScaler 的详细使用说明和示例。
- [Mixed precision](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Authors (2023)
  Publisher: TensorFlow Documentation
  TensorFlow 官方指南，说明如何使用 Keras API 中的 tf.keras.mixed_precision.set_global_policy 启用和配置混合精度训练。
