---
course: "foundations-transformers-architecture"
chapter: "implementation-details-optimization"
lesson: "mixed-precision-training"
sourceId: 2067
sourceUrl: "https://apxml.com/zh/courses/foundations-transformers-architecture/chapter-7-implementation-details-optimization/mixed-precision-training"
title: "混合精度训练"
description: "使用较低精度格式（如FP16）以加快训练并减少内存占用。"
order: 7
plots: ["plots/2067-0.json"]
sourceHash: "4894d052f7576c06ae6036b08c75b7914c4d3bba045e214a5364e1c8954765ba"
sourceCorrections: []
---

使用标准32位浮点精度（FP32）训练大型Transformer模型会消耗大量计算资源并占用大量内存。混合精度训练提供了一个有效的办法，它在较低精度格式（如16位浮点数FP16或BF16）下执行某些操作，同时将主权重 (weight)等主要部分保留在FP32中。这种方式能显著加快计算速度并减少内存占用，通常对最终模型的精度影响很小或没有影响。

### 动机：速度与内存

现代硬件加速器，特别是配备NVIDIA Tensor Cores等专用单元的GPU，在较低精度（FP16或BF16）下执行矩阵乘法操作时，相比FP32能提供很大的性能提升。以16位精度执行前向和后向传播的部分步骤，直接意味着更快的训练迭代。

此外，与FP32相比，使用16位格式可将存储激活值、梯度和可能的模型权重 (weight)所需的内存减半。这种内存节省使得以下成为可能：

1. 训练使用FP32时无法放入内存的更大模型。
2. 使用更大的批处理大小，这可以提高梯度准确性并可能加快收敛速度。
3. 减少分布式训练设置中的通信开销。

### 混合精度训练的工作原理

主要思想是发挥较低精度在速度和内存上的优势，用于大部分计算，同时通过策略性地使用FP32来保持数值稳定性。虽然实现方式略有不同，并且通常由深度学习 (deep learning)框架自动处理，但典型过程包含多个组成部分：

1. **FP32主权重 (weight)：** 模型权重的主要副本通常保存在FP32中。这确保了权重更新（涉及随时间累积的小梯度值）不会受到16位格式有限精度的影响。
2. **FP16/BF16计算：** 在前向和后向传播过程中，权重被转换为FP16或BF16，用于自注意力 (self-attention)层和前馈层中的矩阵乘法等计算密集型操作。在这些传播过程中生成的激活值和梯度也以较低精度格式存储。
3. **损失缩放：** FP16的动态范围比FP32小很多。在FP16中计算的梯度，特别是对于深层网络或小损失值，可能变为零（下溢）。为避免此情况，计算出的损失值在反向传播 (backpropagation)开始*之前*会乘以一个缩放因子。这会放大梯度，将它们推入FP16的可表示范围。
4. **权重更新：** 在更新FP32主权重之前，计算出的梯度（FP16/BF16格式且已放大）会除以相同的缩放因子，以使它们恢复到正确的数量级。这些未缩放的梯度随后被转换为FP32，并使用选定的优化器（例如AdamW）来更新FP32主权重。

现代框架通常使用*动态损失缩放*，其中缩放因子在训练期间自动调整。如果检测到溢出（梯度变为`Inf`或`NaN`），缩放因子会减小。如果梯度在一定步数内保持稳定，缩放因子可能会增加，以更好地使用FP16的动态范围。

### FP16与BF16的选择

使用两种常见的16位格式：

- **FP16 (IEEE 半精度)：** 使用1位符号位、5位指数位和10位尾数位。它比BF16提供更高的精度，但与FP32相比动态范围小得多。这个有限的范围使其更容易受到下溢和溢出的影响，因此需要仔细的损失缩放。大多数现代GPU都对FP16有出色的硬件支持。
- **BF16 (BFloat16)：** 使用1位符号位、8位指数位（与FP32相同）和7位尾数位。它的主要优点是动态范围与FP32相似，因此远不那么容易出现下溢/溢出问题。损失缩放通常不需要或只需要更简单的静态缩放。然而，它比FP16提供较低的精度（尾数位更少）。在TPU和新一代GPU（例如NVIDIA Ampere及更高版本）中很常见对它的硬件支持。

选择通常取决于硬件是否可用。如果两者都支持，BF16可能会提供稍微简单的训练设置，因为它在数值范围上具有鲁棒性，而FP16在其更高精度有利的场景下可能会略微更好，前提是使用了有效的损失缩放。

### 实际实施

深度学习 (deep learning)框架提供方便的API，通过最少的代码改动来实现混合精度训练。

- **PyTorch：** 使用`torch.cuda.amp`（自动混合精度）模块。它提供上下文 (context)管理器（`autocast`）和梯度缩放工具（`GradScaler`）。

  ```python
  # 示例草图 (PyTorch)
  import torch
  from torch.cuda.amp import GradScaler, autocast

  scaler = GradScaler()
  model = YourTransformerModel().cuda()
  optimizer = torch.optim.AdamW(model.parameters(), lr=...)

  for inputs, targets in dataloader:
      inputs, targets = inputs.cuda(), targets.cuda()

      optimizer.zero_grad()

      # 将上下文管理器中的操作转换为FP16/BF16
      with autocast(dtype=torch.float16): # 或者如果支持/需要，使用torch.bfloat16
          outputs = model(inputs)
          loss = compute_loss(outputs, targets)

      # 缩放损失。在缩放后的损失上调用backward()以创建缩放后的梯度。
      scaler.scale(loss).backward()

      # scaler.step()首先取消优化器分配参数的梯度缩放。
      # 如果梯度不是inf/NaN，则调用optimizer.step()。
      scaler.step(optimizer)

      # 更新下一个迭代的缩放因子。
      scaler.update()
  ```
- **TensorFlow：** 使用`tf.keras.mixed_precision` API。您可以设置全局策略或按层应用它。当使用`model.fit`时，TensorFlow会自动处理损失缩放。

  ```python
  # 示例草图 (TensorFlow)
  import tensorflow as tf

  # 设置全局策略（例如，'mixed_float16'）
  policy = tf.keras.mixed_precision.Policy('mixed_float16')
  tf.keras.mixed_precision.set_global_policy(policy)

  # 像往常一样构建模型
  inputs = tf.keras.Input(...)
  # ... 定义Transformer层 ...
  outputs = tf.keras.layers.Dense(vocab_size, activation='softmax', dtype='float32')(x) # 输出层通常保留在FP32中
  model = tf.keras.Model(inputs=inputs, outputs=outputs)

  optimizer = tf.keras.optimizers.AdamW(...)
  # 当使用混合策略时，model.fit会自动处理损失缩放
  model.compile(optimizer=optimizer, loss='...', metrics=[...])
  model.fit(dataset, epochs=...)
  ```

虽然混合精度训练非常有效，但建议监控训练稳定性，并偶尔将最终模型性能与基线FP32运行进行比较，尤其是在首次将其应用于新架构或任务时。某些数值操作，例如大范围约简或需要高精度的计算，有时可能会因框架启发式算法而从自动转换中排除，或可能需要手动配置以保留在FP32中。



![混合精度训练的优点（示意）](plots/2067-0.json)



> 示意性比较，显示混合精度训练可能带来的速度提升（例如，快1.8倍）和内存节省（例如，减少45%）。实际收益取决于模型、硬件和具体的实现方式。

混合精度训练已成为深度学习从业者工具箱中的一项标准技术，特别是对于Transformer等资源密集型模型。通过智能地结合较低精度计算和保持数值稳定性的机制，它使得训练迭代更快，并且在现有硬件限制下使用更大、能力更强的模型成为可能。

## 参考资料

- [Mixed-Precision Training of Deep Neural Networks](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了混合精度训练的基础技术，包括FP32主权重和损失缩放，这些技术构成了现代实现的基础。
- [Automatic Mixed Precision (AMP) Examples](https://pytorch.org/docs/stable/notes/amp_examples.html) — PyTorch Contributors (2024)
  官方指南，提供了使用PyTorch的`torch.cuda.amp`模块实现混合精度训练的实际示例和详细信息。
- [Mixed precision](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Team (2023)
  TensorFlow Keras中启用混合精度训练的官方指南，说明了策略配置和自动损失缩放。
