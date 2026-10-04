---
course: "fine-tuning-adapting-large-language-models"
chapter: "optimization-deployment-considerations"
lesson: "memory-optimization-training"
sourceId: 3083
sourceUrl: "https://apxml.com/zh/courses/fine-tuning-adapting-large-language-models/chapter-7-optimization-deployment-considerations/memory-optimization-training"
title: "训练时的内存优化"
description: "梯度累积、激活检查点和混合精度训练（AMP）等方法。"
order: 1
plots: ["plots/3083-0.json"]
sourceHash: "06b72454bb3efc7cc7ca881e1f0e435f0163184c03f00ccc06cc4acc0d07e633"
sourceCorrections: []
---

大语言模型 (LLM)微调 (fine-tuning)，尤其是在使用全参数 (parameter)更新或甚至更巧妙的参数高效微调（PEFT）方法处理超大模型时，常常让现有硬件资源的限制显现。尽管优化已*训练好*的模型以进行推理 (inference)会在其他地方介绍，但在微调过程中管理内存占用的方法是主要的考虑。GPU内存不足（通常表现为CUDA内存溢出错误）是一个常见难题，会导致训练中断并需要调整。幸运的是，有多种策略可以缓解这种压力，通常是以计算时间略微增加为代价来显著节省内存。

### 梯度累积

导致内存达到限制的直接方式之一是尝试使用大批量大小。尽管较大的批量有时可以带来更稳定的梯度和按周期计算的更快收敛，但批量中的每个样本在正向传播时消耗激活内存，在反向传播 (backpropagation)时消耗梯度内存。

梯度累积提供了一种巧妙的应对方法。它模拟了更大的有效批量大小，而无需同时将整个大批量载入内存。其主要思想是按顺序处理几个较小的“微批量”，计算每个微批量的梯度，并在执行单个优化器步骤和更新模型权重 (weight)*之前*累积这些梯度。

以下是典型的处理流程：

1. **初始化梯度：** 在开始新的有效批量之前，像往常一样将模型的梯度清零。
2. **循环处理微批量：** 对于指定数量的累积步长（`accumulation_steps`）：
   - 载入一个微批量的数据。
   - 执行正向传播：计算模型的预测值和损失。
   - 缩放损失（可选但常见）：将损失除以`accumulation_steps`。这可以防止累积梯度幅度相对于单个大批量梯度变得过大。
   - 执行反向传播：计算当前微批量的梯度。这些梯度会*添加*到累积周期内先前步骤中计算的梯度（大多数框架会自动处理）。
3. **更新权重：** 在遍历所有`accumulation_steps`后，执行优化器步骤（`optimizer.step()`）。这会使用来自所有微批量的聚合梯度更新模型权重。
4. **梯度清零：** 重置梯度（`optimizer.zero_grad()`）以准备下一个累积周期。

实际操作中，如果你的目标批量大小是64，但GPU只能处理批量大小为8的数据，你可以设置`accumulation_steps = 8`（因为 $64 / 8 = 8$）。模型将执行8次正向和反向传播，累积梯度，然后只更新一次权重，从而达到与直接使用批量大小为64相同权重的更新效果。

**考量：**

- **训练时间：** 梯度累积会增加每个训练周期的实际耗时，因为为了相同数量的权重更新，需要进行更多的正向/反向传播。
- **批量归一化 (normalization)：** 如果你的模型使用批量归一化层，请注意这些层是根据*微批量*大小而非有效批量大小计算统计数据的。这种差异有时会影响模型性能，与真正的全批量训练相比。对于主要使用层归一化的Transformer模型，这通常不是一个大问题。

### 激活检查点（梯度检查点）

在深度神经网络 (neural network)（如Transformer）的正向传播过程中，各层中间输出（激活值）通常存储在内存中。这些激活值在反向传播 (backpropagation)时需要用来计算梯度。对于层数多且隐藏维度大的模型，这些存储的激活值所占用的内存会变得相当可观。

激活检查点，也称为梯度检查点，提供了一种权衡：它通过*不*存储所有中间激活值来减少内存使用。相反，它在正向传播期间有策略地只保存一部分激活值。在反向传播时，当需要一个未保存的激活值时，该技术会从最近的已保存激活值开始执行部分正向传播，即时重新计算它。

**权衡：**

- **内存节省：** 可以显著减少内存消耗，特别是对于深度模型。具体节省量取决于模型架构和检查点应用方式。
- **计算成本：** 增加训练时间，因为正向传播的部分内容需要在反向传播期间重新执行。

**实现：**

现代深度学习 (deep learning)框架通常提供工具，可以相对轻松地启用激活检查点。例如，在PyTorch中，你可以使用`torch.utils.checkpoint.checkpoint`。Hugging Face `transformers`库通常允许通过模型配置中的`gradient_checkpointing=True`标志或在`Trainer`设置期间启用它。这抽象化了决定保存哪些激活值和管理重新计算的复杂性。

当面临内存限制时，激活检查点是一种有价值的技术，特别是如果仅靠梯度累积不足，或者你需要为其他目的（例如使用更复杂的优化器）释放内存时。

### 混合精度训练

默认情况下，大多数深度学习 (deep learning)模型使用32位浮点数（$fp32$或单精度）进行训练。混合精度训练指的是在训练过程中结合使用$fp32$和低精度格式，主要是16位浮点数（$fp16$或半精度）。

**优势：**

- **内存占用减少：** 与$fp32$相比，使用$fp16$大约将存储权重 (weight)、激活值和梯度所需的内存减半。这使得可以使用更大的批量大小，或将更大的模型适应到相同的GPU内存中。
- **更快的计算：** 现代GPU（如NVIDIA Tensor Cores）拥有针对$fp16$计算优化的专用硬件，可显著加速矩阵乘法和卷积。



![每个参数的内存使用情况（训练）](plots/3083-0.json)



> 训练期间不同组件每个模型参数 (parameter)的内存使用图示。请注意，权重和梯度直接受益于低精度，而优化器状态（如Adam的动量和方差）通常为了稳定性而保持在$fp32$。

**挑战与解决方案：**

$fp16$的主要挑战是与$fp32$相比其数值范围有限。小的梯度值可能变为零（“下溢”），而大的值可能超出可表示范围（“上溢”），从而导致数值不稳定和收敛不良。

自动混合精度（AMP）框架通过**损失缩放**来解决这个问题：

1. **正向传播：** 某些操作以$fp16$执行（安全且有益时），而其他操作（如归约）可能保留在$fp32$中。
2. **损失缩放：** 在反向传播 (backpropagation)之前，计算出的损失值乘以一个缩放因子（$S$）。
3. **反向传播：** 梯度根据缩放后的损失计算。由于损失被放大，因此得到的梯度也会被$S$放大，有助于防止下溢。
4. **梯度反缩放：** 在优化器更新权重之前，梯度通过除以$S$被反向缩放。
5. **权重更新：** 优化器更新权重（通常使用$fp32$副本的权重以保证稳定性）。
6. **动态缩放因子：** 缩放因子$S$通常在训练期间动态调整。如果发生上溢（梯度变为无穷大或NaN），$S$会减小。如果梯度在一段时间内保持稳定，$S$可能会增加，以充分利用$fp16$的范围。

**实现：**

PyTorch（`torch.cuda.amp`）和TensorFlow（`tf.keras.mixed_precision`）等库提供了AMP实现，通常只需几行代码即可启用（使用自动类型转换上下文 (context)和梯度缩放器）。

**替代方案：$bfloat16$**

一种较新的16位格式，$bfloat16$（$bf16$），正在获得关注。它使用与$fp16$相同的位数，但分配方式不同：更少的位用于精度（尾数），更多的位用于指数。这使得$bf16$拥有与$fp32$相同的动态范围，但精度低于$fp16$。

- **优势：** 与$fp16$相比，更不容易出现上溢/下溢问题，通常无需进行损失缩放。
- **劣势：** 精度降低可能会影响某些敏感模型的收敛。需要硬件支持（例如NVIDIA Ampere/Hopper GPU，Google TPU）。

在可用时，$bf16$与$fp16$相比，可以提供更简单的混合精度优势获取途径，有可能在速度、内存节省和稳定性之间取得良好平衡。

### 结合多种方法

这些内存优化方法并非互斥。通常的做法是结合使用它们以达到最佳效果。例如，你可以使用：

- **梯度累积 + AMP：** 在处理大有效批量的同时，受益于混合精度的内存节省和加速。
- **激活检查点 + AMP：** 显著减少激活内存，同时利用低精度格式。

选择合适的组合取决于具体的模型、硬件限制以及训练期间观察到的实验结果。每种方法都会引入一个权衡，主要是在内存使用和计算时间之间。通过了解和应用梯度累积、激活检查点和混合精度训练，即使面对硬件限制，你也能显著提升微调 (fine-tuning)大语言模型 (LLM)的能力。

## 参考资料

- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) — Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin (2016)
  Journal: arXiv preprint; Publisher: arXiv; Volume: 48; Pages: 686-694; DOI: [10.48550/arXiv.1604.06174](https://doi.org/10.48550/arXiv.1604.06174)
  介绍了梯度检查点技术，通过在反向传播过程中重新计算激活来降低内存消耗。
- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: ICLR; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  提出了使用FP16和损失缩放的自动混合精度训练的基础技术和优势。
- [Automatic Mixed Precision package - torch.cuda.amp](https://pytorch.org/docs/stable/amp.html) — PyTorch Development Team (2025)
  Publisher: PyTorch Foundation
  PyTorch自动混合精度（AMP）工具的官方文档，包含对FP16和BF16的支持。
- [Optimize Your Training For Memory Efficiency](https://huggingface.co/docs/accelerate/usage_guides/memory) — Hugging Face (2024)
  Hugging Face Accelerate 关于训练大型模型内存优化策略的指南，涵盖梯度累积、梯度检查点和混合精度。
