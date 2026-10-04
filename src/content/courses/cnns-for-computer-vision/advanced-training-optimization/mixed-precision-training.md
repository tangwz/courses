---
course: "cnns-for-computer-vision"
chapter: "advanced-training-optimization"
lesson: "mixed-precision-training"
sourceId: 2546
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-2-advanced-training-optimization/mixed-precision-training"
title: "混合精度训练的基本原理"
description: "了解使用混合精度（FP16/FP32）训练的原理和优势，以实现更快的计算和更低的内存消耗。"
order: 7
plots: ["plots/2546-0.json"]
sourceHash: "389fe6020e5e322a802066b4f88e1a4172e78ee9307facfb3648f34bfd753bfb"
sourceCorrections: []
---

训练大型、深度卷积神经网络 (neural network) (CNN)常常对计算资源造成极大压力。随着模型深度和宽度增加，其内存占用显著增加，并且每次训练迭代所需的时间也随之延长。解决这些限制的一种有效方法是混合精度训练。这种方法巧妙地运用低精度浮点数（特别是16位浮点数，即FP16），与标准的32位浮点数（FP32）格式一同使用，以加快训练并降低内存消耗。

现代GPU，特别是那些拥有NVIDIA Tensor Cores等专用硬件的GPU，在使用FP16进行数学运算时，相比FP32能快很多。此外，以FP16格式存储激活值、梯度甚至参数 (parameter)，所需的内存带宽和容量比FP32减半。

## 混合精度如何运作

“混合”精度的说法是因为并非所有训练过程的部分都适合低精度。虽然大部分计算密集型操作，例如前向和反向传播 (backpropagation)中的矩阵乘法和卷积，通常可以在FP16中安全高效地执行，但某些操作需要FP32的更高精度来保持数值稳定性和准确性。

典型的混合精度训练设置包含以下组成部分：

1. **FP32 权重 (weight)主副本：** 模型权重的主副本以FP32格式保存。这确保了小梯度更新能在许多训练步骤中准确累积，防止因FP16精度有限而导致信息丢失。
2. **FP16 计算：** 在每次训练迭代中，FP32主权重被转换为FP16。然后，前向和反向传播使用这些FP16权重和激活值进行计算。这发挥了FP16硬件的速度优势。
3. **FP32 权重更新：** 在反向传播过程中计算出的梯度（最初为FP16）通常在用于更新FP32主权重之前转换回FP32。这保持了权重更新步骤的准确性。

## 解决数值稳定性问题：损失缩放

FP16的一个重要挑战是其动态范围相比FP32有限。FP16中可表示的最小正数远大于FP32。在反向传播 (backpropagation)过程中，梯度值，特别是对于深度网络或靠近输出的层，会变得非常小。如果这些值低于FP16可表示的最小值，它们就会变为零（下溢），实际上停止了网络这些部分的学习。

为应对梯度下溢，采用一种称为**损失缩放**的方法。核心理念很简单：在启动反向传播*之前*，将损失值乘以一个选定的因子 $S$。


$$
\text{缩放后的损失} = S \times \text{原始损失}
$$


根据链式法则，缩放损失也会将梯度按相同因子 $S$ 缩放：


$$
\frac{\partial \text{缩放后的损失}}{\partial w} = S \times \frac{\partial \text{原始损失}}{\partial w}
$$


这种乘法将小梯度值推入FP16的可表示范围，避免它们变为零。在梯度计算完毕并可能转换回FP32后，它们通过除以 $S$ 缩回原值，*之后*优化器再更新FP32主权重 (weight)：


$$
\nabla_{w} \text{原始损失} = \frac{1}{S} \nabla_{w} \text{缩放后的损失}
$$


缩放因子 $S$ 可以静态选择或动态确定。动态损失缩放会在训练期间调整 $S$。如果梯度在一定步数内没有溢出（变为`Inf`或`NaN`），则增加 $S$；如果检测到溢出，则显著降低 $S$。

下图说明了带损失缩放的混合精度训练步骤的流程：

> 典型的混合精度训练迭代中的操作流程，包括权重转换、计算、损失缩放以及权重更新前的梯度取消缩放。

## 优势与实现

混合精度训练的主要优势是：

- **训练时间缩短：** 通过更快的FP16计算实现，特别是在带有专用FP16单元（如Tensor Cores）的硬件上。根据模型、硬件和批次大小的不同，实现1.5倍到3倍或更高的加速很常见。
- **内存使用量降低：** 精度减半大约使训练期间存储的激活值和梯度所需的内存减半。这允许训练更大的模型或使用更大的批次大小，从而可以改善梯度质量，有时还会提升模型性能。



![训练性能比较](plots/2546-0.json)



> 比较显示了与标准FP32基准相比，使用混合精度时训练时间与GPU内存使用量可能出现的减少。实际改进情况因硬件和模型而异。

幸运的是，实现混合精度训练通常不需要手动管理转换和损失缩放。PyTorch（通过`torch.cuda.amp`）和TensorFlow（使用`tf.keras.mixed_precision`）等现代深度学习 (deep learning)框架提供了自动混合精度（AMP）的高级API。这些库自动处理FP16和FP32之间针对适当操作的转换，并管理动态损失缩放，使其启用起来相对简单。

例如，在PyTorch中，启用AMP可能涉及将前向传播计算封装在`autocast`上下文 (context)管理器中，并使用`GradScaler`对象来管理损失缩放和梯度更新。

尽管混合精度适用范围很广，但它在训练非常大的模型（如Transformer或用于高分辨率图像的大型CNN）、GPU内存成为瓶颈，或者在实验中需要快速迭代时尤其有用。通常建议验证使用混合精度训练的模型最终精度与FP32训练的模型精度相当，尽管现代AMP实现中，显著的性能下降并不常见。

## 参考资料

- [Mixed-Precision Training for Deep Neural Networks](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍深度神经网络混合精度训练的开创性论文，涵盖FP16的使用、损失缩放以防止下溢，以及其在带有Tensor Cores的NVIDIA GPU上的优势。
- [Automatic Mixed Precision for torch.cuda operations](https://pytorch.org/docs/stable/amp.html) — PyTorch Authors (2025)
  PyTorch官方文档，解释了如何通过`torch.cuda.amp.autocast`和`torch.cuda.amp.GradScaler`启用和使用自动混合精度（AMP），提供了实际的实现细节。
- [Mixed precision training](https://www.tensorflow.org/guide/mixed_precision) — TensorFlow Authors (2023)
  TensorFlow官方指南，介绍了如何在Keras中使用`tf.keras.mixed_precision.Policy`实现混合精度训练，详细说明了设置和使用方法以提升性能。
