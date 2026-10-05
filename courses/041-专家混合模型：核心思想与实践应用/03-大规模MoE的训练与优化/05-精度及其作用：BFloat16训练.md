# 精度及其作用：BFloat16训练

来源：[原文](https://apxml.com/zh/courses/mixture-of-experts-advanced-implementation/chapter-3-training-large-scale-moes/precision-bfloat16-training)

[返回章节目录](README.md) · [返回课程目录](../README.md)

训练一个参数 (parameter)量达到数十亿甚至数万亿的专家混合模型（Mixture of Experts）对现代硬件构成了严峻挑战。权重 (weight)、激活和梯度的庞大规模会占用大量GPU内存，并对计算吞吐量 (throughput)提出高要求。尽管分布式训练策略可以分割模型，但优化数据本身的数值格式提供了一种辅助且有效的方法来提升效率。正因如此，低精度数据类型，特别是BFloat16，变得必不可少。

在大多数框架中，对所有计算使用完整的32位浮点精度（`FP32`）是默认设置，它提供了宽广的动态范围和高精度。然而，每个`FP32`参数需要4字节存储空间。对于一个拥有5000亿参数的模型，仅权重就将占用2太字节内存，这使其无法在任何单个加速器上运行。

解决方案是采用低精度格式。通过减少用于表示每个数字的比特数，我们可以大幅降低内存占用，并在硬件支持下加速计算。

### 浮点数值类型

过去，`FP32`的主要替代方案是`FP16`（半精度），它使用16位。尽管`FP16`成功地将内存消耗减半，但它有一个显著缺点：动态范围非常有限。其较小的指数分配使其在训练期间容易出现数值不稳定。梯度可能非常小或非常大，容易变为零（下溢）或无穷大（上溢），从而使训练过程不稳定或中止。

这促成了BFloat16（`BF16`），即“脑浮点”格式的出现，这是一种专为深度学习 (deep learning)工作负载设计的格式。与`FP16`一样，它也只使用16位。然而，它做出了不同的权衡。`BF16`为指数分配了与`FP32`相同数量的比特，从而保持了其宽广的动态范围。这代价是尾数（或小数部分）的比特数减少，而尾数负责精度。

下图显示了这三种格式的位分配。

> `FP32`、`BF16`和`FP16`位结构的比较。`BF16`保留了`FP32`的8个指数位，确保了相似的动态范围，而`FP16`牺牲了指数位以获得更多的尾数（精度）位。

对于深度学习来说，`BF16`的宽动态范围远比高精度重要。神经网络 (neural network)对权重 (weight)和激活中的噪声及较低精度表现出很强的适应性。通过防止`FP16`常见的下溢和上溢问题，`BF16`提供了一个更稳定的训练环境，通常可以近乎直接替代`FP32`。

### 使用BFloat16的混合精度训练

虽然可以简单地将整个模型转换为`BF16`，但一种更有效的方法，即**混合精度训练**，是标准做法。这种方法将`BF16`的速度和内存优势与`FP32`在训练循环重要部分的稳定性结合起来。

使用`BF16`的典型混合精度工作流程如下：

1. **在FP32中保留主权重 (weight)：** 模型的权重主副本以完整的`FP32`精度存储。这作为数据的权威来源，确保小的梯度更新不会因`BF16`的较低精度而丢失。
2. **将权重转换为BF16进行计算：** 对于每个训练步骤，`FP32`主权重都会被转换为`BF16`。
3. **在BF16中执行前向和反向传播 (backpropagation)：** 前向和反向传播中的所有矩阵乘法及其他计算都使用`BF16`权重和激活执行。现代GPU拥有专用硬件，例如NVIDIA的Tensor Cores，可以显著加速`BF16`操作。
4. **更新FP32主权重：** 所得的梯度（为`BF16`格式）随后用于更新`FP32`主权重副本。

> 标准混合精度训练步骤中的数据流。计算在`BF16`中加速，而权重更新在`FP32`中执行以保持稳定性。

这种方法兼具两者优点：16位计算的内存和速度优势，以及32位权重更新的数值稳定性。对于MoE模型而言，这不仅是一种优化；更是一项实现途径。将权重和激活的内存占用减半，使得训练更大、专家数量更多的模型变得可行。

### 在PyTorch中的实际实现

像PyTorch这样的深度学习 (deep learning)框架提供了简单的上下文 (context)管理器来自动化混合精度训练。如果您的硬件支持`BF16`（例如NVIDIA A100或H100系列GPU），启用它异常简单。

以下是使用`torch.autocast`的典型训练循环示例。

```python
import torch

# 确保您的模型和数据位于支持BF16的设备上
device = "cuda" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "cpu"
model = MyMoEModel().to(device)
optimizer = torch.optim.AdamW(model.parameters())
data = torch.randn(64, 1024, device=device) # 示例数据

# torch.autocast 上下文管理器会自动处理
# 将符合条件的操作转换为指定的dtype。
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    # 模型的正向传播以BF16运行
    output, aux_loss = model(data)

    # 损失计算也可以在autocast上下文内部进行
    main_loss = loss_fn(output, target)
    total_loss = main_loss + aux_loss

# 梯度是基于BF16正向传播计算的
# .backward() 调用发生在autocast上下文之外
total_loss.backward()

# 优化器更新主FP32权重
optimizer.step()
optimizer.zero_grad()
```

请注意其简便性。`autocast`上下文管理器自动处理操作到`BF16`的转换。当使用`FP16`训练时，需要一个名为`GradScaler`的额外组件来缩放损失，以防止梯度下溢。由于`BF16`拥有更大的动态范围，这个损失缩放步骤通常不需要，这进一步简化了训练代码，并减少了一个需要调整的超参数 (parameter) (hyperparameter)。这种固有的稳定性使得`BF16`成为训练MoE等庞大而复杂模型的首选。

## 参考资料

- [BFloat16: The Secret to High Performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) — Shibo Wang, Pankaj Kanwar (2019)
  Publisher: Google Cloud Blog
  解释了BFloat16的设计，其宽动态范围对深度学习的作用，以及在专用硬件上高效训练大型模型的益处。
- [Mixed-Precision Training](https://arxiv.org/abs/1710.03740) — Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (2018)
  Journal: ICLR 2018; DOI: [10.48550/arXiv.1710.03740](https://doi.org/10.48550/arXiv.1710.03740)
  介绍了混合精度训练的技术，包括使用FP32主权重和FP16计算，BFloat16训练对此框架进行了扩展。
- [Automatic Mixed Precision training](https://pytorch.org/docs/stable/amp.html) — PyTorch Developers (2025)
  Publisher: PyTorch.org
  PyTorch官方关于自动混合精度(AMP)的文档，详细说明了如何使用`torch.autocast`进行高效的BFloat16训练。
- [High-Performance Mixed-Precision Training for Deep Learning](https://ieeexplore.ieee.org/document/8916335) — Minseok Park, George K. Lee, Yunsup Lee, Michael O. Lee (2019)
  Journal: 2019 IEEE High Performance Extreme Computing Conference (HPEC); Publisher: IEEE; Pages: 1-7; DOI: [10.1109/HPEC.2019.8916335](https://doi.org/10.1109/HPEC.2019.8916335)
  讨论了在深度学习硬件加速器环境下，混合精度训练（包括BFloat16）的实现和优势。

---

[上一节](04-%E7%BC%93%E8%A7%A3%E8%B7%AF%E7%94%B1%E5%99%A8Z%E6%8D%9F%E5%A4%B1%E4%B8%8D%E7%A8%B3%E7%9A%84%E5%8A%9E%E6%B3%95.md) · [下一节](06-%E9%A2%84%E8%AE%AD%E7%BB%83MoE%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%BE%AE%E8%B0%83%E7%AD%96%E7%95%A5.md)
