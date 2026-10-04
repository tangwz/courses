# 直通估计器 (STE)

来源：[原文](https://apxml.com/zh/courses/practical-llm-quantization/chapter-4-quantization-aware-training-qat/straight-through-estimator-ste)

[返回章节目录](README.md) · [返回课程目录](../README.md)

量化 (quantization)感知训练 (QAT) 在训练或微调 (fine-tuning)过程中模拟量化效果。这种模拟通过在模型的计算图中插入操作（常被称为“伪量化”节点）来实现。这些节点接收高精度输入（如 FP32 激活值或权重 (weight)），并生成低精度输出（例如模拟 INT8 或 INT4），这些输出随后用于后续操作。

然而，这给依赖梯度下降 (gradient descent)的训练算法带来了重要难题。伪量化中的核心操作是舍入函数，它将连续的输入范围映射到离散的输出值集合。从数学上看，这个函数是一个阶梯函数。

考虑一个简单的舍入函数 Round(x)。它的导数在几乎所有地方（平坦的阶梯上）都为零，并在值跳变点处未定义。标准的反向传播 (backpropagation)依赖于计算梯度来更新模型权重。如果梯度为零或未定义，更新就无法通过量化操作反向传播到之前的层和权重。这会有效阻碍位于量化节点之前的参数 (parameter)的学习过程。

如果梯度被阻碍，模型如何学习适应量化呢？这时，**直通估计器 (STE)** 便发挥作用了。

### 问题所在：不可微分的量化 (quantization)

我们来表示量化函数（包括缩放、舍入以及反量化回模拟低精度步骤的浮点数）为 $y = q(x)$。在训练的前向传播中，我们从输入 $x$ 计算 $y$，并在后续计算中使用 $y$。

在反向传播 (backpropagation)过程中，我们需要计算损失 $L$ 对输入 $x$ 的梯度，表示为 $\frac{\partial L}{\partial x}$。使用链式法则，这通常会这样计算：


$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}
$$


问题出在项 $\frac{\partial y}{\partial x} = \frac{\partial q(x)}{\partial x}$。如前所述，这个导数有问题（大部分为 0，在跳变点未定义）。如果我们使用这个真实梯度，$\frac{\partial L}{\partial x}$ 在几乎所有地方都变为零，阻止了权重 (weight)更新。

### 解决方案：直通估计器 (STE)

直通估计器 (STE) 为这个问题提供了一个实际可行的变通方法。它是在反向传播 (backpropagation)过程中专门使用的近似。核心思想很简单：

1. **前向传播：** 照常计算量化 (quantization)操作：$y = q(x)$。量化效果（精度损失）被完全应用。
2. **反向传播：** 在计算梯度 $\frac{\partial L}{\partial x}$ 时，*忽略* $q(x)$ 的真实导数，转而进行*近似*。最常见的 STE 近似是将梯度 $\frac{\partial y}{\partial x}$ 设为 1。

因此，在使用 STE 的反向传播中，梯度计算变为：


$$
\frac{\partial L}{\partial x} \approx \frac{\partial L}{\partial y} \cdot 1 = \frac{\partial L}{\partial y}
$$


这意味着为量化节点输出计算的梯度 $\frac{\partial L}{\partial y}$ 被“直通”传递，成为用于输入的梯度 $\frac{\partial L}{\partial x}$，如同量化操作是一个恒等函数（$y=x$）*仅用于梯度计算*。

> 此图表展示了 STE 过程。前向传播应用实际的量化 $q(x)$。反向传播使用 STE 近似（$\frac{\partial y}{\partial x} \approx 1$），使传入梯度 $\frac{\partial L}{\partial y}$ 不变地通过，成为 $\frac{\partial L}{\partial x}$。

### STE 为何有效？

在反向传播 (backpropagation)过程中使用一个忽略量化 (quantization)函数真实属性的近似，这看起来可能与直觉相悖。然而，STE 在实际中表现良好，原因有几点：

- **梯度流动：** 它确保梯度不被阻碍，使优化信号能够到达量化节点之前的权重 (weight)。
- **学习适应性：** 尽管梯度计算是近似的，但前向传播仍然使用*实际*的量化值。损失函数 (loss function)反映了这种量化引入的误差。通过允许梯度流动，STE 使模型权重能够以最小化这种量化引起的误差的方式调整。网络学习到的参数 (parameter)对前向传播中应用的舍入效应更具弹性。
- **简洁性：** STE 在标准深度学习 (deep learning)框架中易于实现。大多数提供 QAT 能力的框架都在内部集成了 STE。

### STE 的变体

虽然恒等近似（$\frac{\partial y}{\partial x} = 1$）是 STE 最常见的形式，但也存在一些变体。例如，另一种方法涉及根据量化 (quantization)所用的输入范围剪裁梯度。如果量化函数在量化前将输入 $x$ 有效地剪裁到范围 $[c_{min}, c_{max}]$ 内，那么 STE 可以定义为：


$$
\frac{\partial y}{\partial x} =
\begin{cases}
1 & \text{如果 } c_{min} \le x \le c_{max} \\
0 & \text{否则}
\end{cases}
$$


这种变体阻止了对于已经超出量化范围的输入，梯度反向传播 (backpropagation)，这有时有助于稳定训练。然而，简单的恒等近似通常就足够了。

总之，直通估计器是使量化感知训练成为可能的底层技术。通过为梯度提供一条通过不可微分量化操作的路径，它使深度学习 (deep learning)模型能够在训练或微调 (fine-tuning)期间调整其权重 (weight)，与训练后量化相比，这能使量化模型获得明显更好的精度，尤其是在极低比特宽度时。

## 参考资料

- [Quantized Neural Networks: Training Neural Networks with Low-Precision Weights and Activations](https://arxiv.org/abs/1609.07061) — Matthieu Courbariaux, Itay Hubara, Daniel Soudry, Ran El-Yaniv, Yoshua Bengio (2016)
  Journal: Advances in Neural Information Processing Systems; Publisher: Curran Associates, Inc.; Volume: 29; Pages: 3745-3753
  这篇有影响力的论文将直通估计器应用于训练具有极低精度权重和激活的神经网络，为量化感知训练奠定了重要基础。
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://openaccess.thecvf.com/content_cvpr_2018/papers/Jacob_Quantization_and_Training_CVPR_2018_paper.pdf) — Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, Dmitry Kalenichenko (2018)
  Journal: 2018 IEEE Conference on Computer Vision and Pattern Recognition; Publisher: IEEE; Pages: 2704-2713; DOI: [10.1109/CVPR.2018.00794](https://doi.org/10.1109/CVPR.2018.00794)
  一篇开创性论文，描述了量化感知训练（QAT）方法，该方法在训练期间明确使用直通估计器来模拟量化，从而实现高精度的低比特推理。

---

[上一节](02-%E8%AE%AD%E7%BB%83%E6%9C%9F%E9%97%B4%E6%A8%A1%E6%8B%9F%E9%87%8F%E5%8C%96%E5%BD%B1%E5%93%8D.md) · [下一节](04-%E4%BD%BF%E7%94%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A1%86%E6%9E%B6%E5%AE%9E%E7%8E%B0%E9%87%8F%E5%8C%96%E6%84%9F%E7%9F%A5%E8%AE%AD%E7%BB%83.md)
