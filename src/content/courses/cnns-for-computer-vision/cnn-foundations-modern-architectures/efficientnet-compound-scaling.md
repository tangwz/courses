---
course: "cnns-for-computer-vision"
chapter: "cnn-foundations-modern-architectures"
lesson: "efficientnet-compound-scaling"
sourceId: 2527
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures/efficientnet-compound-scaling"
title: "EfficientNet：模型复合缩放"
description: "了解EfficientNet及复合缩放原理，用于平衡网络深度、宽度和分辨率。"
order: 6
plots: ["plots/2527-0.json"]
sourceHash: "e2ad4c8fbeb2057d6076eaa47d94b8068a8b0983c84f1cced969cd88e092daad"
sourceCorrections: []
---

ResNet和DenseNet等架构已经通过增加网络深度和改进梯度流获得了显著的性能提升。然而，仅仅加深或加宽网络并不总能带来最佳性能或效率。卷积神经网络 (neural network)（CNN）发展中的一个主要问题是：是否存在一种更有章可循的方法来扩展CNN，以获得更好的准确性和效率？EfficientNet通过一种称为**复合缩放**的策略给出了一个令人信服的答案。

### CNN缩放的挑战

传统上，CNN通过以下三个维度之一进行缩放：

1. **深度：** 增加层数（例如，从ResNet-18到ResNet-200）。更深的网络可以捕获更复杂的特征，但会受到梯度消失的困扰，并且需要残差连接等技术。
2. **宽度：** 增加每层中的通道数或滤波器数量（例如，Wide ResNets）。更宽的网络可以捕获更精细的特征，并且通常更容易训练，但计算成本随宽度呈二次方增长，这可能令人望而却步。
3. **分辨率：** 使用更高分辨率的输入图像。更高分辨率的输入提供更多细节，但会增加计算成本，并且需要具有更大感受野的网络（通常通过更多层实现）才能有效处理。

仅缩放其中一个维度通常会导致收益递减。例如，在不增加宽度或输入分辨率的情况下，使网络过深可能难以捕获足够的空间细节。同样，加宽一个较浅的网络可能无法发挥复杂特征层级的潜力。EfficientNet的作者们发现，这些维度是相互依存的，应平衡以获得最佳结果。

### 复合缩放：平衡深度、宽度和分辨率

EfficientNet的核心思想是**复合缩放**。EfficientNet建议不要任意缩放一个维度，而是使用一组固定的缩放系数同时缩放网络深度（$d$）、宽度（$w$）和图像分辨率（$r$）。

缩放由一个单一的复合系数，$\phi$控制。给定一个基准网络（EfficientNet-B0），扩展过程涉及增加$\phi$。深度、宽度和分辨率根据以下规则进行缩放：

- 深度：$d = \alpha^\phi$
- 宽度：$w = \beta^\phi$
- 分辨率：$r = \gamma^\phi$

需满足以下约束条件：


$$
\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$


其中，$\alpha$、$\beta$和$\gamma$是通过对基准模型进行网格搜索确定的常数。它们表示每个维度的缩放程度。约束条件$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$确保每当$\phi$增加时，所需的总浮点运算次数（FLOPS）大约增加$2^\phi$。

直观地说，如果输入图像分辨率（$r$）增加，网络需要更多层（深度$d$）来增大感受野，并需要更多通道（宽度$w$）来捕获更大输入图像中更精细的模式。复合缩放提供了一种系统性的方法来平衡这些因素。

### EfficientNet-B0基准模型与MBConv

复合缩放的有效性依赖于一个强大且高效的基准架构。EfficientNet使用通过神经架构搜索（NAS）找到的基准网络（EfficientNet-B0）。这次搜索对准确性和FLOPS都进行了优化。

EfficientNet-B0的核心构建块是**移动倒置瓶颈卷积（MBConv）**，它与MobileNetV2中使用的相似，但可能通过Squeeze-and-Excitation（SE）优化进行了增强。一个MBConv块通常包含：

1. 一个$1 \times 1$卷积，用于扩展通道数量（“倒置”部分，与ResNet的瓶颈层相反）。
2. 一个深度可分离卷积，用于高效地执行空间滤波。
3. 一个Squeeze-and-Excitation模块（可选，但在EfficientNet中使用），用于执行通道注意力，重新校准通道特征响应。
4. 一个$1 \times 1$卷积，用于将通道投影回。
5. 如果输入和输出维度匹配，则使用跳跃连接（类似于ResNet）。

### EfficientNet系列模型

通过从EfficientNet-B0开始，并应用复合缩放规则，增加复合系数$\phi$的值（通常是从1开始的整数值），可以生成一系列模型（EfficientNet-B1、B2、...、B7等）。随后的每个模型使用的FLOPS大约是前一个的两倍，但目标是更高的准确性，同时保持较高的参数 (parameter)效率。



![准确性与计算量权衡](plots/2527-0.json)



> 比较图表展示了EfficientNet模型（蓝色）在给定计算量下，通常比传统缩放的模型（灰色）获得更高的准确性。

### 优点与考虑事项

EfficientNet在发布时，在ImageNet和几个其他迁移学习 (transfer learning)基准测试中展现了最先进的性能，通常以显著更少的参数 (parameter)和更低的计算成本（FLOPS）获得与大得多的模型相当的准确性。

- **效率：** 它们的主要优势是每参数和每FLOP的优异准确性。
- **可扩展性：** 复合缩放方法为从精心设计的基准模型生成更大、更强大的模型提供了明确的指导。
- **迁移学习：** 由于其效率和出色的性能，预训练 (pre-training)的EfficientNet是迁移学习任务的绝佳选择。

一些考虑事项包括：

- **训练复杂性：** 尽管在推理 (inference)时高效，但训练非常大的EfficientNet仍然可能资源密集，可能受益于第2章讨论的混合精度训练等技术。
- **对NAS的依赖：** 基准架构和缩放因子是通过NAS发现的，这在计算上非常昂贵。然而，实际使用者通常使用预定义的B0-B7模型和缩放因子。

EfficientNet在设计高效且可扩展的CNN架构方面代表着一个重要的进展。通过复合缩放仔细平衡深度、宽度和分辨率，它提供了一个强大的框架，用于开发在计算机视觉中提升准确性和效率边界的模型。预训练版本在TensorFlow（通过`tf.keras.applications.EfficientNetB0`到`B7`）和PyTorch（通常通过`timm`等第三方库）等框架中广泛可用，使其易于用于实际任务。

## 参考资料

- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) — Mingxing Tan, Quoc V. Le (2019)
  Journal: International Conference on Machine Learning; DOI: [10.48550/arXiv.1905.11946](https://doi.org/10.48550/arXiv.1905.11946)
  介绍复合缩放方法和EfficientNet模型系列。
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (2018)
  Journal: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 4510-4520; DOI: [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)
  提出移动倒置瓶颈卷积（MBConv）模块，它是EfficientNet的核心组成部分。
- [Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578) — Barret Zoph, Quoc V. Le (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1611.01578](https://doi.org/10.48550/arXiv.1611.01578)
  关于使用强化学习进行神经网络架构搜索的基础论文，该方法用于发现EfficientNet-B0。
