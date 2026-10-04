---
course: "cnns-for-computer-vision"
chapter: "cnn-foundations-modern-architectures"
lesson: "architectural-design-tradeoffs"
sourceId: 2529
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-1-cnn-foundations-modern-architectures/architectural-design-tradeoffs"
title: "架构设计选择与权衡"
description: "讨论影响CNN架构设计的因素，包括计算成本、准确度和参数效率。"
order: 7
plots: ["plots/2529-0.json"]
sourceHash: "a466c98ab1b2b29151ff75e1cd14b4cb6976874e639d0fb2d0a6557bb420fead"
sourceCorrections: []
---

挑选合适的卷积神经网络 (neural network)（CNN）架构，或者设计一个定制架构，不仅仅是在基准数据集上选择报告准确度最高的模型。ResNet、DenseNet和EfficientNet等架构表明，每个设计都体现了特定的原则，并进行了隐含的权衡。选择架构需要仔细考虑目标应用、可用资源和性能要求。这本质上是在平衡相互竞争的因素。

### 核心三难选择：准确度、速度和大小

架构设计的核心在于三个期望特性之间的基本张力：

1. **准确度：** 首要目标通常是最大化模型在目标任务上的性能（例如，分类准确度、检测平均精度（mAP）、分割交并比（IoU））。更复杂的、容量更大的架构通常能够提供更高的准确度，前提是有足够的数据和适当的训练。
2. **计算成本（速度）：** 这与前向传播所需的计算量有关，通常以浮点运算次数（FLOPs）或乘加运算次数（MACs）衡量。较低的计算成本意味着更快的推理 (inference)速度和潜在更快的训练速度，这对于实时应用或在有限硬件上进行训练非常重要。
3. **模型大小（参数 (parameter)与内存）：** 这包括可训练参数的数量（影响存储大小）以及训练和推理期间的峰值内存使用（激活值）。更小的模型更容易部署，特别是在边缘设备或手机上，并且在运行时消耗更少的内存。

很少有架构能同时在所有三个方面表现出色。提升一个方面通常会以牺牲另一个方面为代价。例如，增加网络深度或宽度通常会提高准确度潜力，但也会增加计算成本和模型大小。

### 影响架构选择的因素

在选择或设计CNN架构时，有几个因素会影响决策过程：

- **任务要求：** 计算机视觉任务的性质在很大程度上决定了架构需求。
  - **图像分类：** 通常受益于能有效学习分层特征的深层架构。全局信息很重要。
  - **目标检测：** 需要定位能力。架构通常需要高分辨率特征图和处理多尺度目标（例如，基于骨干架构构建的特征金字塔网络）的机制。
  - **语义/实例分割：** 需要密集的像素级预测。编码器-解码器结构（如U-Net）或空洞卷积（如DeepLab）是常用的方法，以在保持空间分辨率的同时增大感受野。
- **数据集特点：** 训练数据的大小和性质起着重要作用。
  - **大型数据集（例如ImageNet）：** 可以支持非常深、高容量的模型，而不会过度拟合。像ResNet-152或EfficientNet-B7这样的架构在此类数据上表现出色。
  - **小型或专用数据集：** 使用大型模型时容易出现过拟合 (overfitting)。策略包括使用较浅的网络、采用强正则化 (regularization)，或大量依赖从在大型数据集上预训练 (pre-training)的模型进行迁移学习 (transfer learning)。预训练骨干的选择在这里变得很重要。
- **计算预算：** 硬件限制是一个主要约束。
  - **高性能GPU：** 可以容纳计算密集型模型，如大型ResNets或Transformers。训练时间可能仍然是一个考虑因素。
  - **移动/边缘设备：** 需要高效的架构，具有低FLOPs和低延迟，例如MobileNets、ShuffleNets或较小的EfficientNets。推理 (inference)速度（例如，每秒帧数）通常是重要的衡量标准。
- **内存限制：**
  - **训练：** 深层网络或像DenseNet这样的架构可能由于需要存储反向传播 (backpropagation)的激活值而产生高内存需求。梯度检查点或混合精度训练等技术可以帮助缓解这一问题。
  - **部署：** 模型大小（参数 (parameter)数量）决定了存储需求。推理期间的峰值内存使用（激活值）对于资源受限的设备很重要。

### 通过权衡重新评估架构特性

让我们重新审视前面讨论的架构创新，从这些权衡的角度来看待它们：

- **深度（例如VGG、ResNet）：** 增加深度可以学习更复杂的特征层次结构，从而可能提高准确度。然而，它增加了顺序计算，可能减慢推理 (inference)速度。ResNet的跳跃连接缓解了非常深层网络的梯度消失问题，使得网络可以增加深度而不牺牲可训练性，但更深的ResNet仍然需要更多的计算。
- **宽度（通道）：** 更宽的层增加了容量，可以捕捉特征图中更精细的细节。这会大幅增加FLOPs（在标准卷积中，与宽度呈二次方关系）和参数 (parameter)数量。Network-in-Network和Inception模块尝试使用$1 \times 1$卷积来管理宽度并高效地执行通道级池化或投影。
- **分辨率：** 更高的输入分辨率提供更多的空间细节，有利于需要定位或细粒度识别的任务。然而，计算成本通常与输入分辨率呈二次方关系。EfficientNet强调了平衡分辨率与深度和宽度的必要性。
- **跳跃连接（ResNet）：** 主要改善可训练性并允许更大的深度。恒等快捷连接增加了极少的计算开销和参数，但通过要求存储中间特征图（$x$）以进行加法 $y = \mathcal{F}(x) + x$ 来增加内存使用。
- **密集连接（DenseNet）：** 通过促进特征重用，实现了高参数效率（用更少的参数获得良好准确度）。由于特征图增长较小，计算可能很高效。主要缺点是高内存消耗，因为需要保留所有前置层的中间特征图以进行拼接。
- **多分支设计（Inception）：** 有效地捕捉多尺度特征，通常能带来良好的准确度。如果设计得当（例如，使用$1 \times 1$卷积进行降维），计算可能很高效。复杂之处在于模块结构的设计和调整。
- **复合缩放（EfficientNet）：** 提供了一种有原则的方法，可以同时平衡深度、宽度和分辨率，与仅缩放一个维度相比，实现了更好的准确度-效率权衡。它提供了一系列模型（B0-B7），允许用户根据其资源限制进行选择。

### 权衡空间的视觉呈现

我们可以直观地展示不同模型系列在准确度和效率之间的关系。下面的图表显示了ImageNet Top-1准确度与计算成本（FLOPs）的简化视图。目标是左上角（高准确度，低FLOPs）的模型代表了更高的效率。



![准确度与计算成本对比 (ImageNet)](plots/2529-0.json)



> ImageNet Top-1准确度与来自不同架构家族的选定模型的计算成本（FLOPs，对数坐标）对比图。EfficientNet体现出有利的权衡曲线。

类似图表可以针对特定硬件的准确度与参数 (parameter)数量或准确度与推理 (inference)延迟创建。这些直观呈现有助于在资源限制下比较架构。

### 实际实施考量

实际方面影响选择：

- **预训练 (pre-training)权重 (weight)的可用性：** 使用在大型数据集（如ImageNet）上预训练的模型是迁移学习 (transfer learning)的标准做法。在您选择的框架（TensorFlow、PyTorch）中，给定架构的预训练权重的可用性和质量是一个主要因素。
- **实现和修改的便捷性：** 某些架构比其他架构更简单、更容易实现或调整。流行库中找到的标准构建块通常倾向于ResNet等成熟架构。
- **训练稳定性和超参数 (parameter) (hyperparameter)：** 一些复杂架构可能对超参数选择（学习率、优化器、权重初始化）更敏感，并且需要更仔细的调整以实现稳定训练。
- **对下游任务的适应性：** 尽管ImageNet分类是一个常见基准，但请评估骨干架构的特征如何很好地迁移到您的特定下游任务（例如，检测、分割）。某些架构学习到的特征可能更适合特定应用。

总之，选择CNN架构涉及在多维度的权衡空间中进行取舍。没有适用于所有情况的单一“最佳”架构。最优选择在很大程度上取决于您项目的具体限制和目标，包括目标任务、数据集、可用计算资源和期望的性能指标。了解不同架构的设计原则和固有折衷使您能够做出明智的决策。后续章节将介绍模型压缩和自动化神经架构搜索（NAS）等技术，这些技术明确旨在优化这些权衡。

## 参考资料

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (2015)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 770-778; DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)
  提出残差学习和跳跃连接，使训练深度神经网络以提高准确性成为可能。
- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) — Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger (2017)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 4700-4708; DOI: [10.48550/arXiv.1608.06993](https://doi.org/10.48550/arXiv.1608.06993)
  提出一种前馈连接所有层的网络，促进特征复用并减少参数数量。
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) — Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen (2018)
  Journal: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 4510-4520; DOI: [10.48550/arXiv.1801.04381](https://doi.org/10.48550/arXiv.1801.04381)
  描述一种适用于移动和边缘设备的有效架构，通过倒残差结构实现低计算成本和参数数量。
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) — Mingxing Tan, Quoc V. Le (2019)
  Journal: International Conference on Machine Learning; Pages: 6105-6114; DOI: [10.48550/arXiv.1905.11946](https://doi.org/10.48550/arXiv.1905.11946)
  提出一种复合缩放方法，统一缩放网络深度、宽度和分辨率，实现更好的准确性和效率平衡。
