---
course: "cnns-for-computer-vision"
chapter: "image-segmentation-techniques"
lesson: "deeplab-aspp"
sourceId: 2578
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-4-image-segmentation-techniques/deeplab-aspp"
title: "DeepLab 系列：空洞空间金字塔池化"
description: "了解 DeepLab 架构版本和空洞空间金字塔池化 (ASPP) 模块。"
order: 5
plots: []
sourceHash: "96a692de4feda66e2314d863deac00d99f462e5bcbebc72f1832fe2b0c4130dd"
sourceCorrections: []
---

基于空洞卷积（或称膨胀卷积）的概念，这类卷积能够在不增加参数 (parameter)数量或显著降低空间分辨率的情况下，扩大滤波器的感受野。在此基础上，我们现在考察如何有效地在多个尺度上获取上下文 (context)信息。虽然单个空洞卷积能扩大感受野，但图像中的对象可能以多种尺寸出现，这使得模型需要同时理解不同空间范围内的上下文。对于复杂场景，简单地使用固定膨胀率的空洞卷积可能不够充分。

由谷歌研究人员开发的 DeepLab 系列模型，代表了一系列具有影响力的架构，它们专门设计用于解决语义分割中的多尺度问题。DeepLab 各版本（v1、v2、v3、v3+）引入并改进的核心创新是空洞空间金字塔池化（ASPP）。

### 空洞空间金字塔池化 (ASPP)

ASPP 通过使用多个以不同膨胀率并行操作的滤波器来处理输入的卷积特征层，以此应对多尺度问题。这能同时有效地获取多个不同尺度下的图像上下文 (context)。

其核心思想包括：

1. **并行空洞卷积：** 对相同的输入特征图应用多个具有不同膨胀率（例如，率分别为6、12、18）的并行空洞卷积层。每个膨胀率会从每个特征点周围不同大小的区域获取信息。
2. **1x1 卷积：** 并行包含一个标准的1x1卷积。这有助于在原始尺度上获取精细信息。
3. **图像级特征（全局上下文）：** 包含图像级上下文通常通过一个全局平均池化分支实现。输入特征图被池化成一个单一特征向量 (vector)，然后通过一个1x1卷积（通常带有批归一化 (normalization)和ReLU激活），再双线性上采样回输入特征图的空间尺寸。这个分支提供全局概括信息。
4. **拼接与合并：** 所有并行分支（空洞卷积、1x1卷积、图像池化）产生的特征图沿通道维度进行拼接。
5. **最终处理：** 这个组合后的特征图通常会通过一个最终的1x1卷积（同样常带有批归一化和ReLU），以汇合多尺度信息并降低通道维度，从而在分割预测层之前生成最终的特征表示。

ASPP 模块的结构可以如下所示：

> 空洞空间金字塔池化（ASPP）模块的结构。输入特征图由具有不同特点（1x1卷积、不同膨胀率的空洞卷积、图像池化）的并行分支处理，然后进行拼接和合并。

### DeepLab 架构变体

DeepLab 模型通常使用强大的CNN分类器（如ResNet或Xception）作为骨干网络，但会对其进行修改以适应密集预测任务。这通常包括：

- 将最终的全连接层替换为卷积层。
- 移除后续的池化层或改变其步幅。
- 在骨干网络的后期阶段采用空洞卷积，以在不牺牲感受野的情况下保持更高的空间分辨率输出（更大的特征图）。例如，输出步幅可能从32（分类中常见）减小到16或8。

然后将 ASPP 模块应用于这个修改后的骨干网络提取的特征。

- **DeepLabv3：** 在DeepLabv2的基础上改进，通过使用具有不同膨胀率的ASPP和图像级特征，更可靠地整合多尺度上下文 (context)。它通常在ASPP内部应用批归一化 (normalization)。
- **DeepLabv3+：** 通过添加一个简单但有效的解码器模块，进一步增强了架构。这个解码器通常获取ASPP输出中丰富的语义特征，并将其与骨干网络早期阶段的低级特征（包含更精细的空间细节）结合。这种结合过程通常涉及对ASPP输出进行上采样并将其与低级特征（在经过1x1卷积进行通道降维后）进行拼接，然后通过几个卷积层来完善分割图，尤其改善了物体边界处的预测。

### 优点

DeepLab 方法，特别是结合 ASPP 后，为语义分割带来了显著的优点：

- **多尺度上下文 (context)：** 明确地在多个尺度上获取信息，使模型能有效应对物体尺寸的变化。
- **受控特征分辨率：** 采用空洞卷积计算密集特征图，与单独的传统上采样/反卷积方法相比，无需过多的内存或计算。
- **先进的性能：** DeepLab 变体在 PASCAL VOC 2012 和 Cityscapes 等标准语义分割基准测试中持续取得了出色的成果。

通过将强大的骨干网络与 ASPP 的多尺度上下文聚合能力以及可能的精修解码器结合，DeepLab 系列为实现图像中细致的像素级理解提供了一个强大的框架。在实现或使用 DeepLab 模型时，必须仔细考虑骨干网络的选择、ASPP 中使用的具体膨胀率以及解码器的结构（如果使用），因为这些元素会影响性能和计算成本。

## 参考资料

- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://openreview.net/forum?id=S1L2v_CqYp) — Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam (2017)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1706.05587](https://doi.org/10.48550/arXiv.1706.05587)
  介绍了带有图像级特征和批量归一化的空洞空间金字塔池化 (ASPP)，构成了 DeepLabv3 的基础。
- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611) — Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam (2018)
  Journal: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: Institute of Electrical and Electronics Engineers (IEEE); Pages: 801-809; DOI: [10.1109/CVPR.2018.00696](https://doi.org/10.1109/CVPR.2018.00696)
  提出了 DeepLabv3+，通过引入高效的解码器模块，结合高级和低级特征，改善了边界分割效果，从而扩展了 DeepLabv3。
- [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://ieeexplore.ieee.org/document/7946252) — Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, Alan L Yuille (2017)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI); Publisher: IEEE; Volume: 40; Pages: 834-848; DOI: [10.1109/TPAMI.2017.2699184](https://doi.org/10.1109/TPAMI.2017.2699184)
  一篇基础论文，完善了 DeepLab 架构，详细阐述了空洞卷积及其在语义分割中用于多尺度上下文的应用。
- [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122) — Fisher Yu, Vladlen Koltun (2016)
  Journal: International Conference on Learning Representations (ICLR); DOI: [10.48550/arXiv.1511.07122](https://doi.org/10.48550/arXiv.1511.07122)
  介绍了空洞卷积，用于聚合多尺度上下文信息而不损失分辨率，这是 DeepLab 系列模型所利用的核心技术。
