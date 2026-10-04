---
course: "cnns-for-computer-vision"
chapter: "model-compression-efficient-dl"
lesson: "designing-efficient-architectures"
sourceId: 2636
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-8-model-compression-efficient-dl/designing-efficient-architectures"
title: "设计高效架构"
description: "Study CNN architectures specifically designed for efficiency, such as MobileNets and ShuffleNets."
order: 5
plots: []
sourceHash: "4ef02afd4ed2ec7ffd4e5596c02651b6f3423e30a584a50eb17e3573dc55aa09"
sourceCorrections: []
---

虽然剪枝和量化 (quantization)等技术修改现有的模型使其更小、更快，但另一种方法是从一开始就设计高效的网络架构。这些架构包含专门设计用于最小化计算成本（以浮点运算或FLOPs衡量）和参数 (parameter)数量的构建块，使其适用于部署到移动设备、嵌入 (embedding)式系统及其他资源有限的环境。

### 高效架构设计的核心原则

许多高效架构的核心思想是将计算开销大的操作（例如标准卷积层）替换为更经济的替代方案，这些替代方案能有效模拟其功能。有两种基本技术表现突出：

1. **深度可分离卷积：** 这可能是高效CNN设计中影响最大的创新，由MobileNet推广。标准的$3 \times 3$卷积同时处理所有输入通道以生成每个输出特征图。深度可分离卷积将其分为两个不同的步骤：

   - **逐深度卷积：** 空间卷积独立地应用于*每个输入通道*。如果有$C_{in}$个输入通道，则使用$C_{in}$个单独的$K \times K$滤波器（例如，$3 \times 3$）。此步骤过滤每个通道内的空间信息，但不组合跨通道的信息。
   - **逐点卷积：** $1 \times 1$卷积应用于跨通道。此步骤获取逐深度卷积的输出，并组合跨通道的信息以生成最终的$C_{out}$个输出通道。它使用$C_{out}$个大小为$1 \times 1 \times C_{in}$的滤波器。

   计算节省非常可观。对于具有$C_{in}$个输入通道、$C_{out}$个输出通道、滤波器大小为$K \times K$以及特征图大小为$H \times W$的标准卷积，其成本近似为：
   $\text{标准成本} \approx H \times W \times K \times K \times C_{in} \times C_{out}$
   对于深度可分离卷积：
   $\text{逐深度成本} \approx H \times W \times K \times K \times C_{in} \quad (\text{逐深度步骤})$
   $\text{逐点成本} \approx H \times W \times 1 \times 1 \times C_{in} \times C_{out} \quad (\text{逐点步骤})$
   $\text{可分离成本} = \text{逐深度成本} + \text{逐点成本}$
   $\text{可分离成本} \approx H \times W \times C_{in} \times (K \times K + C_{out})$
   相较于标准卷积，其减少系数大致为：
   $\frac{\text{可分离成本}}{\text{标准成本}} \approx \frac{H \times W \times C_{in} \times (K \times K + C_{out})}{H \times W \times K \times K \times C_{in} \times C_{out}} = \frac{1}{C_{out}} + \frac{1}{K^2}$
   对于像$K=3$和较大的$C_{out}$等典型值，成本降低通常在$8-9 \times$左右。

   > 标准卷积和深度可分离卷积操作的比较。可分离版本将过程分解为两个计算成本较低的步骤。
2. **组卷积：** 最初在AlexNet中提出以应对内存限制，组卷积将输入通道分为几个组。卷积随后在每个组内独立进行。如果将$C_{in}$个通道分成$G$个组，则每个卷积在$C_{in}/G$个输入通道上操作以生成$C_{out}/G$个输出通道。这些通道随后被连接起来。这将参数 (parameter)数量和计算量减少了$G$倍。深度卷积是一种极端情况，其中组数等于输入通道数（$G=C_{in}$）。
3. **通道混洗：** 在ShuffleNet中着重使用了此操作，它有助于在使用组卷积时实现通道组之间的信息流通。在组卷积之后，输出特征图中的通道在馈送到下一个组卷积之前被“混洗”或重新排列。这确保了下一层可以处理来自前一层不同组的信息，减少了组卷积可能导致的隔离。

### 著名高效架构

#### MobileNet (V1, V2, V3)

MobileNet家族开创了深度可分离卷积的大规模应用。

- **MobileNetV1：** 直接用深度可分离卷积替换了大多数标准卷积。引入了宽度（通道数）和分辨率乘数作为超参数 (parameter) (hyperparameter)，以方便地权衡准确性与延迟/大小。
- **MobileNetV2：** 引入了带线性瓶颈的倒置残差块。残差连接，类似于ResNet，有助于深层网络中的梯度流动。“倒置”结构表示该块首先使用$1 \times 1$卷积来*扩展*通道维度，在扩展空间中应用轻量级的$3 \times 3$逐深度卷积，然后使用$1 \times 1$线性卷积（不带ReLU）将其投影回较小维度。发现这有助于防止窄层中的信息损失。
- **MobileNetV3：** 结合了神经网络 (neural network)架构搜索（NAS）、Squeeze-and-Excitation（SE）模块（一种通道注意力形式）的思路以及更新的块结构（使用h-swish激活函数 (activation function)），以进一步提高准确性和效率。它有“大”和“小”两个版本，针对不同的资源限制。

#### ShuffleNet (V1, V2)

ShuffleNet专注于优化效率，考虑了内存访问成本（MAC）等因素。

- **ShuffleNetV1：** 采用逐点组卷积和通道混洗来降低计算成本，同时维持跨通道组的信息流动。
- **ShuffleNetV2：** 提出了实用高效网络设计的指导方针，认为仅仅最小化FLOPs是不够的。它建议平衡通道宽度，避免过多的组卷积（因为会增加MAC），并最小化逐元素操作。最终的架构使用了通道分割机制，并仔细平衡操作以在实践中获得更快的速度。

#### EfficientNet

EfficientNet引入了复合缩放。它没有独立缩放网络维度（深度、宽度、分辨率），而是提出了一种有原则的方法，使用复合系数$\phi$来联合缩放它们。从一个良好的基线架构（EfficientNet-B0，通过NAS发现）开始，它共同缩放深度（$\alpha^\phi$）、宽度（$\beta^\phi$）和分辨率（$\gamma^\phi$），其中$\alpha, \beta, \gamma$是通过网格搜索发现的常数，满足$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$。这种平衡缩放使EfficientNet能够以明显更少的参数和FLOPs，在各种计算预算（B0到B7）下，达到与以往模型相比领先的准确度。

### 设计考量

选择或设计高效架构时，请考虑：

- **目标平台：** CPU、GPU、移动GPU、DSP或专用硬件（如TPU、NPU）具有不同的性能特点。为一个平台优化的架构可能不适合另一个平台（例如，在某些移动硬件上，MAC可能比FLOPs更具限制性）。
- **延迟与吞吐量 (throughput)：** 实时推理 (inference)速度（延迟）是否非常重要，还是高效处理大量批次（吞吐量）更重要？
- **准确度要求：** 为了效率可以牺牲多少准确度？MobileNetV3-Small与MobileNetV3-Large，或EfficientNet-B0与EfficientNet-B7，代表了这种权衡曲线上的不同点。
- **内存带宽：** 像通道混洗或连接这样的操作可能会占用大量内存，从而影响速度。

设计高效架构是一个活跃的研究方向。通过了解深度可分离卷积、组卷积等构建块，以及MobileNet、ShuffleNet和EfficientNet中使用的设计原则，您可以更好地选择或调整模型，以便在资源受限的环境中部署，补充本章前面讨论的模型压缩技术。

## 参考资料

- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://doi.org/10.1109/CVPR.2018.00474) — Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen (2018)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Publisher: IEEE; Pages: 4510-4520; DOI: [10.1109/CVPR.2018.00474](https://doi.org/10.1109/CVPR.2018.00474)
  提出了带有线性瓶颈的倒残差块，这是进一步提高移动卷积网络效率和准确性的关键创新。
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf) — Mingxing Tan and Quoc V. Le (2019)
  Journal: Proceedings of the 36th International Conference on Machine Learning (ICML); Publisher: PMLR (Proceedings of Machine Learning Research); Volume: 97; Pages: 6105-6114; DOI: [10.5555/3306121.3306233](https://doi.org/10.5555/3306121.3306233)
  提出了一种规范的复合缩放方法，以统一缩放网络深度、宽度和分辨率，以显著更少的资源实现了最先进的准确性。
