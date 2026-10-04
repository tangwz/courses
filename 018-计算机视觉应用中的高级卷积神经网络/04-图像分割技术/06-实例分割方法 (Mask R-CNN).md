# 实例分割方法 (Mask R-CNN)

来源：[原文](https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-4-image-segmentation-techniques/instance-segmentation-mask-r-cnn)

[返回章节目录](README.md) · [返回课程目录](../README.md)

语义分割为图像中的每个像素分配一个类别标签（如“汽车”、“人物”、“道路”），而实例分割则在此基础上更进一步。它的目标是识别并描绘图像中每个不同的*目标实例*。因此，它不会将所有属于人的像素标记 (token)为“人”，而是为“人1”、“人2”和“人3”输出单独的掩码。这提供了对场景更全面的理解，能够分离同一类别中重叠的对象。

实例分割本质上结合了目标检测（用边界框定位单个目标）和语义分割（对像素进行分类）的元素。针对此任务，Mask R-CNN是一种主要且高效的方法之一。

### Mask R-CNN：扩展目标检测以进行分割

Mask R-CNN由Facebook AI研究院（FAIR）的研究人员开发，直接构建在Faster R-CNN目标检测框架之上。回顾一下，Faster R-CNN是一个两阶段检测器：

1. 区域提议网络（RPN）提议候选目标边界框（感兴趣区域，即RoIs）。
2. 第二阶段使用RoIPool为每个RoI提取特征，然后利用这些特征进行分类（盒子里是什么目标？）和边界框回归（优化框的坐标）。

Mask R-CNN扩展了第二阶段，在现有的分类和边界框回归分支之外，增加了第三个并行分支，用于为每个RoI预测分割掩码。

> Mask R-CNN 概述。它在标准Faster R-CNN头部（蓝色）之外，增加了一个并行的掩码预测分支（黄色），用于处理来自RoIAlign的特征。

### 掩码预测头部

Mask R-CNN中的核心新增部分是这个掩码头部。它通常被实现为一个应用于每个RoI的小型全卷积网络（FCN）。

- **输入：** 它接收为RoI生成的池化特征图（使用RoIAlign，接下来会讨论）。
- **架构：** 它通常由多个卷积层组成，然后是一个最终层，该层为每个类别输出一个二进制掩码。对于大小为 $M \times M$ 的输入RoI，掩码头部可能会输出一个 $K \times M' \times M'$ 的张量，其中 $K$ 是类别数，而 $M'$ 是稍大的分辨率（例如，$28 \times 28$），以捕捉比特征图分辨率更精细的细节。
- **输出：** 在推理 (inference)时，如果RoI被分类为属于类别 $c$，则会选中相应的第 $c$ 个输出掩码，将其调整到RoI的尺寸，并进行阈值处理以生成最终的二进制实例掩码。

### RoIAlign：用于掩码的精确特征提取

Mask R-CNN引入了一项重要的创新，即RoIAlign。Faster R-CNN中使用的原始RoIPool操作涉及量化 (quantization)步骤。当将RoI（具有连续坐标）映射到特征图的离散网格上时，RoIPool会对坐标进行四舍五入。然后它将RoI划分为空间区域（bin），并在每个区域内进行最大池化。尽管这对分类和边界框回归来说效果足够好，但空间量化不准确性对预测像素精确的掩码有害。四舍五入造成的小偏差会显著降低分割边界的质量。

RoIAlign避免了这种量化。它没有对RoI边界进行四舍五入，而是使用双线性插值来计算每个RoI区域内四个规则采样位置处输入特征的精确值。这些采样值随后被聚合（通常通过最大池化或平均池化）以生成该区域的池化特征图。此过程保留了精确的空间位置信息，使得提取的特征与原始图像区域之间的对齐 (alignment)效果更好，这对于高质量的实例分割来说非常重要。

### 训练 Mask R-CNN

Mask R-CNN采用多任务损失函数 (loss function)进行端到端训练。每个采样RoI上的总损失 $L$ 是分类损失 $L_{cls}$、边界框回归损失 $L_{box}$ 和掩码分割损失 $L_{mask}$ 的和：


$$
L = L_{cls} + L_{box} + L_{mask}
$$


- $L_{cls}$ 和 $L_{box}$ 与Faster R-CNN中使用的相同。
- $L_{mask}$ 通常定义为平均二元交叉熵损失。对于与真实类别 $k$ 关联的给定RoI，$L_{mask}$ 仅在第 $k$ 个掩码上计算（其他掩码输出被忽略）。它衡量预测掩码像素与该特定实例的真实掩码之间的差异。

这种多任务训练使网络能够同时学习对目标进行分类、优化其边界框以及生成详细的分割掩码，所有这些都运用了主干网络的共享卷积特征。

### 推理 (inference)与输出

在推理过程中，Mask R-CNN遵循Faster R-CNN的步骤，通过RPN生成RoIs。对于每个提议的RoI，它预测一个类别标签、一个边界框优化和一个像素级掩码。应用置信度分数阈值来过滤检测结果。基于边界框执行非极大值抑制（NMS）以移除重复检测。对于保留的检测结果，会选中掩码头部预测的相应类别特定掩码，将其调整到最终边界框的尺寸，并通常在0.5处进行阈值处理以得到最终的二进制实例掩码。

输出是一组目标，每个目标都有一个类别标签、一个置信度分数、一个边界框，以及一个精确的像素级分割掩码，用于识别该特定实例。

Mask R-CNN在发布时在实例分割基准上表现出领先水平，并且仍然是一个强大且广泛使用的基准方法。它的设计巧妙地将目标检测和分割结合到一个单一的、可训练的框架中，为实例级场景理解的许多后续发展开辟了道路。尽管存在其他方法，包括为速度设计的单阶段方法（如YOLACT或SOLO），理解Mask R-CNN为解决实例分割问题提供了坚实根基。

## 参考资料

- [Mask R-CNN](https://doi.org/10.1109/ICCV.2017.322) — Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick (2017)
  Journal: Proceedings of the IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 2961-2969; DOI: [10.1109/ICCV.2017.322](https://doi.org/10.1109/ICCV.2017.322)
  介绍Mask R-CNN的基础论文，详细阐述了其架构、RoIAlign层以及用于高质量实例分割的多任务损失。
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://doi.org/10.1109/TPAMI.2016.2577031) — Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun (2016)
  Journal: IEEE Transactions on Pattern Analysis and Machine Intelligence; Publisher: IEEE; Volume: 39; Pages: 1137-1149; DOI: [10.1109/TPAMI.2016.2577031](https://doi.org/10.1109/TPAMI.2016.2577031)
  关于Faster R-CNN的开创性论文，Mask R-CNN正是基于此两阶段目标检测框架进行实例分割的扩展。
- [Fully Convolutional Networks for Semantic Segmentation](https://doi.org/10.1109/CVPR.2015.7298965) — Jonathan Long, Evan Shelhamer, and Trevor Darrell (2015)
  Journal: IEEE Conference Publication (CVPR); Publisher: IEEE; Pages: 3431-3440; DOI: [10.1109/CVPR.2015.7298965](https://doi.org/10.1109/CVPR.2015.7298965)
  介绍了全卷积网络（FCN），它构成了Mask R-CNN中掩码预测头的基础。
- [A Survey of Instance Segmentation: Past, Present and Future](https://doi.org/10.1016/j.patcog.2020.107412) — Yunlong He, Mingyuan Mao, Chenglu Wen, Yan Zeng, Kaixiang Yang, and Zongyuan Ge (2020)
  Journal: Pattern Recognition; Publisher: Elsevier; Volume: 105; Pages: 107412; DOI: [10.1016/j.patcog.2020.107412](https://doi.org/10.1016/j.patcog.2020.107412)
  一份全面的综述，概述了实例分割方法，包括Mask R-CNN，并讨论了当前的趋势和挑战。

---

[上一节](05-DeepLab%20%E7%B3%BB%E5%88%97%EF%BC%9A%E7%A9%BA%E6%B4%9E%E7%A9%BA%E9%97%B4%E9%87%91%E5%AD%97%E5%A1%94%E6%B1%A0%E5%8C%96.md) · [下一节](07-%E5%88%86%E5%89%B2%E7%9A%84%E8%AF%84%E4%BC%B0%E6%8C%87%E6%A0%87.md)
