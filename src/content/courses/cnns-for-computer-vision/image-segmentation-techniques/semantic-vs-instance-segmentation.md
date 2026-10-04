---
course: "cnns-for-computer-vision"
chapter: "image-segmentation-techniques"
lesson: "semantic-vs-instance-segmentation"
sourceId: 2569
sourceUrl: "https://apxml.com/zh/courses/cnns-for-computer-vision/chapter-4-image-segmentation-techniques/semantic-vs-instance-segmentation"
title: "语义分割与实例分割"
description: "阐明语义分割（像素分类）和实例分割（识别物体实例）的不同之处。"
order: 1
plots: []
sourceHash: "73e43e035465c9a0c20750188e4010a8a8c32baac9d6ec790b4318bd1af86a16"
sourceCorrections: []
---

如本章引言所述，图像分割使计算机视觉对图像的理解，比分类或目标检测更为精细。图像分割不同于为图像分配单一标签或在物体周围绘制边界框，它为*每个像素*分配一个类别。这种密集预测可提供场景中物体和区域的精确轮廓。然而，在这个目标之下，物体处理方式存在重要区别，这引出了两种主要的分割任务。

### 语义分割

语义分割是将图像中每个像素分类到预定义类别集合中的任务。可以将其视为为每个像素分配一个语义标签（例如“道路”、“天空”、“人物”、“汽车”、“建筑”）。输出通常是与输入图像大小相同的图，其中每个像素的值对应其预测类别。

想象一幅图像，其中包含多辆汽车在路上。语义分割模型的目标是，将属于*任何*汽车的所有像素标记 (token)为“汽车”，所有道路像素标记为“道路”，等等。它知道每个像素位置存在*什么*，但不区分同一物体类别的不同实例。所有汽车都属于单一语义类别“汽车”。

**特点：**

- 为每个像素分配一个类别标签。
- 不区分同一类别的不同物体。
- 输出是类别标签的密集图。

**应用：** 语义分割对于场景理解很有价值，尤其是在整体环境和区域类型很重要的情况下。例子包括：

- 自动驾驶：识别可驾驶区域（道路）、人行道、车道标记和静态障碍物。
- 医学图像分析：在MRI或CT等扫描中描绘不同类型的组织、器官或异常。
- 地理空间分析：从卫星图像中分类土地覆盖类型（水体、森林、城市区域）。

### 实例分割

实例分割将此任务向前推进了一步。它不仅对每个像素进行分类，还能识别每个像素属于*哪个物体实例*。回到多辆汽车在路上的例子，实例分割模型会将属于第一辆汽车的所有像素识别为“car\_instance\_1”，将第二辆汽车的所有像素识别为“car\_instance\_2”，并适当地将道路像素标记 (token)为“road”。

本质上，实例分割同时进行物体检测和语义分割。它找到单个物体，并为每个检测到的实例提供精确的像素级遮罩。

**特点：**

- 为属于可数物体（“事物”）的像素分配类别标签*和*唯一实例ID。
- 区分同一类别的不同物体。
- 对于背景类别（例如天空、道路、草地等“背景”），其行为类似于语义分割。
- 输出通常包括边界框（类似于物体检测）以及每个检测到的物体实例的像素遮罩。

**应用：** 实例分割在需要与场景中的单个物体进行交互或分析时非常有用。例子包括：

- 机器人：使机械臂能够从一组相似物体中识别并抓取特定物体。
- 自动驾驶：跟踪单个车辆或行人以进行运动预测。
- 照片/视频编辑：允许用户精确选择和操作特定物体。
- 生物成像：计数和测量单个细胞或生物体。

### 可视化差异

核心差异在于同一类别的单个物体是否被视为不同的实体。语义分割将它们归为同一类别标签下，而实例分割则将它们分离。

> 对比展示了语义分割和实例分割对于包含多个同类别物体的图像的不同目标和输出。

理解这种区别非常重要，因为网络架构、损失函数 (loss function)和评估指标在语义分割和实例分割任务之间通常不同。实例分割通常被认为是一个更复杂的问题，因为它既需要正确的分类，也需要精确描绘潜在重叠的物体实例。在本章后续内容中，我们将考察适用于这两种任务的架构，从常用于语义分割的初始方法开始，然后转向能够进行实例级预测的方法。

## 参考资料

- [Fully Convolutional Networks for Semantic Segmentation](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) — Jonathan Long, Evan Shelhamer, Trevor Darrell (2015)
  Journal: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR); Pages: 3431-3440; DOI: [10.1109/CVPR.2015.7298965](https://doi.org/10.1109/CVPR.2015.7298965)
  介绍了基础的FCN架构，开创了语义分割的端到端学习方法。
- [Mask R-CNN](https://research.fb.com/wp-content/uploads/2017/08/maskrcnn.pdf) — Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross B. Girshick (2017)
  Journal: 2017 IEEE International Conference on Computer Vision (ICCV); Publisher: IEEE; Pages: 2980-2988; DOI: [10.1109/ICCV.2017.322](https://doi.org/10.1109/ICCV.2017.322)
  提出了Mask R-CNN，这是一种广泛采用的实例分割架构，通过增加一个预测对象掩码的并行分支来扩展Faster R-CNN。
- [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://doi.org/10.1007/978-3-030-01234-2_49) — Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam (2018)
  Journal: European Conference on Computer Vision (ECCV); Publisher: Springer International Publishing; Pages: 801-818; DOI: [10.1007/978-3-030-01234-2_49](https://doi.org/10.1007/978-3-030-01234-2_49)
  介绍了DeepLabv3+，这是一种先进的语义分割模型，利用空洞卷积和改进的编码器-解码器结构。
